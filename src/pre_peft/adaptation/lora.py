from __future__ import annotations

from typing import Any
import gc
import os
import random

import numpy as np
import pandas as pd

from pre_peft.config import LoadedConfig
from pre_peft.model_source import local_model_root, local_pretrained_kwargs


def _dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("LoRA adaptation requires the package extra: pip install -e '.[full]'") from exc
    return torch, functional, LoraConfig, TaskType, get_peft_model, (DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer)


def _class_weights(labels: np.ndarray, torch: Any, device: str) -> Any:
    counts = np.maximum(np.bincount(labels.astype(int), minlength=2).astype(float), 1.0)
    return torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)


def _loss(logits: Any, labels: Any, target_model: str, class_weights: Any, functional: Any, gamma: float) -> Any:
    if target_model == "balanced_lora":
        return functional.cross_entropy(logits, labels, weight=class_weights)
    if target_model == "focal_lora":
        # The reported objective includes class weights in the focal modulation.
        cross_entropy = functional.cross_entropy(logits, labels, reduction="none", weight=class_weights)
        return (((1.0 - torch_exp_neg(cross_entropy)) ** gamma) * cross_entropy).mean()
    return functional.cross_entropy(logits, labels)


def torch_exp_neg(values: Any) -> Any:
    import torch

    return torch.exp(-values)


def configure_determinism(torch: Any, seed: int) -> dict[str, object]:

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "seed": int(seed),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


def training_loss_decreased(initial_loss: float, final_loss: float) -> bool:

    return bool(np.isfinite(initial_loss) and np.isfinite(final_loss) and final_loss < initial_loss)


def _mean_training_objective(model: Any, loader: Any, target_model: str, class_weights: Any, functional: Any, gamma: float, device: str) -> float:
    was_training = bool(model.training)
    model.eval()
    values: list[float] = []
    with __import__("torch").no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items() if key != "sample_id"}
            logits = model(**batch).logits
            values.append(float(_loss(logits, batch["labels"], target_model, class_weights, functional, gamma).detach().cpu()))
    if was_training:
        model.train()
    return float(np.mean(values))


def train_and_predict(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    protocol: LoadedConfig,
    target_model: str,
    seed: int,
    fold: int | None,
    record_training_loss_audit: bool = False,
) -> pd.DataFrame:

    torch, functional, LoraConfig, TaskType, get_peft_model, extras = _dependencies()
    DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer = extras

    class PeptideDataset(Dataset):
        def __init__(self, table: pd.DataFrame, tokenizer: Any) -> None:
            self.table = table.reset_index(drop=True)
            self.tokenizer = tokenizer

        def __len__(self) -> int:
            return len(self.table)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.table.iloc[index]
            encoded = self.tokenizer(
                str(row.sequence),
                truncation=True,
                max_length=protocol.model.maximum_sequence_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(int(row.y), dtype=torch.long)
            item["sample_id"] = str(row.sample_id)
            return item

    effective_seed = seed + (fold or 0)
    configure_determinism(torch, effective_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_root = local_model_root()
    pretrained_kwargs = local_pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), revision=protocol.model.revision, **pretrained_kwargs)
    train_dataset = PeptideDataset(train_table, tokenizer)
    generator = torch.Generator().manual_seed(effective_seed)
    train_loader = DataLoader(train_dataset, batch_size=protocol.target_adaptation.batch_size, shuffle=True, generator=generator)
    training_audit_loader = DataLoader(train_dataset, batch_size=protocol.target_adaptation.batch_size, shuffle=False)
    test_loader = DataLoader(PeptideDataset(test_table, tokenizer), batch_size=protocol.target_adaptation.batch_size, shuffle=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        str(model_root),
        revision=protocol.model.revision,
        num_labels=2,
        **pretrained_kwargs,
    )
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=protocol.target_adaptation.lora_rank,
            lora_alpha=protocol.target_adaptation.lora_alpha,
            lora_dropout=protocol.target_adaptation.lora_dropout,
            target_modules=list(protocol.target_adaptation.lora_target_modules),
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.target_adaptation.learning_rate, weight_decay=protocol.target_adaptation.weight_decay)
    class_weights = _class_weights(train_table.y.to_numpy(), torch, device)
    initial_training_loss = float("nan")
    final_training_loss = float("nan")
    if record_training_loss_audit:
        initial_training_loss = _mean_training_objective(
            model,
            training_audit_loader,
            target_model,
            class_weights,
            functional,
            protocol.target_adaptation.focal_gamma,
            device,
        )
    model.train()
    for _ in range(protocol.target_adaptation.epochs):
        for batch in train_loader:
            batch.pop("sample_id")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = _loss(logits, batch["labels"], target_model, class_weights, functional, protocol.target_adaptation.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    if record_training_loss_audit:
        final_training_loss = _mean_training_objective(
            model,
            training_audit_loader,
            target_model,
            class_weights,
            functional,
            protocol.target_adaptation.focal_gamma,
            device,
        )
    loss_decreased = training_loss_decreased(initial_training_loss, final_training_loss)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sample_ids = batch.pop("sample_id")
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["labels"]
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            losses = functional.cross_entropy(logits, labels, reduction="none")
            for sample_id, label, probability, loss in zip(sample_ids, labels.cpu().numpy(), probabilities.cpu().numpy(), losses.cpu().numpy()):
                prediction = int(float(probability) >= protocol.fold_evaluation.fixed_threshold)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "y": int(label),
                        "probability": float(probability),
                        "prediction": prediction,
                        "fixed_threshold_error": int(prediction != int(label)),
                        "loss": float(loss),
                        "task": str(test_table.task.iloc[0]),
                        "fold": int(fold) if fold is not None else -1,
                        "seed": int(seed),
                        "target_model": target_model,
                        "initial_training_loss": initial_training_loss if record_training_loss_audit else None,
                        "final_training_loss": final_training_loss if record_training_loss_audit else None,
                        "training_loss_decreased": loss_decreased if record_training_loss_audit else None,
                    }
                )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)
