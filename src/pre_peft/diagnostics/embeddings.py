from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pre_peft.config import LoadedConfig
from pre_peft.model_source import local_model_root, local_pretrained_kwargs
from pre_peft.runtime import RunContext, run_atomic_unit


def _runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Embedding extraction requires the package extra: pip install -e '.[full]'") from exc
    return torch, AutoModel, AutoTokenizer


def attention_masked_mean(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def embed_sequences(sequences: Sequence[str], protocol: LoadedConfig) -> np.ndarray:
    torch, AutoModel, AutoTokenizer = _runtime_dependencies()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_root = local_model_root()
    pretrained_kwargs = local_pretrained_kwargs()
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), revision=protocol.model.revision, **pretrained_kwargs)
    model = AutoModel.from_pretrained(str(model_root), revision=protocol.model.revision, **pretrained_kwargs).to(device)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), protocol.model.embedding_batch_size):
            encoded = tokenizer(
                list(sequences[start : start + protocol.model.embedding_batch_size]),
                truncation=True,
                max_length=protocol.model.maximum_sequence_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            batches.append(attention_masked_mean(output.last_hidden_state, encoded["attention_mask"]).cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(batches).astype("float32")


def embedding_path(context: RunContext, task: str) -> Path:
    return context.run_dir / "pre_adaptation_diagnostics" / "embeddings" / f"{task}.npz"


def write_embedding_archive(path: Path, sample_ids: Sequence[str], embeddings: np.ndarray) -> None:

    normalized_ids = np.asarray([str(sample_id) for sample_id in sample_ids], dtype=np.str_)
    normalized_embeddings = np.asarray(embeddings, dtype=np.float32)
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, sample_ids=normalized_ids, embeddings=normalized_embeddings)


def ensure_embeddings(task_paths: dict[str, Path], protocol: LoadedConfig, context: RunContext) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for task, task_path in sorted(task_paths.items()):
        output = embedding_path(context, task)

        def worker(temporary: Path, task_path: Path = task_path) -> None:
            table = pd.read_csv(task_path)
            embeddings = embed_sequences(table.sequence.astype(str).tolist(), protocol)
            write_embedding_archive(temporary, table.sample_id.astype(str).tolist(), embeddings)

        run_atomic_unit(
            context,
            f"embedding.{task}",
            {"task": task, "model": protocol.model.identifier, "revision": protocol.model.revision},
            output,
            worker,
        )
        outputs[task] = output
    return outputs


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        sample_ids = archive["sample_ids"].astype(str)
        embeddings = archive["embeddings"].astype("float32")
    return sample_ids, embeddings
