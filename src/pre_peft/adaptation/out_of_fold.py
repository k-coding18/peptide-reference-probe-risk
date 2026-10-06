from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pre_peft.adaptation.lora import train_and_predict
from pre_peft.config import LoadedConfig
from pre_peft.runtime import RunContext, atomic_write_csv, run_atomic_unit


def _splitter() -> Any:
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        raise RuntimeError("Out-of-fold adaptation requires the package extra: pip install -e '.[full]'") from exc
    return StratifiedKFold


def oof_prediction_path(context: RunContext, task: str, target_model: str, seed: int, fold: int | None = None) -> Path:
    base = context.run_dir / "target_adaptation" / "oof" / task / target_model / f"seed={seed}"
    return base / (f"fold={fold}.csv" if fold is not None else "predictions.csv")


def evaluate_target_adaptation(task_paths: dict[str, Path], protocol: LoadedConfig, context: RunContext) -> dict[str, Path]:
    StratifiedKFold = _splitter()
    outputs: dict[str, Path] = {}
    for task, task_path in sorted(task_paths.items()):
        table = pd.read_csv(task_path)
        labels = table.y.astype(int).to_numpy()
        for seed in protocol.fold_evaluation.seeds:
            folds = list(StratifiedKFold(n_splits=protocol.fold_evaluation.fold_count, shuffle=True, random_state=int(seed)).split(table, labels))
            for target_model in protocol.target_adaptation.target_models:
                fold_outputs: list[Path] = []
                for fold, (train_indices, test_indices) in enumerate(folds):
                    output = oof_prediction_path(context, task, target_model, int(seed), fold)

                    def worker(temporary: Path, train_indices: Any = train_indices, test_indices: Any = test_indices, fold: int = fold) -> None:
                        prediction = train_and_predict(
                            table.iloc[train_indices].reset_index(drop=True),
                            table.iloc[test_indices].reset_index(drop=True),
                            protocol,
                            target_model,
                            int(seed),
                            fold,
                        )
                        prediction.to_csv(temporary, index=False)

                    run_atomic_unit(
                        context,
                        f"target-adaptation.{task}.{target_model}.seed-{seed}.fold-{fold}",
                        {"task": task, "target_model": target_model, "seed": int(seed), "fold": fold},
                        output,
                        worker,
                    )
                    fold_outputs.append(output)
                combined = pd.concat([pd.read_csv(path) for path in fold_outputs], ignore_index=True)
                if combined.sample_id.duplicated().any() or len(combined) != len(table):
                    raise RuntimeError(f"Invalid OOF coverage for {task}, {target_model}, seed={seed}")
                aggregate = oof_prediction_path(context, task, target_model, int(seed))
                atomic_write_csv(aggregate, combined.sort_values("sample_id"))
                outputs[f"{task}:{target_model}:{seed}"] = aggregate
    context.emit("target_adaptation_completed", units=len(outputs))
    return outputs
