from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pre_peft.config import LoadedConfig
from pre_peft.data.task_construction import read_task_paths
from pre_peft.diagnostics.embeddings import embedding_path, load_embeddings
from pre_peft.diagnostics.reference_probes import diagnostic_path
from pre_peft.runtime import RunContext, run_atomic_unit


def _nearest_distances(train_x: np.ndarray, test_x: np.ndarray, k_values: tuple[int, ...]) -> dict[int, np.ndarray]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError("Distance controls require the package extra: pip install -e '.[full]'") from exc
    maximum = min(max(k_values), len(train_x))
    distances, _ = NearestNeighbors(n_neighbors=maximum, metric="cosine").fit(train_x).kneighbors(test_x)
    return {k: distances[:, : min(k, maximum)].mean(axis=1) for k in k_values}


def control_path(context: RunContext, task: str, seed: int) -> Path:
    return context.run_dir / "distance_controls" / task / f"seed={seed}.csv"


def evaluate_distance_controls(protocol: LoadedConfig, context: RunContext) -> dict[str, Path]:

    task_paths = read_task_paths(context)
    k_values = tuple(sorted(set(protocol.distance_controls.k_values)))
    outputs: dict[str, Path] = {}
    for task, task_path in sorted(task_paths.items()):
        table = pd.read_csv(task_path)
        sample_ids, embeddings = load_embeddings(embedding_path(context, task))
        positions = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        for seed in protocol.fold_evaluation.seeds:
            diagnostics = pd.read_csv(diagnostic_path(context, task, int(seed)))
            output = control_path(context, task, int(seed))

            def worker(temporary: Path) -> None:
                frames: list[pd.DataFrame] = []
                for fold, held_out in diagnostics.groupby("fold", sort=True):
                    test_ids = held_out.sample_id.astype(str).tolist()
                    test_indices = np.array([positions[item] for item in test_ids], dtype=int)
                    train_indices = np.array([index for index, item in enumerate(sample_ids) if item not in set(test_ids)], dtype=int)
                    values = _nearest_distances(embeddings[train_indices], embeddings[test_indices], k_values)
                    frame = held_out[["sample_id", "y", "task", "fold", "seed"]].copy()
                    frame["exact_nearest_neighbor_distance"] = values[min(k_values)]
                    for k_value in k_values:
                        frame[f"knn_embedding_distance_k{k_value}"] = values[k_value]
                    frame["knn_embedding_distance"] = values[20 if 20 in values else max(k_values)]
                    frames.append(frame)
                pd.concat(frames, ignore_index=True).sort_values(["fold", "sample_id"]).to_csv(temporary, index=False)

            run_atomic_unit(context, f"distance-controls.{task}.seed-{seed}", {"task": task, "seed": int(seed), "k_values": list(k_values)}, output, worker)
            outputs[f"{task}:{seed}"] = output
    context.emit("distance_controls_completed", units=len(outputs))
    return outputs
