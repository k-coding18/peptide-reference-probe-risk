from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pre_peft.config import LoadedConfig
from pre_peft.diagnostics.embeddings import ensure_embeddings, load_embeddings
from pre_peft.diagnostics.risk_scores import compose_audit_score, compose_candidate_compatible_scores
from pre_peft.runtime import RunContext, run_atomic_unit


def _sklearn() -> tuple[Any, Any, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Reference probes require the package extra: pip install -e '.[full]'") from exc
    return LogisticRegression, StratifiedKFold, NearestNeighbors, StandardScaler


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    return -(probabilities * np.log(probabilities) + (1.0 - probabilities) * np.log(1.0 - probabilities)) / np.log(2.0)


def _fit_minmax(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    return (low, high if high - low >= 1e-12 else low + 1.0)


def _apply_minmax(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    return np.clip((np.asarray(values, dtype=float) - low) / (high - low), 0.0, 1.0)


def _query_distances(train_x: np.ndarray, query_x: np.ndarray, k_values: tuple[int, ...]) -> dict[int, np.ndarray]:
    _, _, NearestNeighbors, _ = _sklearn()
    k_max = min(max(k_values), len(train_x))
    if k_max < 1:
        raise ValueError("A reference fold must contain at least one embedding")
    model = NearestNeighbors(n_neighbors=k_max, metric="cosine").fit(train_x)
    distances, _ = model.kneighbors(query_x)
    return {k: distances[:, : min(k, k_max)].mean(axis=1) for k in k_values}


def _self_distances(reference_x: np.ndarray, k: int) -> np.ndarray:
    _, _, NearestNeighbors, _ = _sklearn()
    if len(reference_x) < 2:
        return np.zeros(len(reference_x), dtype=float)
    model = NearestNeighbors(n_neighbors=min(k + 1, len(reference_x)), metric="cosine").fit(reference_x)
    distances, _ = model.kneighbors(reference_x)
    return distances[:, 1:].mean(axis=1)


def _probe_family(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, protocol: LoadedConfig, seed: int) -> dict[str, np.ndarray]:
    LogisticRegression, StratifiedKFold, _, StandardScaler = _sklearn()
    positive_rate = float(train_y.mean())
    if len(np.unique(train_y)) < 2:
        train_matrix = np.full((len(protocol.reference_probes.regularization_values), len(train_y)), positive_rate)
        query_matrix = np.full((len(protocol.reference_probes.regularization_values), len(query_x)), positive_rate)
    else:
        n_splits = min(protocol.fold_evaluation.fold_count, int(np.bincount(train_y, minlength=2).min()))
        inner = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_x, train_y)) if n_splits >= 2 else []
        train_rows: list[np.ndarray] = []
        query_rows: list[np.ndarray] = []
        for index, c_value in enumerate(protocol.reference_probes.regularization_values):
            train_probabilities = np.full(len(train_y), positive_rate, dtype=float)
            for fold, (fit_indices, held_indices) in enumerate(inner):
                scaler = StandardScaler()
                fit_x = scaler.fit_transform(train_x[fit_indices])
                held_x = scaler.transform(train_x[held_indices])
                model = LogisticRegression(
                    C=float(c_value),
                    max_iter=protocol.reference_probes.maximum_iterations,
                    solver=protocol.reference_probes.solver,
                    class_weight=protocol.reference_probes.class_weight,
                    random_state=seed + 101 * index + fold,
                ).fit(fit_x, train_y[fit_indices])
                train_probabilities[held_indices] = model.predict_proba(held_x)[:, 1]
            scaler = StandardScaler()
            model = LogisticRegression(
                C=float(c_value),
                max_iter=protocol.reference_probes.maximum_iterations,
                solver=protocol.reference_probes.solver,
                class_weight=protocol.reference_probes.class_weight,
                random_state=seed + 307 * index,
            ).fit(scaler.fit_transform(train_x), train_y)
            train_rows.append(train_probabilities)
            query_rows.append(model.predict_proba(scaler.transform(query_x))[:, 1])
        train_matrix = np.vstack(train_rows)
        query_matrix = np.vstack(query_rows)
    train_mean = train_matrix.mean(axis=0)
    query_mean = query_matrix.mean(axis=0)
    return {
        "train_uncertainty": binary_entropy(train_mean),
        "train_variance": train_matrix.var(axis=0),
        "query_probability": query_mean,
        "query_uncertainty": binary_entropy(query_mean),
        "query_variance": query_matrix.var(axis=0),
    }


def diagnostic_frame_against_reference(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    train_x: np.ndarray,
    test_x: np.ndarray,
    protocol: LoadedConfig,
    seed: int,
    fold: int,
) -> pd.DataFrame:

    train_y = train_table.y.astype(int).to_numpy()
    query_y = test_table.y.astype(int).to_numpy()
    k_values = tuple(sorted(set((*protocol.distance_controls.k_values, 20))))
    query_distances = _query_distances(train_x, test_x, k_values)
    density_raw = query_distances[min(20, max(k_values))]
    reference_density = _self_distances(train_x, min(20, max(k_values)))
    _, _, NearestNeighbors, _ = _sklearn()
    neighbor_model = NearestNeighbors(n_neighbors=min(20, len(train_x)), metric="cosine").fit(train_x)
    _, indices = neighbor_model.kneighbors(test_x)
    boundary_raw = (train_y[indices] != query_y[:, None]).mean(axis=1)
    probes = _probe_family(train_x, train_y, test_x, protocol, seed)
    scores = compose_candidate_compatible_scores(
        {
            "embedding_density": density,
            "probe_uncertainty": uncertainty,
            "probe_variance": variance,
        }
        for density, uncertainty, variance in zip(
            _apply_minmax(density_raw, _fit_minmax(reference_density)),
            _apply_minmax(probes["query_uncertainty"], _fit_minmax(probes["train_uncertainty"])),
            _apply_minmax(probes["query_variance"], _fit_minmax(probes["train_variance"])),
        )
    )
    output = test_table[["sample_id", "sequence", "length", "y", "task"]].copy()
    output["fold"] = int(fold)
    output["seed"] = int(seed)
    output["embedding_density"] = _apply_minmax(density_raw, _fit_minmax(reference_density))
    output["probe_probability"] = probes["query_probability"]
    output["probe_uncertainty"] = _apply_minmax(probes["query_uncertainty"], _fit_minmax(probes["train_uncertainty"]))
    output["probe_variance"] = _apply_minmax(probes["query_variance"], _fit_minmax(probes["train_variance"]))
    output["boundary_exposure"] = _apply_minmax(boundary_raw, (0.0, 1.0))
    output["local_probe_score"] = [row["local_probe_score"] for row in scores]
    output["probe_only_score"] = [row["probe_only_score"] for row in scores]
    output["audit_score"] = [
        compose_audit_score(density, boundary, uncertainty, variance)
        for density, boundary, uncertainty, variance in zip(
            output.embedding_density, output.boundary_exposure, output.probe_uncertainty, output.probe_variance
        )
    ]
    output["length_score"] = _apply_minmax(test_table.length.to_numpy(), _fit_minmax(train_table.length.to_numpy()))
    positive_rate = float(train_y.mean())
    output["class_frequency_score"] = np.where(query_y == 1, 1.0 - positive_rate, positive_rate)
    task_offset = int(sha256(str(test_table.task.iloc[0]).encode("utf-8")).hexdigest()[:8], 16)
    output["random_score"] = np.random.default_rng(seed + fold * 10007 + task_offset).random(len(output))
    output["normalization_reference"] = "reference_fold_minmax"
    output["candidate_compatible"] = True
    return output


def diagnostic_path(context: RunContext, task: str, seed: int) -> Path:
    return context.run_dir / "pre_adaptation_diagnostics" / "foldwise" / task / f"seed={seed}.csv"


def run_diagnostics(task_paths: dict[str, Path], protocol: LoadedConfig, context: RunContext) -> dict[str, Path]:

    _, StratifiedKFold, _, _ = _sklearn()
    embedding_paths = ensure_embeddings(task_paths, protocol, context)
    outputs: dict[str, Path] = {}
    for task, task_path in sorted(task_paths.items()):
        table = pd.read_csv(task_path)
        sample_ids, embeddings = load_embeddings(embedding_paths[task])
        if not np.array_equal(sample_ids, table.sample_id.astype(str).to_numpy()):
            raise RuntimeError(f"Embedding sample order mismatch for {task}")
        labels = table.y.astype(int).to_numpy()
        for seed in protocol.fold_evaluation.seeds:
            output = diagnostic_path(context, task, int(seed))

            def worker(temporary: Path, seed: int = int(seed)) -> None:
                folds = StratifiedKFold(n_splits=protocol.fold_evaluation.fold_count, shuffle=True, random_state=seed)
                frames: list[pd.DataFrame] = []
                for fold, (train_indices, test_indices) in enumerate(folds.split(table, labels)):
                    frames.append(
                        diagnostic_frame_against_reference(
                            table.iloc[train_indices].reset_index(drop=True),
                            table.iloc[test_indices].reset_index(drop=True),
                            embeddings[train_indices],
                            embeddings[test_indices],
                            protocol,
                            seed,
                            fold,
                        )
                    )
                pd.concat(frames, ignore_index=True).sort_values(["fold", "sample_id"]).to_csv(temporary, index=False)

            run_atomic_unit(context, f"diagnostics.{task}.seed-{seed}", {"task": task, "seed": int(seed)}, output, worker)
            outputs[f"{task}:{seed}"] = output
    context.emit("diagnostics_completed", units=len(outputs))
    return outputs
