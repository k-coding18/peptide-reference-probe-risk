from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pre_peft.adaptation.lora import train_and_predict
from pre_peft.config import LoadedConfig
from pre_peft.data.task_construction import read_task_paths
from pre_peft.diagnostics.embeddings import embedding_path, load_embeddings
from pre_peft.diagnostics.reference_probes import diagnostic_frame_against_reference
from pre_peft.evaluation.metrics import metric_record
from pre_peft.runtime import RunContext, atomic_write_csv, run_atomic_unit


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _lcs_length(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    for left_residue in left:
        current = [0]
        for index, right_residue in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_residue == right_residue else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def alignment_similarity(left: str, right: str) -> float:

    if not left or not right:
        return 0.0
    try:
        from Bio.Align import PairwiseAligner

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = 0.0
        aligner.extend_gap_score = 0.0
        return float(aligner.score(left, right)) / max(len(left), len(right))
    except ImportError:
        return _lcs_length(left, right) / max(len(left), len(right))


def build_similarity_components(sequences: list[str], threshold: float) -> tuple[list[int], int]:
    union_find = UnionFind(len(sequences))
    edges = 0
    lengths = [len(sequence) for sequence in sequences]
    for left in range(len(sequences)):
        for right in range(left + 1, len(sequences)):
            if min(lengths[left], lengths[right]) / max(lengths[left], lengths[right]) < threshold:
                continue
            if alignment_similarity(sequences[left], sequences[right]) >= threshold:
                union_find.union(left, right)
                edges += 1
    roots = [union_find.find(index) for index in range(len(sequences))]
    mapping = {root: group for group, root in enumerate(sorted(set(roots)))}
    return [mapping[root] for root in roots], edges


def choose_component_split(
    groups: list[int],
    labels: list[int] | np.ndarray,
    test_fraction: float,
    seed: int,
    trials: int,
    minimum_train_positives: int,
    minimum_test_positives: int,
) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=int)
    group_statistics = {
        group: {"n": int(sum(member == group for member in groups)), "pos": int(labels_array[np.asarray(groups) == group].sum())}
        for group in sorted(set(groups))
    }
    random = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_score = float("inf")
    target_size = len(groups) * test_fraction
    for _ in range(trials):
        order = random.permutation(list(group_statistics))
        selected: list[int] = []
        selected_size = selected_positive = 0
        for group in order:
            statistics = group_statistics[int(group)]
            if selected_size < target_size or selected_positive < minimum_test_positives:
                selected.append(int(group))
                selected_size += statistics["n"]
                selected_positive += statistics["pos"]
        mask = np.isin(np.asarray(groups), selected)
        train_positive = int(labels_array[~mask].sum())
        test_positive = int(labels_array[mask].sum())
        if not mask.any() or mask.all() or train_positive < minimum_train_positives or test_positive < minimum_test_positives:
            continue
        score = abs(mask.mean() - test_fraction) + abs(test_positive / mask.sum() - labels_array.mean())
        if score < best_score:
            best_mask, best_score = mask, score
    if best_mask is None:
        raise RuntimeError("No feasible component-held-out split satisfies the configured positive-count requirements")
    return best_mask


def cross_partition_similarity_audit(split: pd.DataFrame, threshold: float) -> tuple[int, float]:

    train = split.loc[split["split"] == "train", "sequence"].astype(str).tolist()
    test = split.loc[split["split"] == "test", "sequence"].astype(str).tolist()
    violations = 0
    maximum_similarity = 0.0
    for test_sequence in test:
        for train_sequence in train:
            longest_length = max(len(test_sequence), len(train_sequence))
            if longest_length == 0:
                continue
            length_ratio = min(len(test_sequence), len(train_sequence)) / longest_length
            similarity = 0.0
            if length_ratio > maximum_similarity or length_ratio >= threshold:
                similarity = alignment_similarity(test_sequence, train_sequence)
                maximum_similarity = max(maximum_similarity, similarity)
            if length_ratio >= threshold and similarity >= threshold:
                violations += 1
    return violations, maximum_similarity


def _distance_scores(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized_train = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    normalized_test = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12)
    distances = 1.0 - normalized_test @ normalized_train.T
    return distances.min(axis=1), np.sort(distances, axis=1)[:, : min(20, distances.shape[1])].mean(axis=1)


def similarity_result_path(context: RunContext, task: str, seed: int, repeat: int) -> Path:
    return context.run_dir / "similarity_holdout" / task / f"seed={seed}" / f"repeat={repeat}.csv"


def evaluate_similarity_holdout(protocol: LoadedConfig, context: RunContext) -> dict[str, Path]:

    if not protocol.similarity_holdout.enabled:
        return {}
    task_paths = read_task_paths(context)
    outputs: dict[str, Path] = {}
    for task, task_path in sorted(task_paths.items()):
        table = pd.read_csv(task_path)
        sample_ids, embeddings = load_embeddings(embedding_path(context, task))
        if not np.array_equal(sample_ids, table.sample_id.astype(str).to_numpy()):
            raise RuntimeError(f"Embedding sample order mismatch for similarity evaluation: {task}")
        for repeat in range(protocol.similarity_holdout.repeats):
            for seed in protocol.similarity_holdout.model_seeds:
                output = similarity_result_path(context, task, int(seed), repeat)

                def worker(temporary: Path, seed: int = int(seed), repeat: int = repeat) -> None:
                    groups, edge_count = build_similarity_components(table.sequence.astype(str).tolist(), protocol.similarity_holdout.threshold)
                    test_mask = choose_component_split(
                        groups,
                        table.y.to_numpy(),
                        protocol.similarity_holdout.test_fraction,
                        protocol.task_construction.sampling_seed + int(round(protocol.similarity_holdout.threshold * 1000)) + repeat,
                        protocol.similarity_holdout.component_assignment_trials,
                        protocol.similarity_holdout.minimum_train_positives,
                        protocol.similarity_holdout.minimum_test_positives,
                    )
                    split = table.copy()
                    split["similarity_group"] = groups
                    split["split"] = np.where(test_mask, "test", "train")
                    violations, maximum_similarity = cross_partition_similarity_audit(split, protocol.similarity_holdout.threshold)
                    if violations:
                        raise RuntimeError(f"Similarity-held-out split has {violations} cross-partition edges")
                    train_table = split.loc[~test_mask].reset_index(drop=True)
                    test_table = split.loc[test_mask].reset_index(drop=True)
                    diagnostics = diagnostic_frame_against_reference(
                        train_table,
                        test_table,
                        embeddings[~test_mask],
                        embeddings[test_mask],
                        protocol,
                        seed,
                        0,
                    )
                    exact_distance, knn_distance = _distance_scores(embeddings[~test_mask], embeddings[test_mask])
                    diagnostics["exact_nearest_neighbor_distance"] = exact_distance
                    diagnostics["knn_embedding_distance"] = knn_distance
                    prediction = train_and_predict(
                        train_table,
                        test_table,
                        protocol,
                        "focal_lora",
                        seed,
                        fold=0,
                        record_training_loss_audit=True,
                    )
                    merged = diagnostics.merge(prediction, on=["sample_id", "y", "task", "fold", "seed"], validate="one_to_one")
                    merged["high_loss"] = (merged.loss >= merged.loss.quantile(1.0 - protocol.fold_evaluation.high_loss_fraction)).astype(int)
                    rows: list[dict[str, object]] = []
                    for endpoint, target in (("fixed_threshold_error", merged.fixed_threshold_error), ("high_loss", merged.high_loss)):
                        for score in ["local_probe_score", "probe_only_score", "exact_nearest_neighbor_distance", "knn_embedding_distance", "random_score"]:
                            rows.append(
                                {
                                    "task": task,
                                    "seed": seed,
                                    "repeat": repeat,
                                    "target_model": "focal_lora",
                                    "endpoint": endpoint,
                                    "score": score,
                                    "n": len(merged),
                                    "similarity_threshold": protocol.similarity_holdout.threshold,
                                    "component_count": len(set(groups)),
                                    "component_edge_count": edge_count,
                                    "cross_edge_violations": violations,
                                    "max_test_to_train_alignment_similarity": maximum_similarity,
                                    **metric_record(target, merged[score], protocol.fold_evaluation.top_risk_fraction),
                                }
                            )
                    pd.DataFrame(rows).to_csv(temporary, index=False)
                    atomic_write_csv(output.parent / f"split.repeat={repeat}.csv", split)
                    atomic_write_csv(
                        output.parent / f"training_audit.repeat={repeat}.csv",
                        prediction.loc[:, ["initial_training_loss", "final_training_loss", "training_loss_decreased"]]
                        .drop_duplicates()
                        .assign(task=task, seed=seed, repeat=repeat),
                    )

                run_atomic_unit(
                    context,
                    f"similarity-holdout.{task}.seed-{seed}.repeat-{repeat}",
                    {"task": task, "seed": int(seed), "repeat": repeat, "threshold": protocol.similarity_holdout.threshold},
                    output,
                    worker,
                )
                outputs[f"{task}:{seed}:{repeat}"] = output
    context.emit("similarity_holdout_completed", units=len(outputs))
    return outputs
