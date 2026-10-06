from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pre_peft.config import LoadedConfig
from pre_peft.evaluation.metrics import average_precision, binary_auroc, brier_score, expected_calibration_error, high_loss_endpoint, metric_record, risk_coverage_curve
from pre_peft.evaluation.task_statistics import paired_task_intervals, risk_coverage_task_intervals, task_cluster_intervals
from pre_peft.runtime import RunContext, atomic_write_csv, atomic_write_json


SCORES = (
    "audit_score",
    "local_probe_score",
    "probe_only_score",
    "probe_uncertainty",
    "probe_variance",
    "embedding_density",
    "exact_nearest_neighbor_distance",
    "knn_embedding_distance",
    "class_frequency_score",
    "length_score",
    "random_score",
)


def _combined_units(context: RunContext, protocol: LoadedConfig) -> list[pd.DataFrame]:

    frames: list[pd.DataFrame] = []
    for prediction_path in sorted((context.run_dir / "target_adaptation" / "oof").glob("*/*/seed=*/predictions.csv")):
        seed = int(prediction_path.parent.name.split("=", 1)[1])
        target_model = prediction_path.parent.parent.name
        task = prediction_path.parent.parent.parent.name
        diagnostics_path = context.run_dir / "pre_adaptation_diagnostics" / "foldwise" / task / f"seed={seed}.csv"
        controls_path = context.run_dir / "distance_controls" / task / f"seed={seed}.csv"
        if not diagnostics_path.is_file() or not controls_path.is_file():
            raise FileNotFoundError(f"Missing diagnostics or distance controls for {task}, seed={seed}")
        diagnostics = pd.read_csv(diagnostics_path)
        controls = pd.read_csv(controls_path)
        predictions = pd.read_csv(prediction_path)
        merged = diagnostics.merge(controls, on=["sample_id", "y", "task", "fold", "seed"], validate="one_to_one")
        merged = merged.merge(predictions, on=["sample_id", "y", "task", "fold", "seed"], validate="one_to_one")
        if len(merged) != len(predictions):
            raise RuntimeError(f"Metric merge lost samples for {task}, {target_model}, seed={seed}")
        merged["target_model"] = target_model
        merged["high_loss"] = high_loss_endpoint(merged.loss, protocol.fold_evaluation.high_loss_fraction)
        frames.append(merged)
    if not frames:
        raise FileNotFoundError("No out-of-fold predictions are available. Run evaluate-adaptation first.")
    return frames


def _metric_rows(units: Iterable[pd.DataFrame], protocol: LoadedConfig) -> pd.DataFrame:

    rows: list[dict[str, object]] = []
    for unit in units:
        task = str(unit.task.iloc[0])
        seed = int(unit.seed.iloc[0])
        target_model = str(unit.target_model.iloc[0])
        for endpoint, target in (("fixed_threshold_error", unit.fixed_threshold_error), ("high_loss", unit.high_loss)):
            for score in SCORES:
                if score in unit.columns:
                    rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "target_model": target_model,
                            "endpoint": endpoint,
                            "score": score,
                            "n": len(unit),
                            **metric_record(target, unit[score], protocol.fold_evaluation.top_risk_fraction),
                        }
                    )
    return pd.DataFrame(rows).sort_values(["endpoint", "target_model", "score", "task", "seed"]).reset_index(drop=True)


def summarize_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:

    if rows.empty:
        return pd.DataFrame(
            columns=[
                "endpoint",
                "target_model",
                "score",
                "n_task_seed_units",
                "n_tasks",
                "mean_auroc",
                "median_auroc",
                "mean_auprc",
                "mean_top_fraction_lift",
            ]
        )
    return (
        rows.groupby(["endpoint", "target_model", "score"], as_index=False)
        .agg(
            n_task_seed_units=("auroc", "count"),
            n_tasks=("task", "nunique"),
            mean_auroc=("auroc", "mean"),
            median_auroc=("auroc", "median"),
            mean_auprc=("auprc", "mean"),
            mean_top_fraction_lift=("top_fraction_lift", "mean"),
        )
        .sort_values(["endpoint", "target_model", "mean_auroc"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def _risk_coverage(units: Iterable[pd.DataFrame], protocol: LoadedConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    primary_target = str(protocol.target_adaptation.primary_target_model)
    for unit in units:
        if str(unit.target_model.iloc[0]) != primary_target:
            continue
        for score in ("local_probe_score", "probe_only_score", "class_frequency_score"):
            for row in risk_coverage_curve(
                unit.fixed_threshold_error,
                unit[score],
                coverages=protocol.estimation.coverage_levels,
                sample_labels=unit.y,
            ):
                rows.append({"task": str(unit.task.iloc[0]), "seed": int(unit.seed.iloc[0]), "score": score, **row})
    units_frame = pd.DataFrame(rows)
    if units_frame.empty:
        return units_frame, units_frame
    summary = (
        units_frame.groupby(["score", "coverage"], as_index=False)
        .agg(
            n_task_seed_units=("task", "count"),
            mean_error_rate=("error_rate", "mean"),
            median_error_rate=("error_rate", "median"),
            mean_positive_rate=("positive_rate", "mean"),
        )
        .sort_values(["score", "coverage"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return units_frame, summary


def _similarity_summary(context: RunContext) -> pd.DataFrame:
    paths = sorted((context.run_dir / "similarity_holdout").glob("*/seed=*/repeat=*.csv"))
    if not paths:
        return pd.DataFrame()
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return (
        rows.groupby(["endpoint", "score"], as_index=False)
        .agg(
            n_task_units=("task", "count"),
            n_tasks=("task", "nunique"),
            mean_auroc=("auroc", "mean"),
            median_auroc=("auroc", "median"),
            mean_auprc=("auprc", "mean"),
            max_cross_edge_violations=("cross_edge_violations", "max"),
            max_test_to_train_alignment_similarity=("max_test_to_train_alignment_similarity", "max"),
        )
        .sort_values(["endpoint", "mean_auroc"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _target_diagnostics(units: Iterable[pd.DataFrame], bin_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for unit in units:
        labels = unit.y.to_numpy(dtype=int)
        probability = unit.probability.to_numpy(dtype=float)
        rows.append({
            "task": str(unit.task.iloc[0]), "seed": int(unit.seed.iloc[0]), "target_model": str(unit.target_model.iloc[0]),
            "auroc": binary_auroc(labels, probability), "auprc": average_precision(labels, probability),
            "brier_score": brier_score(labels, probability),
            "expected_calibration_error": expected_calibration_error(labels, probability, bin_count),
            "fixed_threshold_error_rate": float(unit.fixed_threshold_error.mean()),
        })
    frame = pd.DataFrame(rows)
    columns = ["auroc", "auprc", "brier_score", "expected_calibration_error", "fixed_threshold_error_rate"]
    summary = frame.groupby("target_model", as_index=False)[columns].mean()
    return frame, summary


def _paired_intervals(metrics: pd.DataFrame, protocol: LoadedConfig, sequence_separated: bool = False) -> pd.DataFrame:
    if sequence_separated:
        pairs = [("local_probe_score", "knn_embedding_distance"), ("probe_only_score", "knn_embedding_distance")]
    else:
        pairs = [("local_probe_score", "probe_uncertainty"), ("local_probe_score", "probe_only_score"),
                 ("local_probe_score", "knn_embedding_distance"), ("local_probe_score", "random_score"),
                 ("probe_only_score", "knn_embedding_distance")]
    frames = []
    for target_model, subset in metrics.groupby("target_model", sort=True):
        for endpoint in ("fixed_threshold_error", "high_loss"):
            intervals = paired_task_intervals(subset, endpoint, pairs, protocol.estimation.resampling_seed, protocol.estimation.task_resamples)
            intervals.insert(0, "target_model", target_model)
            frames.append(intervals)
    return pd.concat(frames, ignore_index=True)


def summarize_run(context: RunContext, protocol: LoadedConfig) -> dict[str, Path]:

    result_dir = context.run_dir / "results"
    units = _combined_units(context, protocol)
    metric_units = _metric_rows(units, protocol)
    metrics = summarize_metric_rows(metric_units)
    coverage_units, coverage = _risk_coverage(units, protocol)
    similarity = _similarity_summary(context)
    target_units, target_summary = _target_diagnostics(units, protocol.estimation.calibration_bins)
    task_means = metric_units.groupby(["task", "target_model", "endpoint", "score"], as_index=False).auroc.mean()
    ranking_intervals = task_cluster_intervals(metric_units, ["target_model", "endpoint", "score"], "auroc", protocol.estimation.resampling_seed, protocol.estimation.task_resamples)
    paired_intervals = _paired_intervals(metric_units, protocol)
    coverage_intervals = risk_coverage_task_intervals(coverage_units, ["local_probe_score", "probe_only_score", "class_frequency_score"], protocol.estimation.resampling_seed, protocol.estimation.task_resamples)
    similarity_paths = sorted((context.run_dir / "similarity_holdout").glob("*/seed=*/repeat=*.csv"))
    similarity_units = pd.concat([pd.read_csv(path) for path in similarity_paths], ignore_index=True) if similarity_paths else pd.DataFrame()
    similarity_intervals = _paired_intervals(similarity_units, protocol, sequence_separated=True) if not similarity_units.empty else pd.DataFrame()
    outputs = {
        "metric_units": result_dir / "metric_units.csv",
        "metric_summary": result_dir / "metric_summary.csv",
        "risk_coverage_summary": result_dir / "risk_coverage_summary.csv",
        "similarity_holdout_summary": result_dir / "similarity_holdout_summary.csv",
        "task_mean_auroc": result_dir / "task_mean_auroc.csv",
        "ranking_intervals": result_dir / "ranking_intervals.csv",
        "paired_auroc_intervals": result_dir / "paired_auroc_intervals.csv",
        "risk_coverage_units": result_dir / "risk_coverage_units.csv",
        "risk_coverage_intervals": result_dir / "risk_coverage_intervals.csv",
        "target_diagnostic_units": result_dir / "target_diagnostic_units.csv",
        "target_diagnostic_summary": result_dir / "target_diagnostic_summary.csv",
        "sequence_separated_intervals": result_dir / "sequence_separated_intervals.csv",
    }
    atomic_write_csv(outputs["metric_units"], metric_units)
    atomic_write_csv(outputs["metric_summary"], metrics)
    atomic_write_csv(outputs["risk_coverage_summary"], coverage)
    atomic_write_csv(outputs["similarity_holdout_summary"], similarity)
    for name, frame in {
        "task_mean_auroc": task_means, "ranking_intervals": ranking_intervals,
        "paired_auroc_intervals": paired_intervals, "risk_coverage_units": coverage_units,
        "risk_coverage_intervals": coverage_intervals, "target_diagnostic_units": target_units,
        "target_diagnostic_summary": target_summary, "sequence_separated_intervals": similarity_intervals,
    }.items():
        atomic_write_csv(outputs[name], frame)
    atomic_write_json(
        result_dir / "result_manifest.json",
        {
            "schema_version": 1,
            "run_id": context.run_id,
            "primary_target_model": str(protocol.target_adaptation.primary_target_model),
            "metric_unit_count": int(len(metric_units)),
            "task_count": int(metrics.n_tasks.max()) if not metrics.empty else 0,
            "files": {name: path.relative_to(context.run_dir).as_posix() for name, path in outputs.items()},
        },
    )
    context.emit("result_summary_completed", output=result_dir.relative_to(context.run_dir).as_posix())
    return outputs


def _compatible_contexts(contexts: list[RunContext]) -> None:
    if len(contexts) != 3 or len({context.run_dir for context in contexts}) != 3:
        raise ValueError("Provide exactly three distinct completed run directories")
    fields = ("config_sha256", "code_sha256", "input_contract_sha256", "model_sha256")
    inconsistent = [field for field in fields if len({str(getattr(context, field)) for context in contexts}) != 1]
    if inconsistent:
        raise RuntimeError(f"Three-run summaries use incompatible contracts: {inconsistent}")


def summarize_three_runs(run_dirs: Iterable[Path], output_dir: Path) -> dict[str, object]:

    contexts = [RunContext.open(path) for path in run_dirs]
    _compatible_contexts(contexts)
    summaries: list[pd.DataFrame] = []
    for context in contexts:
        path = context.run_dir / "results" / "metric_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Result summary is missing: {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "run_id", context.run_id)
        summaries.append(frame)
    combined = pd.concat(summaries, ignore_index=True)
    focal = combined[combined.target_model == "focal_lora"].copy()
    vector = focal.pivot(index="run_id", columns=["endpoint", "score"], values="mean_auroc")
    median = vector.median(axis=0)
    selected_run = str(((vector - median) ** 2).sum(axis=1).sort_values(kind="stable").index[0])
    calibration = (
        focal.groupby(["endpoint", "score"], as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            median_auroc=("mean_auroc", "median"),
            minimum_auroc=("mean_auroc", "min"),
            maximum_auroc=("mean_auroc", "max"),
        )
        .sort_values(["endpoint", "median_auroc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "three_run_metric_summary.csv", calibration)
    atomic_write_csv(output_dir / "three_run_metric_values.csv", focal)
    atomic_write_json(
        output_dir / "three_run_manifest.json",
        {
            "schema_version": 1,
            "run_count": len(contexts),
            "selected_run": selected_run,
            "runs": [context.run_id for context in contexts],
            "selection_rule": "minimum squared distance to the three-run median focal-LoRA AUROC vector",
        },
    )
    return {"selected_run": selected_run, "summary_path": output_dir / "three_run_metric_summary.csv"}
