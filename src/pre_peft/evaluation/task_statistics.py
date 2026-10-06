from __future__ import annotations

import numpy as np
import pandas as pd


def paired_task_intervals(frame: pd.DataFrame, endpoint: str, pairs: list[tuple[str, str]], seed: int, n_boot: int) -> pd.DataFrame:
    subset = frame[frame.endpoint == endpoint]
    wide = subset.pivot_table(index="task", columns="score", values="auroc", aggfunc="mean")
    random = np.random.default_rng(seed)
    rows = []
    for score_a, score_b in pairs:
        if score_a not in wide or score_b not in wide:
            continue
        values = (wide[score_a] - wide[score_b]).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        draws = np.array([random.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
        rows.append({
            "endpoint": endpoint, "score_a": score_a, "score_b": score_b,
            "n_tasks": len(values), "n_positive_tasks": int((values > 0).sum()),
            "delta_auroc_task_mean": float(values.mean()),
            "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        })
    return pd.DataFrame(rows)


def task_cluster_intervals(frame: pd.DataFrame, group_columns: list[str], value_column: str, seed: int, n_boot: int) -> pd.DataFrame:
    # Average repetitions within each task before resampling tasks.
    task_values = frame.groupby(["task", *group_columns], as_index=False)[value_column].mean()
    random = np.random.default_rng(seed)
    rows = []
    for keys, group in task_values.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group[value_column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        draws = np.array([random.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
        row = dict(zip(group_columns, keys))
        row.update(n_tasks=int(len(values)), mean=float(values.mean()), ci_low=float(np.quantile(draws, 0.025)), ci_high=float(np.quantile(draws, 0.975)))
        rows.append(row)
    return pd.DataFrame(rows)


def risk_coverage_task_intervals(units: pd.DataFrame, scores: list[str], seed: int, n_boot: int) -> pd.DataFrame:
    task_values = (
        units[units.score.isin(scores)].groupby(["task", "score", "coverage"], as_index=False)
        .agg(error_rate=("error_rate", "mean"), positive_rate=("positive_rate", "mean"))
    )
    derived = []
    for (task, score), group in task_values.groupby(["task", "score"]):
        full = group.loc[np.isclose(group.coverage, 1.0), "error_rate"]
        if full.empty or float(full.iloc[0]) <= 0:
            continue
        full_error = float(full.iloc[0])
        for row in group.itertuples():
            flagged_fraction = 1.0 - float(row.coverage)
            if flagged_fraction <= 0:
                continue
            flagged_error = (full_error - float(row.coverage) * float(row.error_rate)) / flagged_fraction
            derived.append({"task": task, "score": score, "coverage": float(row.coverage), "error_enrichment": flagged_error / full_error})
    long_values = pd.concat([
        task_values.rename(columns={"error_rate": "value"}).assign(metric="unflagged_error_rate")[["task", "score", "coverage", "metric", "value"]],
        task_values.rename(columns={"positive_rate": "value"}).assign(metric="unflagged_positive_prevalence")[["task", "score", "coverage", "metric", "value"]],
        pd.DataFrame(derived, columns=["task", "score", "coverage", "error_enrichment"]).rename(columns={"error_enrichment": "value"}).assign(metric="flagged_error_enrichment")[["task", "score", "coverage", "metric", "value"]],
    ], ignore_index=True)
    return task_cluster_intervals(long_values, ["score", "coverage", "metric"], "value", seed, n_boot)
