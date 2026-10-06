from __future__ import annotations

from typing import Iterable, Mapping


def compose_candidate_compatible_scores(rows: Iterable[Mapping[str, float]]) -> list[dict[str, float]]:

    output: list[dict[str, float]] = []
    for row in rows:
        density = float(row["embedding_density"])
        uncertainty = float(row["probe_uncertainty"])
        variance = float(row["probe_variance"])
        output.append(
            {
                "local_probe_score": (density + uncertainty + variance) / 3.0,
                "probe_only_score": (uncertainty + variance) / 2.0,
            }
        )
    return output


def compose_audit_score(embedding_density: float, boundary_exposure: float, probe_uncertainty: float, probe_variance: float) -> float:

    return (float(embedding_density) + float(boundary_exposure) + float(probe_uncertainty) + float(probe_variance)) / 4.0
