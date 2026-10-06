from __future__ import annotations

from pathlib import Path

from pre_peft.config import LoadedConfig
from pre_peft.data.paper_inputs import PAPER_TASK_NAMES, PaperInputBundle, materialize_paper_inputs, validate_paper_inputs
from pre_peft.runtime import RunContext


def prepare_tasks(
    protocol: LoadedConfig,
    context: RunContext,
    input_root: Path,
    contract_path: Path,
    bundle: PaperInputBundle | None = None,
) -> dict[str, Path]:

    verified = bundle or validate_paper_inputs(
        input_root=input_root,
        contract_path=contract_path,
        expected_task_count=int(protocol.task_construction.task_count),
        expected_tasks=PAPER_TASK_NAMES if int(protocol.task_construction.task_count) == len(PAPER_TASK_NAMES) else None,
    )
    task_paths = materialize_paper_inputs(verified, context)
    if len(task_paths) != int(protocol.task_construction.task_count):
        raise RuntimeError(f"Expected {protocol.task_construction.task_count} task tables, obtained {len(task_paths)}")
    context.emit("task_construction_completed", task_count=len(task_paths), mode="materialized_paper_inputs")
    return task_paths


def read_task_paths(context: RunContext) -> dict[str, Path]:
    task_dir = context.run_dir / "task_construction" / "tasks"
    paths = {path.stem: path for path in sorted(task_dir.glob("*.csv"))}
    if not paths:
        raise FileNotFoundError("No materialized paper task tables are available. Run prepare-data first.")
    return paths
