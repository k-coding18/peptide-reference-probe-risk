from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from pre_peft.adaptation.out_of_fold import evaluate_target_adaptation
from pre_peft.config import load_protocol
from pre_peft.data.paper_inputs import PAPER_TASK_NAMES, PaperInputError, record_paper_input_validation, validate_paper_inputs
from pre_peft.data.task_construction import prepare_tasks, read_task_paths
from pre_peft.diagnostics.reference_probes import run_diagnostics
from pre_peft.evaluation.distance_controls import evaluate_distance_controls
from pre_peft.evaluation.result_summary import summarize_run
from pre_peft.evaluation.similarity_holdout import evaluate_similarity_holdout
from pre_peft.model_source import configure_local_model
from pre_peft.provenance import project_code_sha256
from pre_peft.runtime import RunContext, atomic_write_json


MODEL_COMMANDS = {
    "all",
    "prepare-data",
    "run-diagnostics",
    "evaluate-adaptation",
    "evaluate-distance-controls",
    "evaluate-similarity-holdout",
}


def _project_root(path: str | None) -> Path:
    return Path(path).resolve() if path else Path.cwd().resolve()


def _path_from_root(root: Path, value: str | None, default: str) -> Path:
    selected = Path(value) if value else Path(default)
    return selected.resolve() if selected.is_absolute() else (root / selected).resolve()


def _create_context(
    project_root: Path,
    run_dir: str,
    protocol: Any,
    input_contract_sha256: str,
    model_sha256: str,
) -> RunContext:
    requested = Path(run_dir)
    requested = requested if requested.is_absolute() else project_root / requested
    configuration_hash = sha256((protocol.sha256 + input_contract_sha256 + model_sha256).encode("utf-8")).hexdigest()
    return RunContext.create(
        requested.parent,
        requested.name,
        configuration_hash,
        project_code_sha256(project_root),
        input_contract_sha256=input_contract_sha256,
        model_sha256=model_sha256,
    )


def _record_stage(context: RunContext, name: str, action: Callable[[], Any]) -> Any:
    context.emit("stage_started", stage=name)
    try:
        result = action()
    except Exception as exc:
        context.emit("stage_failed", stage=name, error=repr(exc))
        raise
    atomic_write_json(context.run_dir / name / "result.json", {"stage": name, "result": _relative_outputs(result, context.run_dir)})
    context.emit("stage_completed", stage=name)
    return result


def _relative_outputs(value: Any, run_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _relative_outputs(item, run_dir) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relative_outputs(item, run_dir) for item in value]
    if isinstance(value, Path):
        return value.relative_to(run_dir).as_posix()
    if isinstance(value, str) and value.startswith(str(run_dir) + "/"):
        return Path(value).relative_to(run_dir).as_posix()
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen pre-adaptation failure-risk protocol from local inputs.")
    parser.add_argument(
        "command",
        choices=[
            "all",
            "validate-inputs",
            "prepare-data",
            "run-diagnostics",
            "evaluate-adaptation",
            "evaluate-distance-controls",
            "evaluate-similarity-holdout",
            "summarize-results",
        ],
    )
    parser.add_argument("--project-root", help="Repository root. Defaults to the current directory.")
    parser.add_argument("--run-dir", help="Directory for resumable generated artifacts.")
    parser.add_argument("--config", default="configs/protocol.toml", help="Frozen experiment protocol.")
    parser.add_argument("--input-root", required=True, help="Directory containing the 22 paper input CSV tables and their contract.")
    parser.add_argument("--task-contract", help="Defaults to <input-root>/paper_task_contract.toml.")
    parser.add_argument("--input-contract", default="configs/input_contract.toml", help="Frozen 22-task identity contract.")
    parser.add_argument("--model-root", help="Local ESM2-t6 directory at the revision specified in configs/protocol.toml.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _project_root(args.project_root)
    protocol = load_protocol(_path_from_root(root, args.config, "configs/protocol.toml"))
    input_root = _path_from_root(root, args.input_root, "data/paper_inputs")
    task_contract = _path_from_root(root, args.task_contract, str(input_root / "paper_task_contract.toml"))
    input_contract = _path_from_root(root, args.input_contract, "configs/input_contract.toml")
    try:
        bundle = validate_paper_inputs(
            input_root=input_root,
            contract_path=task_contract,
            expected_task_count=int(protocol.task_construction.task_count),
            expected_tasks=PAPER_TASK_NAMES,
            frozen_contract_path=input_contract,
        )
    except PaperInputError as exc:
        raise SystemExit(f"Input validation failed: {exc}") from exc

    command = args.command
    run_dir = args.run_dir or ("runs/input-validation" if command == "validate-inputs" else "runs/reproduction")
    if command == "summarize-results":
        if not args.run_dir:
            raise RuntimeError("--run-dir is required for summarize-results")
        requested = _path_from_root(root, args.run_dir, "runs/reproduction")
        context = RunContext.open(requested)
        if context.input_contract_sha256 != bundle.contract_sha256:
            raise RuntimeError("The supplied input package differs from the contract recorded by this run")
    else:
        model_sha256 = ""
        if command in MODEL_COMMANDS:
            if not args.model_root:
                raise RuntimeError(f"--model-root is required for {command}")
            model_sha256 = configure_local_model(Path(args.model_root), protocol)
        context = _create_context(root, run_dir, protocol, bundle.contract_sha256, model_sha256)
        if command in MODEL_COMMANDS:
            atomic_write_json(
                context.run_dir / "model_provenance.json",
                {
                    "identifier": protocol.model.identifier,
                    "revision": protocol.model.revision,
                    "model_sha256": model_sha256,
                    "offline": True,
                },
            )

    def validate_stage() -> dict[str, object]:
        return record_paper_input_validation(bundle, context)

    if command in {"all", "validate-inputs", "prepare-data"}:
        _record_stage(context, "data_validation", validate_stage)
    if command in {"all", "prepare-data"}:
        _record_stage(
            context,
            "task_construction",
            lambda: {task: str(path) for task, path in prepare_tasks(protocol, context, input_root, task_contract, bundle).items()},
        )
    if command in {"all", "run-diagnostics"}:
        _record_stage(context, "pre_adaptation_diagnostics", lambda: run_diagnostics(read_task_paths(context), protocol, context))
    if command in {"all", "evaluate-adaptation"}:
        _record_stage(context, "target_adaptation", lambda: evaluate_target_adaptation(read_task_paths(context), protocol, context))
    if command in {"all", "evaluate-distance-controls"}:
        _record_stage(context, "distance_controls", lambda: evaluate_distance_controls(protocol, context))
    if command in {"all", "evaluate-similarity-holdout"}:
        _record_stage(context, "similarity_holdout", lambda: evaluate_similarity_holdout(protocol, context))
    if command in {"all", "summarize-results"}:
        outputs = _record_stage(context, "result_summary", lambda: summarize_run(context, protocol))
        print(f"Result summary: {context.run_dir / 'results' / 'metric_summary.csv'}")
        print(f"Risk coverage: {context.run_dir / 'results' / 'risk_coverage_summary.csv'}")
        print(f"Similarity summary: {context.run_dir / 'results' / 'similarity_holdout_summary.csv'}")
        if command == "summarize-results":
            print(f"Summary files: {len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
