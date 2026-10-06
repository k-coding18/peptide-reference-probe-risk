from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from pre_peft.config import load_protocol
from pre_peft.evaluation.result_summary import summarize_three_runs


def _run(command: list[str], project_root: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the protocol from local paper inputs and a verified local ESM2-t6 snapshot."
    )
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-root", required=True, help="Extracted 22-task paper input package.")
    parser.add_argument("--model-root", required=True, help="Local ESM2-t6 directory at the revision specified in configs/protocol.toml.")
    parser.add_argument("--run-root", default="runs/reproduction", help="Directory for resumable generated artifacts.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_root = Path(args.input_root).resolve()
    model_root = Path(args.model_root).resolve()
    run_root = Path(args.run_root)
    run_root = run_root.resolve() if run_root.is_absolute() else (project_root / run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for index in range(1, 4):
        run_dir = run_root / f"run-{index:02d}"
        run_dirs.append(run_dir)
        _run(
            [
                sys.executable,
                "-m",
                "pre_peft.cli",
                "all",
                "--project-root",
                str(project_root),
                "--input-root",
                str(input_root),
                "--model-root",
                str(model_root),
                "--run-dir",
                str(run_dir),
            ],
            project_root,
        )

    protocol = load_protocol(project_root / "configs" / "protocol.toml")
    summary = summarize_three_runs(run_dirs, run_root / "three_run_summary")
    print(f"Primary target model: {protocol.target_adaptation.primary_target_model}")
    print(f"Representative run: {summary['selected_run']}")
    print(f"Three-run summary: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
