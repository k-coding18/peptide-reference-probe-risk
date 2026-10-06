from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping
import json
import os
import platform
import tempfile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=path.suffix) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


@dataclass(frozen=True)
class UnitResult:
    status: str
    output: Path
    marker: Path


@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    config_sha256: str
    code_sha256: str
    input_contract_sha256: str = ""
    model_sha256: str = ""

    @classmethod
    def create(
        cls,
        run_root: Path,
        run_id: str,
        config_sha256: str,
        code_sha256: str,
        input_contract_sha256: str = "",
        model_sha256: str = "",
    ) -> "RunContext":
        run_dir = Path(run_root).resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        context = cls(
            run_dir=run_dir,
            run_id=run_id,
            config_sha256=config_sha256,
            code_sha256=code_sha256,
            input_contract_sha256=input_contract_sha256,
            model_sha256=model_sha256,
        )
        manifest = run_dir / "run_manifest.json"
        if not manifest.exists():
            atomic_write_json(
                manifest,
                {
                    "run_id": run_id,
                    "created_at": utc_now(),
                    "config_sha256": config_sha256,
                    "code_sha256": code_sha256,
                    "input_contract_sha256": input_contract_sha256,
                    "model_sha256": model_sha256,
                    "python": platform.python_version(),
                },
            )
        else:
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                existing.get("config_sha256") != config_sha256
                or existing.get("code_sha256") != code_sha256
                or existing.get("input_contract_sha256", "") != input_contract_sha256
                or existing.get("model_sha256", "") != model_sha256
            ):
                raise RuntimeError("The existing run directory was created by another configuration or code revision. Choose a new run directory.")
        return context

    @classmethod
    def open(cls, run_dir: Path) -> "RunContext":

        run_dir = Path(run_dir).resolve()
        manifest = run_dir / "run_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Run manifest is missing: {manifest}")
        values = json.loads(manifest.read_text(encoding="utf-8"))
        return cls(
            run_dir=run_dir,
            run_id=str(values["run_id"]),
            config_sha256=str(values["config_sha256"]),
            code_sha256=str(values["code_sha256"]),
            input_contract_sha256=str(values.get("input_contract_sha256", "")),
            model_sha256=str(values.get("model_sha256", values.get("model_contract_sha256", ""))),
        )

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    def emit(self, event_type: str, **payload: Any) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": utc_now(), "event_type": event_type, "run_id": self.run_id, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        print(f"[{record['time']}] {event_type}: {json.dumps(payload, sort_keys=True, default=str)}", flush=True)

    def marker_path(self, unit_name: str) -> Path:
        safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in unit_name)
        return self.run_dir / "checkpoints" / f"{safe}.json"


def _valid_marker(marker: Path, fingerprint: str, output: Path) -> bool:
    if not marker.is_file() or not output.is_file() or output.stat().st_size == 0:
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("fingerprint") == fingerprint and data.get("output_sha256") == sha256(output.read_bytes()).hexdigest()


def run_atomic_unit(
    context: RunContext,
    unit_name: str,
    parameters: Mapping[str, Any],
    output: Path,
    worker: Callable[[Path], None],
) -> UnitResult:

    output = Path(output)
    marker = context.marker_path(unit_name)
    fingerprint = stable_json_hash(
        {
            "unit": unit_name,
            "parameters": dict(parameters),
            "config_sha256": context.config_sha256,
            "code_sha256": context.code_sha256,
        }
    )
    if _valid_marker(marker, fingerprint, output):
        context.emit("unit_skipped", unit=unit_name, output=output.relative_to(context.run_dir).as_posix())
        return UnitResult(status="skipped", output=output, marker=marker)

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".artifact"
    temporary = output.with_name(f".{output.stem}.{fingerprint[:12]}.partial{suffix}")
    temporary.unlink(missing_ok=True)
    context.emit("unit_started", unit=unit_name, parameters=dict(parameters))
    try:
        worker(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"Unit {unit_name} did not create a nonempty artifact: {temporary}")
        os.replace(temporary, output)
        atomic_write_json(
            marker,
            {
                "unit": unit_name,
                "fingerprint": fingerprint,
                "output": output.relative_to(context.run_dir).as_posix(),
                "output_sha256": sha256(output.read_bytes()).hexdigest(),
                "completed_at": utc_now(),
            },
        )
        context.emit("unit_completed", unit=unit_name, output=output.relative_to(context.run_dir).as_posix())
        return UnitResult(status="completed", output=output, marker=marker)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        context.emit("unit_failed", unit=unit_name, error=repr(exc))
        raise
