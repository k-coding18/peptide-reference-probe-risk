from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import json

import pandas as pd

from pre_peft.runtime import RunContext, atomic_write_csv, atomic_write_json
from pre_peft.toml_compat import TOMLDecodeError, loads


REQUIRED_COLUMNS = ("sample_id", "sequence", "length", "y", "task")
HEMOLYSIS_TASK = "peptidebert_hemolysis"
CHECKSUM_FILENAME = "SHA256SUMS"

PAPER_TASK_NAMES = (
    "mftp_aap",
    "mftp_abp",
    "mftp_acp",
    "mftp_acvp",
    "mftp_adp",
    "mftp_aep",
    "mftp_afp",
    "mftp_ahivp",
    "mftp_ahp",
    "mftp_aip",
    "mftp_amrsap",
    "mftp_app",
    "mftp_atp",
    "mftp_avp",
    "mftp_bbp",
    "mftp_bip",
    "mftp_cpp",
    "mftp_dppip",
    "mftp_qsp",
    "mftp_sbp",
    "mftp_thp",
    HEMOLYSIS_TASK,
)


class PaperInputError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenTaskIdentity:

    semantic_sha256: str
    n: int
    positives: int
    negatives: int


@dataclass(frozen=True)
class FrozenPaperInputContract:

    path: Path
    sha256: str
    tasks: Mapping[str, FrozenTaskIdentity]


@dataclass(frozen=True)
class PaperInputBundle:
    input_root: Path
    contract_path: Path
    contract_sha256: str
    task_paths: Mapping[str, Path]
    task_hashes: Mapping[str, str]
    file_hashes: Mapping[str, str]
    audit: pd.DataFrame
    frozen_contract_sha256: str = ""


def semantic_task_sha256(table: pd.DataFrame) -> str:

    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise PaperInputError(f"Task table lacks required columns: {missing}")
    canonical = table.loc[:, REQUIRED_COLUMNS].copy()
    canonical["sample_id"] = canonical["sample_id"].astype(str)
    canonical["sequence"] = canonical["sequence"].astype(str)
    canonical["length"] = canonical["length"].astype(int)
    canonical["y"] = canonical["y"].astype(int)
    canonical["task"] = canonical["task"].astype(str)
    text = canonical.to_csv(index=False, lineterminator="\n")
    return sha256(text.encode("utf-8")).hexdigest()


def _load_contract(path: Path) -> tuple[dict[str, str], str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise PaperInputError(f"Paper task contract is missing: {path}")
    raw = path.read_bytes()
    try:
        values = loads(raw.decode("utf-8"))
    except TOMLDecodeError as exc:
        raise PaperInputError(f"Invalid paper task contract: {path}") from exc
    hashes = values.get("semantic_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise PaperInputError("Paper task contract lacks [semantic_sha256]")
    normalized = {str(task): str(digest) for task, digest in hashes.items()}
    invalid = [task for task, digest in normalized.items() if len(digest) != 64]
    if invalid:
        raise PaperInputError(f"Paper task contract contains invalid SHA-256 values: {invalid}")
    return normalized, sha256(raw).hexdigest()


def load_frozen_paper_input_contract(
    path: Path,
    expected_tasks: tuple[str, ...],
) -> FrozenPaperInputContract:

    path = Path(path).resolve()
    if not path.is_file():
        raise PaperInputError(f"Frozen paper-input contract is missing: {path}")
    raw = path.read_bytes()
    try:
        values = loads(raw.decode("utf-8"))
    except TOMLDecodeError as exc:
        raise PaperInputError(f"Invalid frozen paper-input contract: {path}") from exc
    if int(values.get("schema_version", -1)) != 1:
        raise PaperInputError(f"Frozen paper-input contract has an unsupported schema version: {path}")
    columns = tuple(str(column) for column in values.get("semantic_columns", ()))
    if columns != REQUIRED_COLUMNS:
        raise PaperInputError(f"Frozen paper-input contract has incompatible semantic columns: {path}")
    raw_tasks = values.get("tasks")
    if not isinstance(raw_tasks, dict):
        raise PaperInputError(f"Frozen paper-input contract lacks [tasks]: {path}")

    expected = tuple(str(task) for task in expected_tasks)
    if set(raw_tasks) != set(expected):
        missing = sorted(set(expected).difference(raw_tasks))
        unexpected = sorted(set(raw_tasks).difference(expected))
        raise PaperInputError(
            f"Frozen paper-input contract task names differ from the protocol; missing={missing}, unexpected={unexpected}"
        )

    tasks: dict[str, FrozenTaskIdentity] = {}
    for task in expected:
        entry = raw_tasks[task]
        if not isinstance(entry, dict):
            raise PaperInputError(f"Frozen paper-input contract entry is invalid for {task}")
        try:
            identity = FrozenTaskIdentity(
                semantic_sha256=str(entry["semantic_sha256"]),
                n=int(entry["n"]),
                positives=int(entry["positives"]),
                negatives=int(entry["negatives"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperInputError(f"Frozen paper-input contract entry is incomplete for {task}") from exc
        if len(identity.semantic_sha256) != 64 or identity.n <= 0 or identity.positives < 25 or identity.negatives < 25:
            raise PaperInputError(f"Frozen paper-input contract entry is invalid for {task}")
        if identity.n != identity.positives + identity.negatives:
            raise PaperInputError(f"Frozen paper-input class counts are inconsistent for {task}")
        tasks[task] = identity
    return FrozenPaperInputContract(path=path, sha256=sha256(raw).hexdigest(), tasks=tasks)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_task_names(expected_tasks: tuple[str, ...] | None, expected_task_count: int) -> tuple[str, ...] | None:
    if expected_tasks is None:
        return None
    normalized = tuple(str(task) for task in expected_tasks)
    if len(normalized) != expected_task_count or len(set(normalized)) != len(normalized):
        raise PaperInputError("Configured paper task names are not a unique complete task set")
    if expected_task_count == len(PAPER_TASK_NAMES) and HEMOLYSIS_TASK not in normalized:
        raise PaperInputError("Configured paper task names lack peptidebert_hemolysis")
    return normalized


def _validate_checksum_manifest(input_root: Path, task_paths: Mapping[str, Path]) -> dict[str, str]:
    manifest = input_root / CHECKSUM_FILENAME
    if not manifest.is_file():
        raise PaperInputError(f"Paper-input checksum manifest is missing: {manifest}")
    declared: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            digest, name = stripped.split(maxsplit=1)
        except ValueError as exc:
            raise PaperInputError(f"Invalid checksum manifest line: {line!r}") from exc
        name = name.lstrip("*").strip()
        if len(digest) != 64 or Path(name).name != name or not name.endswith(".csv"):
            raise PaperInputError(f"Invalid checksum manifest entry: {line!r}")
        if name in declared:
            raise PaperInputError(f"Duplicate checksum manifest entry: {name}")
        declared[name] = digest
    expected_names = {path.name for path in task_paths.values()}
    if set(declared) != expected_names:
        missing = sorted(expected_names.difference(declared))
        unexpected = sorted(set(declared).difference(expected_names))
        raise PaperInputError(f"Checksum manifest differs from paper-input tables; missing={missing}, unexpected={unexpected}")
    observed = {name: _sha256_file(task_paths[Path(name).stem]) for name in sorted(declared)}
    mismatches = [name for name, digest in declared.items() if observed[name] != digest]
    if mismatches:
        raise PaperInputError(f"Paper-input file SHA-256 mismatch: {mismatches}")
    return {Path(name).stem: observed[name] for name in sorted(observed)}


def _validate_table(task: str, table: pd.DataFrame) -> dict[str, int]:
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise PaperInputError(f"{task} lacks columns: {missing}")
    if table.empty:
        raise PaperInputError(f"{task} is empty")
    if table.sample_id.astype(str).duplicated().any():
        raise PaperInputError(f"{task} contains duplicate sample identifiers")
    if table.sequence.astype(str).duplicated().any():
        raise PaperInputError(f"{task} contains duplicate sequences")
    labels = pd.to_numeric(table.y, errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise PaperInputError(f"{task} contains labels outside {{0, 1}}")
    lengths = pd.to_numeric(table.length, errors="raise").astype(int)
    observed_lengths = table.sequence.astype(str).str.len().astype(int)
    if not lengths.equals(observed_lengths):
        raise PaperInputError(f"{task} length column does not match sequence strings")
    if set(table.task.astype(str)) != {task}:
        raise PaperInputError(f"{task} has a mismatched task column")
    positives = int(labels.sum())
    negatives = int(len(table) - positives)
    if positives < 25 or negatives < 25:
        raise PaperInputError(f"{task} has insufficient class support: positives={positives}, negatives={negatives}")
    return {"n": int(len(table)), "positives": positives, "negatives": negatives}


def _validate_frozen_task_identity(
    task: str,
    counts: Mapping[str, int],
    semantic_sha256: str,
    frozen_contract: FrozenPaperInputContract,
) -> None:
    expected = frozen_contract.tasks[task]
    observed = {"n": int(counts["n"]), "positives": int(counts["positives"]), "negatives": int(counts["negatives"])}
    expected_counts = {"n": expected.n, "positives": expected.positives, "negatives": expected.negatives}
    if observed != expected_counts or semantic_sha256 != expected.semantic_sha256:
        raise PaperInputError(
            f"Frozen paper-input mismatch for {task}: observed counts={observed}, semantic_sha256={semantic_sha256}; "
            f"expected counts={expected_counts}, semantic_sha256={expected.semantic_sha256}; "
            f"contract={frozen_contract.path}"
        )


def validate_paper_inputs(
    input_root: Path,
    contract_path: Path,
    expected_task_count: int = 22,
    expected_tasks: tuple[str, ...] | None = None,
    frozen_contract_path: Path | None = None,
) -> PaperInputBundle:

    input_root = Path(input_root).resolve()
    contract, contract_sha256 = _load_contract(contract_path)
    if len(contract) != expected_task_count:
        raise PaperInputError(f"Expected {expected_task_count} task hashes, found {len(contract)}")
    configured_tasks = _expected_task_names(expected_tasks, expected_task_count)
    paths = {path.stem: path for path in sorted(input_root.glob("*.csv"))}
    if set(paths) != set(contract):
        missing = sorted(set(contract).difference(paths))
        unexpected = sorted(set(paths).difference(contract))
        raise PaperInputError(f"Paper-input task names differ from contract; missing={missing}, unexpected={unexpected}")
    if configured_tasks is not None and set(paths) != set(configured_tasks):
        missing = sorted(set(configured_tasks).difference(paths))
        unexpected = sorted(set(paths).difference(configured_tasks))
        raise PaperInputError(f"Paper-input task names differ from the frozen paper protocol; missing={missing}, unexpected={unexpected}")

    frozen_contract = None
    if frozen_contract_path is not None:
        frozen_contract = load_frozen_paper_input_contract(
            frozen_contract_path,
            configured_tasks or tuple(sorted(paths)),
        )

    file_hashes = _validate_checksum_manifest(input_root, paths)

    audit_rows: list[dict[str, object]] = []
    observed_hashes: dict[str, str] = {}
    for task, path in sorted(paths.items()):
        table = pd.read_csv(path)
        counts = _validate_table(task, table)
        digest = semantic_task_sha256(table)
        if digest != contract[task]:
            raise PaperInputError(f"Semantic task hash mismatch: {task}")
        if frozen_contract is not None:
            _validate_frozen_task_identity(task, counts, digest, frozen_contract)
        observed_hashes[task] = digest
        audit_rows.append({"task": task, "semantic_sha256": digest, **counts})
    audit = pd.DataFrame(audit_rows).sort_values("task").reset_index(drop=True)
    return PaperInputBundle(
        input_root,
        Path(contract_path).resolve(),
        contract_sha256,
        paths,
        observed_hashes,
        file_hashes,
        audit,
        frozen_contract.sha256 if frozen_contract is not None else "",
    )


def record_paper_input_validation(bundle: PaperInputBundle, context: RunContext) -> dict[str, object]:

    output = context.run_dir / "data_validation" / "paper_input_validation.json"
    payload = {
        "mode": "offline_paper_inputs",
        "contract": bundle.contract_path.name,
        "contract_sha256": bundle.contract_sha256,
        "frozen_contract_sha256": bundle.frozen_contract_sha256,
        "task_count": len(bundle.task_paths),
        "tasks": [
            {
                "task": row.task,
                "file": bundle.task_paths[str(row.task)].name,
                "semantic_sha256": row.semantic_sha256,
                "file_sha256": bundle.file_hashes[str(row.task)],
                "n": int(row.n),
                "positives": int(row.positives),
                "negatives": int(row.negatives),
            }
            for row in bundle.audit.itertuples(index=False)
        ],
    }
    atomic_write_json(output, payload)
    context.emit("paper_inputs_validated", task_count=len(bundle.task_paths), contract_sha256=bundle.contract_sha256)
    return payload


def materialize_paper_inputs(bundle: PaperInputBundle, context: RunContext) -> dict[str, Path]:

    task_dir = context.run_dir / "task_construction" / "tasks"
    paths: dict[str, Path] = {}
    for task, source in sorted(bundle.task_paths.items()):
        target = task_dir / source.name
        table = pd.read_csv(source)
        atomic_write_csv(target, table.loc[:, REQUIRED_COLUMNS])
        paths[task] = target
    atomic_write_csv(context.run_dir / "task_construction" / "task_audit.csv", bundle.audit)
    atomic_write_json(
        context.run_dir / "task_construction" / "task_construction_manifest.json",
        {
            "mode": "materialized_paper_inputs",
            "contract_sha256": bundle.contract_sha256,
            "frozen_contract_sha256": bundle.frozen_contract_sha256,
            "task_count": len(paths),
            "tasks": {task: path.relative_to(context.run_dir).as_posix() for task, path in paths.items()},
        },
    )
    context.emit("paper_inputs_materialized", task_count=len(paths))
    return paths

