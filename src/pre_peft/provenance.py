from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_code_sha256(project_root: Path) -> str:

    project_root = Path(project_root).resolve()
    paths: list[Path] = []
    source_root = project_root / "src" / "pre_peft"
    config_root = project_root / "configs"
    if source_root.exists():
        paths.extend(path for path in source_root.rglob("*.py") if path.is_file())
    if config_root.exists():
        paths.extend(path for path in config_root.rglob("*.toml") if path.is_file())
    pyproject = project_root / "pyproject.toml"
    if pyproject.is_file():
        paths.append(pyproject)
    lines = [f"{path.relative_to(project_root).as_posix()}\t{sha256_file(path)}\n" for path in sorted(paths)]
    return sha256("".join(lines).encode("utf-8")).hexdigest()
