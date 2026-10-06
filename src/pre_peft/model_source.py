from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import re

from pre_peft.config import LoadedConfig
from pre_peft.provenance import sha256_file


MODEL_ROOT_ENV = "PRE_PEFT_MODEL_ROOT"


def configure_local_model(model_root: Path, protocol: LoadedConfig) -> str:
    root = Path(model_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("The supplied local model directory is missing")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "esm" or config.get("hidden_size") != 320 or config.get("num_hidden_layers") != 6:
        raise ValueError("The local model configuration does not match ESM2-t6")
    if re.fullmatch(r"[0-9a-f]{40}", root.name) and root.name != protocol.model.revision:
        raise ValueError("The local snapshot revision differs from the protocol")
    recorded_revision = config.get("_commit_hash")
    if recorded_revision and recorded_revision != protocol.model.revision:
        raise ValueError("The local model revision differs from the protocol")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts))
    digest = sha256()
    for path in files:
        digest.update(f"{path.relative_to(root).as_posix()}\t{sha256_file(path)}\n".encode("utf-8"))
    os.environ[MODEL_ROOT_ENV] = str(root)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    return digest.hexdigest()


def local_model_root() -> Path:
    value = os.environ.get(MODEL_ROOT_ENV)
    if not value:
        raise RuntimeError("Provide --model-root before loading the model")
    root = Path(value).resolve()
    if not root.is_dir():
        raise FileNotFoundError("The configured local model directory is missing")
    return root


def local_pretrained_kwargs() -> dict[str, bool]:
    return {"local_files_only": True}
