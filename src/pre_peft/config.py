from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from pre_peft.toml_compat import TOMLDecodeError, loads


class ConfigError(ValueError):
    pass


class ConfigNode:

    def __init__(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            object.__setattr__(self, key, self._convert(value))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError(f"Configuration is immutable: cannot assign {name!r}")

    def __delattr__(self, name: str) -> None:
        raise TypeError(f"Configuration is immutable: cannot delete {name!r}")

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return tuple(cls._convert(item) for item in value)
        return value

    def as_dict(self) -> dict[str, Any]:
        def unpack(value: Any) -> Any:
            if isinstance(value, ConfigNode):
                return {key: unpack(item) for key, item in value.__dict__.items()}
            if isinstance(value, tuple):
                return [unpack(item) for item in value]
            return value

        return {key: unpack(value) for key, value in self.__dict__.items()}


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    values: ConfigNode
    sha256: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.values, name)


def _load(path: Path) -> LoadedConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"Configuration file is missing: {path}")
    raw = path.read_bytes()
    try:
        values = loads(raw.decode("utf-8"))
    except TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    return LoadedConfig(path=path, values=ConfigNode(values), sha256=sha256(raw).hexdigest())


def load_protocol(path: Path) -> LoadedConfig:
    config = _load(path)
    required = [
        "task_construction",
        "fold_evaluation",
        "model",
        "reference_probes",
        "target_adaptation",
        "distance_controls",
        "similarity_holdout",
    ]
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise ConfigError(f"Protocol configuration lacks required sections: {missing}")
    if config.task_construction.task_count != 22:
        raise ConfigError("The public protocol requires exactly 22 tasks")
    if len(config.fold_evaluation.seeds) == 0 or config.fold_evaluation.fold_count < 2:
        raise ConfigError("The out-of-fold protocol requires at least two folds and one seed")
    if len(config.model.revision) != 40:
        raise ConfigError("The protein-language-model revision must be an immutable 40-character commit")
    return config
