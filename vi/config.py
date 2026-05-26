"""Layered config: CLI > project.toml > config.toml."""
import tomllib
from pathlib import Path
from typing import Any

from . import paths


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_global() -> dict:
    return _load(paths.root() / "config.toml")


def load_project(project_dir: Path) -> dict:
    return _load(project_dir / "project.toml")


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def merged(project_dir: Path) -> dict:
    return _deep_merge(load_global(), load_project(project_dir))


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
