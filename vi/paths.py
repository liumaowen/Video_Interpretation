"""Path resolution. All commands use absolute paths to avoid cwd drift."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def root() -> Path:
    return ROOT


def projects_dir() -> Path:
    return ROOT / "projects"


def shared_path(*parts) -> Path:
    return ROOT.joinpath("shared", *parts)


def project_path(name: str, *parts) -> Path:
    return projects_dir().joinpath(name, *parts)


def find_project_root(cwd: Path | None = None) -> Path | None:
    """Walk up from cwd to find a project.toml. Returns project dir or None."""
    cur = (cwd or Path.cwd()).resolve()
    while True:
        if (cur / "project.toml").exists():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def resolve_project(name: str | None) -> Path:
    """Return absolute project dir. If name is None, infer from cwd."""
    if name:
        p = project_path(name)
        if not p.exists():
            raise FileNotFoundError(f"Project not found: {p}")
        return p
    p = find_project_root()
    if not p:
        raise SystemExit(
            "No project name given and cwd is not inside a project. "
            "Pass <name> or cd into projects/<name>/."
        )
    return p


def list_projects() -> list[Path]:
    pd = projects_dir()
    if not pd.exists():
        return []
    return sorted(
        d for d in pd.iterdir()
        if d.is_dir() and (d / "project.toml").exists()
    )
