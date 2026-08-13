from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


class UnsafePrivatePathError(ValueError):
    """A private output could overlap source data or the public repository."""


def _contains(parent: Path, child: Path) -> bool:
    parent_text = os.path.normcase(str(parent.resolve(strict=False)))
    child_text = os.path.normcase(str(child.resolve(strict=False)))
    try:
        return os.path.commonpath((parent_text, child_text)) == parent_text
    except ValueError:
        return False


def _overlaps(left: Path, right: Path) -> bool:
    return _contains(left, right) or _contains(right, left)


def public_repository_root() -> Path | None:
    """Locate the source checkout; installed wheels intentionally return ``None``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def validate_private_workspace(workspace: Path, archive_roots: Iterable[Path]) -> Path:
    """Require a dedicated workspace disjoint from archives and public source."""
    resolved = workspace.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise UnsafePrivatePathError("Private workspace cannot be a filesystem root")
    if any(_overlaps(resolved, root.resolve(strict=False)) for root in archive_roots):
        raise UnsafePrivatePathError("Private workspace overlaps a source archive")
    repository = public_repository_root()
    if repository is not None and _overlaps(resolved, repository):
        raise UnsafePrivatePathError("Private workspace overlaps the public repository")
    return resolved


def validate_private_output(
    output: Path,
    *,
    workspace: Path,
    archive_roots: Iterable[Path],
) -> Path:
    """Require a private output to be a strict child of the validated workspace."""
    roots = tuple(archive_roots)
    resolved_workspace = validate_private_workspace(workspace, roots)
    resolved_output = output.resolve(strict=False)
    if resolved_output == resolved_workspace or not _contains(resolved_workspace, resolved_output):
        raise UnsafePrivatePathError("Private output must be inside the configured workspace")
    if any(_overlaps(resolved_output, root.resolve(strict=False)) for root in roots):
        raise UnsafePrivatePathError("Private output overlaps a source archive")
    repository = public_repository_root()
    if repository is not None and _overlaps(resolved_output, repository):
        raise UnsafePrivatePathError("Private output overlaps the public repository")
    return resolved_output
