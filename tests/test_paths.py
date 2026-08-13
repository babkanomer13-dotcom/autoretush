from pathlib import Path

import pytest

from autoretush.paths import (
    UnsafePrivatePathError,
    public_repository_root,
    validate_private_output,
    validate_private_workspace,
)


def test_private_output_must_be_inside_disjoint_workspace(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    workspace = tmp_path / "private-workspace"
    archive.mkdir()

    assert validate_private_workspace(workspace, [archive]) == workspace.resolve()
    assert (
        validate_private_output(
            workspace / "manifests" / "pairs.jsonl",
            workspace=workspace,
            archive_roots=[archive],
        )
        == (workspace / "manifests" / "pairs.jsonl").resolve()
    )

    with pytest.raises(UnsafePrivatePathError, match="inside"):
        validate_private_output(
            tmp_path / "elsewhere" / "pairs.jsonl",
            workspace=workspace,
            archive_roots=[archive],
        )
    with pytest.raises(UnsafePrivatePathError, match="overlaps a source archive"):
        validate_private_workspace(archive / "private", [archive])


def test_public_repository_cannot_be_a_private_workspace() -> None:
    repository = public_repository_root()
    assert repository is not None

    with pytest.raises(UnsafePrivatePathError, match="public repository"):
        validate_private_workspace(repository / "private-output", [])
