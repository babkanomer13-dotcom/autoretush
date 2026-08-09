import json
from pathlib import Path

from typer.testing import CliRunner

import autoretush.cli as cli_module
from autoretush.cli import app
from autoretush.paths import public_repository_root


def _config(path: Path, *, archive: Path, workspace: Path) -> None:
    path.write_text(
        f'archive_roots: ["{archive.as_posix()}"]\n'
        f'workspace: "{workspace.as_posix()}"\n'
        "materialize:\n"
        "  mode: copy\n"
        f'  destination: "{(workspace / "dataset").as_posix()}"\n',
        encoding="utf-8",
    )


def test_full_pair_cli_passes_reproducibility_inputs_and_creates_run_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "archive"
    source = archive / "institution" / "layout"
    processed = source / "pp"
    processed.mkdir(parents=True)
    (source / "before.jpg").write_bytes(b"before")
    (processed / "after.jpg").write_bytes(b"after")
    workspace = tmp_path / "private-workspace"
    config = tmp_path / "local.yaml"
    _config(config, archive=archive, workspace=workspace)
    output = workspace / "manifests" / "full.jsonl"
    monkeypatch.setattr(cli_module, "match_group", lambda _group, _config: [])

    result = CliRunner().invoke(
        app,
        [
            "pair",
            "--config",
            str(config),
            "--output",
            str(output),
            "--target-per-root",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.read_bytes() == b""
    metadata_path = output.with_name(f".{output.name}.pairing-staging") / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema"] == 2
    assert metadata["group_count"] == 1
    assert str(archive) not in metadata_path.read_text(encoding="utf-8")


def test_mutating_cli_rejects_output_inside_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    workspace = tmp_path / "private-workspace"
    config = tmp_path / "local.yaml"
    _config(config, archive=archive, workspace=workspace)

    result = CliRunner().invoke(
        app,
        [
            "inventory",
            "--config",
            str(config),
            "--output",
            str(archive / "inventory.json"),
        ],
    )

    assert result.exit_code != 0
    assert "Private output must be inside" in result.output
    assert not (archive / "inventory.json").exists()


def test_inventory_cli_does_not_overwrite_existing_private_report(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    workspace = tmp_path / "private-workspace"
    config = tmp_path / "local.yaml"
    _config(config, archive=archive, workspace=workspace)
    output = workspace / "inventory" / "inventory.json"
    output.parent.mkdir(parents=True)
    output.write_text("keep-me", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["inventory", "--config", str(config), "--output", str(output)],
    )

    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output
    assert output.read_text(encoding="utf-8") == "keep-me"


def test_configured_workspace_cannot_live_in_public_repository(tmp_path: Path) -> None:
    repository = public_repository_root()
    assert repository is not None
    archive = tmp_path / "archive"
    archive.mkdir()
    config = tmp_path / "local.yaml"
    _config(config, archive=archive, workspace=repository / "private-output")

    result = CliRunner().invoke(app, ["inventory", "--config", str(config)])

    assert result.exit_code != 0
    assert "overlaps the public" in result.output
    assert not (repository / "private-output").exists()
