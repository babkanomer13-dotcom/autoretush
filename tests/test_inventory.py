from pathlib import Path

import pytest

from autoretush.config import (
    AppConfig,
    InventoryConfig,
    MaterializeConfig,
    PairingConfig,
    SelectionConfig,
)
from autoretush.inventory import build_inventory, discover_groups, write_inventory


def _config(root: Path, workspace: Path) -> AppConfig:
    return AppConfig(
        archive_roots=(root,),
        processed_folder_names=("pp",),
        workspace=workspace,
        inventory=InventoryConfig(processed_images_recursive=True),
        selection=SelectionConfig(),
        pairing=PairingConfig(),
        materialize=MaterializeConfig(mode="copy", destination=workspace / "dataset"),
    )


def test_inventory_finds_nested_processed_images(tmp_path: Path) -> None:
    source = tmp_path / "season" / "album" / "дети на разворот"
    nested = source / "pp" / "nested"
    nested.mkdir(parents=True)
    (source / "before.JPG").write_bytes(b"before")
    (nested / "after.jpeg").write_bytes(b"after")

    config = _config(tmp_path / "season", tmp_path / "workspace")
    groups = list(discover_groups(config))
    report = build_inventory(config)

    assert len(groups) == 1
    assert len(groups[0].before) == 1
    assert len(groups[0].after) == 1
    assert report["pair_count_upper_bound"] == 1


def test_recursive_source_scan_never_includes_processed_subtrees(tmp_path: Path) -> None:
    root = tmp_path / "season"
    source = root / "album"
    processed = source / "pp"
    nested_source = source / "nested"
    processed.mkdir(parents=True)
    nested_source.mkdir()
    (source / "before.jpg").write_bytes(b"before")
    (nested_source / "another-before.jpg").write_bytes(b"before")
    after = processed / "after.jpg"
    after.write_bytes(b"after")
    base = _config(root, tmp_path / "workspace")
    config = AppConfig(
        archive_roots=base.archive_roots,
        processed_folder_names=base.processed_folder_names,
        workspace=base.workspace,
        inventory=InventoryConfig(source_images_direct_only=False),
        selection=base.selection,
        pairing=base.pairing,
        materialize=base.materialize,
    )

    [group] = list(discover_groups(config))

    assert {record.path.name for record in group.before} == {
        "before.jpg",
        "another-before.jpg",
    }
    assert after not in {record.path for record in group.before}
    assert [record.path for record in group.after] == [after]


def test_inventory_report_never_overwrites_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "inventory.json"
    destination.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        write_inventory({"schema_version": 1}, destination)

    assert destination.read_text(encoding="utf-8") == "keep"
