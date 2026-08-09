from pathlib import Path

from autoretush.config import (
    AppConfig,
    InventoryConfig,
    MaterializeConfig,
    PairingConfig,
    SelectionConfig,
)
from autoretush.inventory import build_inventory, discover_groups


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
