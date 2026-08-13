from pathlib import Path

import pytest

from autoretush.config import ConfigError, load_config


def test_load_config_normalizes_keywords(tmp_path: Path) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        """
archive_roots: ["E:/archive"]
processed_folder_names: ["PP"]
workspace: "E:/local"
selection:
  include_path_keywords: ["ДЕТИ"]
  exclude_path_keywords: ["Учитель"]
materialize:
  mode: copy
  destination: "E:/local/dataset"
  min_overlap_ratio: 0.65
  min_edge_correlation: 0.25
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.processed_folder_names == ("pp",)
    assert config.selection.include_path_keywords == ("дети",)
    assert config.selection.exclude_path_keywords == ("учитель",)
    assert config.inventory.processed_images_recursive is True
    assert config.materialize.min_overlap_ratio == 0.65
    assert config.materialize.min_edge_correlation == 0.25


def test_review_threshold_must_be_lower(tmp_path: Path) -> None:
    config_path = tmp_path / "local.yaml"
    config_path.write_text(
        """
archive_roots: ["E:/archive"]
workspace: "E:/local"
pairing:
  auto_accept_score: 0.7
  review_score: 0.8
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="review_score"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("inventory", "source_images_direct_only", '"false"', "must be a boolean"),
        ("pairing", "shortlist_size", "1.9", "must be an integer"),
        ("pairing", "max_long_side", '"1200"', "must be an integer"),
        ("pairing", "min_inlier_ratio", '"0.6"', "must be numeric"),
    ],
)
def test_config_rejects_implicit_scalar_coercion(
    tmp_path: Path,
    section: str,
    field: str,
    value: str,
    message: str,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f'archive_roots: ["{archive.as_posix()}"]\n'
        f'workspace: "{workspace.as_posix()}"\n'
        f"{section}:\n"
        f"  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_config(config_path)
