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
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.processed_folder_names == ("pp",)
    assert config.selection.include_path_keywords == ("дети",)
    assert config.selection.exclude_path_keywords == ("учитель",)
    assert config.inventory.processed_images_recursive is True


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
