from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a local configuration is incomplete or unsafe."""


@dataclass(frozen=True)
class InventoryConfig:
    source_images_direct_only: bool = True
    processed_images_recursive: bool = True


@dataclass(frozen=True)
class PairingConfig:
    shortlist_size: int = 8
    max_long_side: int = 1200
    min_keypoint_inliers: int = 30
    min_inlier_ratio: float = 0.60
    min_source_coverage: float = 0.08
    max_phash_distance: int = 28
    auto_accept_score: float = 0.78
    review_score: float = 0.62


@dataclass(frozen=True)
class SelectionConfig:
    include_path_keywords: tuple[str, ...] = ()
    exclude_path_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterializeConfig:
    mode: str
    destination: Path
    min_overlap_ratio: float = 0.50
    min_edge_correlation: float = 0.15


@dataclass(frozen=True)
class AppConfig:
    archive_roots: tuple[Path, ...]
    processed_folder_names: tuple[str, ...]
    workspace: Path
    inventory: InventoryConfig
    selection: SelectionConfig
    pairing: PairingConfig
    materialize: MaterializeConfig


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    return Path(value).expanduser()


def _bounded_float(value: Any, field: str, *, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric")
    result = float(value)
    if not low <= result <= high:
        raise ConfigError(f"{field} must be between {low} and {high}")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    if value <= 0:
        raise ConfigError(f"{field} must be positive")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def load_config(path: Path) -> AppConfig:
    """Load and validate a private local YAML configuration."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    root = _mapping(raw, "config")
    raw_roots = root.get("archive_roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ConfigError("archive_roots must contain at least one path")
    archive_roots = tuple(_path(item, "archive_roots[]") for item in raw_roots)

    raw_names = root.get("processed_folder_names", ["pp"])
    if not isinstance(raw_names, list) or not raw_names:
        raise ConfigError("processed_folder_names must be a non-empty list")
    processed_names = tuple(str(item).strip().casefold() for item in raw_names if str(item).strip())
    if not processed_names:
        raise ConfigError("processed_folder_names contains no usable names")

    workspace = _path(root.get("workspace"), "workspace")

    inv = _mapping(root.get("inventory"), "inventory")
    inventory = InventoryConfig(
        source_images_direct_only=_boolean(
            inv.get("source_images_direct_only", True),
            "inventory.source_images_direct_only",
        ),
        processed_images_recursive=_boolean(
            inv.get("processed_images_recursive", True),
            "inventory.processed_images_recursive",
        ),
    )

    selection_raw = _mapping(root.get("selection"), "selection")

    def keywords(field: str) -> tuple[str, ...]:
        values = selection_raw.get(field, [])
        if not isinstance(values, list):
            raise ConfigError(f"selection.{field} must be a list")
        return tuple(str(value).strip().casefold() for value in values if str(value).strip())

    selection = SelectionConfig(
        include_path_keywords=keywords("include_path_keywords"),
        exclude_path_keywords=keywords("exclude_path_keywords"),
    )

    pair = _mapping(root.get("pairing"), "pairing")
    pairing = PairingConfig(
        shortlist_size=_positive_int(pair.get("shortlist_size", 8), "pairing.shortlist_size"),
        max_long_side=_positive_int(pair.get("max_long_side", 1200), "pairing.max_long_side"),
        min_keypoint_inliers=_positive_int(
            pair.get("min_keypoint_inliers", 30), "pairing.min_keypoint_inliers"
        ),
        min_inlier_ratio=_bounded_float(
            pair.get("min_inlier_ratio", 0.60),
            "pairing.min_inlier_ratio",
            low=0.0,
            high=1.0,
        ),
        min_source_coverage=_bounded_float(
            pair.get("min_source_coverage", 0.08),
            "pairing.min_source_coverage",
            low=0.0,
            high=1.0,
        ),
        max_phash_distance=_positive_int(
            pair.get("max_phash_distance", 28), "pairing.max_phash_distance"
        ),
        auto_accept_score=_bounded_float(
            pair.get("auto_accept_score", 0.78),
            "pairing.auto_accept_score",
            low=0.0,
            high=1.0,
        ),
        review_score=_bounded_float(
            pair.get("review_score", 0.62),
            "pairing.review_score",
            low=0.0,
            high=1.0,
        ),
    )
    if pairing.review_score >= pairing.auto_accept_score:
        raise ConfigError("pairing.review_score must be lower than auto_accept_score")

    mat = _mapping(root.get("materialize"), "materialize")
    materialize = MaterializeConfig(
        mode=str(mat.get("mode", "copy")).casefold(),
        destination=_path(
            mat.get("destination", str(workspace / "dataset")), "materialize.destination"
        ),
        min_overlap_ratio=_bounded_float(
            mat.get("min_overlap_ratio", 0.50),
            "materialize.min_overlap_ratio",
            low=0.0,
            high=1.0,
        ),
        min_edge_correlation=_bounded_float(
            mat.get("min_edge_correlation", 0.15),
            "materialize.min_edge_correlation",
            low=-1.0,
            high=1.0,
        ),
    )
    if materialize.mode != "copy":
        raise ConfigError("Only safe copy materialization is supported")

    return AppConfig(
        archive_roots=archive_roots,
        processed_folder_names=processed_names,
        workspace=workspace,
        inventory=inventory,
        selection=selection,
        pairing=pairing,
        materialize=materialize,
    )


def validate_archive_roots(config: AppConfig) -> None:
    missing = [path for path in config.archive_roots if not path.is_dir()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise ConfigError(f"Archive roots are missing or not directories: {rendered}")
