from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from autoretush.config import AppConfig

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    size_bytes: int
    extension: str


@dataclass(frozen=True)
class PairGroup:
    group_id: str
    archive_root: Path
    source_dir: Path
    processed_dir: Path
    before: tuple[ImageRecord, ...]
    after: tuple[ImageRecord, ...]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS


def _images(
    directory: Path,
    *,
    recursive: bool,
    excluded_dir_names: Iterable[str] = (),
) -> tuple[ImageRecord, ...]:
    if recursive:
        excluded = {name.casefold() for name in excluded_dir_names}
        discovered: list[Path] = []
        for current, dir_names, file_names in os.walk(directory, followlinks=False):
            dir_names[:] = sorted(
                (name for name in dir_names if name.casefold() not in excluded),
                key=str.casefold,
            )
            current_path = Path(current)
            discovered.extend(current_path / name for name in sorted(file_names, key=str.casefold))
        iterator: Iterable[Path] = discovered
    else:
        iterator = directory.iterdir()
    records: list[ImageRecord] = []
    for path in iterator:
        try:
            if not is_image(path):
                continue
            stat = path.stat()
        except (OSError, PermissionError):
            continue
        records.append(
            ImageRecord(path=path, size_bytes=stat.st_size, extension=path.suffix.casefold())
        )
    return tuple(sorted(records, key=lambda item: str(item.path).casefold()))


def _group_id(root: Path, processed_dir: Path) -> str:
    try:
        relative = processed_dir.relative_to(root)
    except ValueError:
        relative = processed_dir
    digest = hashlib.sha256(str(relative).casefold().encode("utf-8")).hexdigest()[:16]
    return f"g_{digest}"


def discover_groups(config: AppConfig) -> Iterator[PairGroup]:
    """Yield parent/processed groups without modifying or opening image contents."""
    accepted_names = set(config.processed_folder_names)
    seen: set[Path] = set()

    for root in config.archive_roots:
        for current, dir_names, _ in os.walk(root, followlinks=False):
            current_path = Path(current)
            if current_path.name.casefold() not in accepted_names:
                continue
            resolved = current_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            source_dir = current_path.parent
            before = _images(
                source_dir,
                recursive=not config.inventory.source_images_direct_only,
                excluded_dir_names=config.processed_folder_names,
            )
            after = _images(
                current_path,
                recursive=config.inventory.processed_images_recursive,
            )
            yield PairGroup(
                group_id=_group_id(root, current_path),
                archive_root=root,
                source_dir=source_dir,
                processed_dir=current_path,
                before=before,
                after=after,
            )

            # Avoid scanning the same processed subtree for nested names.
            dir_names.clear()


def build_inventory(config: AppConfig) -> dict[str, object]:
    groups = list(discover_groups(config))
    before_ext: Counter[str] = Counter()
    after_ext: Counter[str] = Counter()
    before_bytes = 0
    after_bytes = 0
    for group in groups:
        before_ext.update(item.extension for item in group.before)
        after_ext.update(item.extension for item in group.after)
        before_bytes += sum(item.size_bytes for item in group.before)
        after_bytes += sum(item.size_bytes for item in group.after)

    return {
        "schema_version": 1,
        "archive_roots": [str(path) for path in config.archive_roots],
        "group_count": len(groups),
        "non_empty_both": sum(bool(group.before and group.after) for group in groups),
        "equal_non_empty_counts": sum(
            bool(group.before) and len(group.before) == len(group.after) for group in groups
        ),
        "before_count": sum(len(group.before) for group in groups),
        "after_count": sum(len(group.after) for group in groups),
        "pair_count_upper_bound": sum(min(len(group.before), len(group.after)) for group in groups),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "before_extensions": dict(sorted(before_ext.items())),
        "after_extensions": dict(sorted(after_ext.items())),
        "groups": [
            {
                "group_id": group.group_id,
                "archive_root": str(group.archive_root),
                "source_dir": str(group.source_dir),
                "processed_dir": str(group.processed_dir),
                "before_count": len(group.before),
                "after_count": len(group.after),
                "before_bytes": sum(item.size_bytes for item in group.before),
                "after_bytes": sum(item.size_bytes for item in group.after),
            }
            for group in groups
        ],
    }


def write_inventory(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite inventory report: {path}")
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex[:12]}")
    try:
        encoded = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def group_summary(group: PairGroup) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "archive_root": str(group.archive_root),
        "source_dir": str(group.source_dir),
        "processed_dir": str(group.processed_dir),
        "before": [{**asdict(item), "path": str(item.path)} for item in group.before],
        "after": [{**asdict(item), "path": str(item.path)} for item in group.after],
    }
