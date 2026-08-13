from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from autoretush.inventory import ImageRecord, PairGroup
from autoretush.pairing import snapshot as snapshot_module
from autoretush.pairing.snapshot import (
    InputSnapshotError,
    build_pairing_algorithm_id,
    current_pairing_algorithm_id,
    snapshot_pair_group,
)


def _record(path: Path) -> ImageRecord:
    return ImageRecord(
        path=path,
        size_bytes=path.stat().st_size,
        extension=path.suffix,
    )


def _group(
    root: Path,
    *,
    before: tuple[ImageRecord, ...],
    after: tuple[ImageRecord, ...],
) -> PairGroup:
    source = root / "set"
    return PairGroup(
        group_id="g_test",
        archive_root=root,
        source_dir=source,
        processed_dir=source / "pp",
        before=before,
        after=after,
    )


def test_pair_group_snapshot_changes_with_content_role_order_and_relative_descriptor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    source = root / "set"
    processed = source / "pp"
    processed.mkdir(parents=True)
    before_a = source / "a.bin"
    before_b = source / "b.bin"
    after = processed / "result.bin"
    before_a.write_bytes(b"AAAA")
    before_b.write_bytes(b"BBBB")
    after.write_bytes(b"CCCC")

    baseline_group = _group(
        root,
        before=(_record(before_a), _record(before_b)),
        after=(_record(after),),
    )
    baseline = snapshot_pair_group(baseline_group, chunk_size=2)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", baseline)

    before_a.write_bytes(b"ZZZZ")
    content_changed = _group(
        root,
        before=(_record(before_a), _record(before_b)),
        after=(_record(after),),
    )
    assert snapshot_pair_group(content_changed) != baseline
    before_a.write_bytes(b"AAAA")

    reordered = _group(
        root,
        before=(_record(before_b), _record(before_a)),
        after=(_record(after),),
    )
    assert snapshot_pair_group(reordered) != baseline

    roles_swapped = _group(
        root,
        before=(_record(after),),
        after=(_record(before_a), _record(before_b)),
    )
    assert snapshot_pair_group(roles_swapped) != baseline

    renamed = source / "renamed.bin"
    before_a.rename(renamed)
    descriptor_changed = _group(
        root,
        before=(_record(renamed), _record(before_b)),
        after=(_record(after),),
    )
    assert snapshot_pair_group(descriptor_changed) != baseline


def test_snapshot_rejects_outside_root_and_stale_inventory(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(InputSnapshotError, match="outside the archive root"):
        snapshot_pair_group(_group(root, before=(_record(outside),), after=()))

    inside = root / "inside.bin"
    inside.write_bytes(b"old")
    stale = _record(inside)
    inside.write_bytes(b"new-and-larger")
    with pytest.raises(InputSnapshotError, match="changed after inventory"):
        snapshot_pair_group(_group(root, before=(stale,), after=()))


def test_snapshot_accepts_inventory_paths_from_a_relative_archive_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("archive")
    source = root / "set" / "before.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"content")
    record = ImageRecord(path=source, size_bytes=7, extension=".bin")
    group = _group(root, before=(record,), after=())

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", snapshot_pair_group(group))


def test_snapshot_rejects_symlink_source(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    target = root / "target.bin"
    target.write_bytes(b"content")
    link = root / "link.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks is not enabled on this host")

    with pytest.raises(InputSnapshotError, match="symlink or reparse point"):
        snapshot_pair_group(_group(root, before=(_record(link),), after=()))


def test_snapshot_detects_change_during_hash(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    source = root / "source.bin"
    source.write_bytes(b"A" * 128)
    group = _group(root, before=(_record(source),), after=())
    original_stream = snapshot_module._stream_content

    def stream_then_touch(handle, digest, chunk_size):
        total = original_stream(handle, digest, chunk_size)
        current = source.stat()
        os.utime(source, ns=(current.st_atime_ns, current.st_mtime_ns + 10_000_000))
        return total

    monkeypatch.setattr(snapshot_module, "_stream_content", stream_then_touch)
    with pytest.raises(InputSnapshotError, match="changed during hashing"):
        snapshot_pair_group(group, chunk_size=16)


def test_algorithm_id_depends_on_all_sources_and_runtime_versions() -> None:
    sources = {
        "fingerprint": b"fingerprint source",
        "geometry": b"geometry source",
        "matcher": b"matcher source",
    }
    runtime = {
        "python": "3.12.10",
        "numpy": "2.2.6",
        "opencv": "4.12.0",
        "scipy": "1.16.1",
    }
    baseline = build_pairing_algorithm_id(sources, runtime)
    assert re.fullmatch(r"autoretush-pairing-v1:[0-9a-f]{64}", baseline)

    for component in sources:
        changed = {**sources, component: sources[component] + b" changed"}
        assert build_pairing_algorithm_id(changed, runtime) != baseline
    for component in runtime:
        changed_runtime = {**runtime, component: runtime[component] + ".post1"}
        assert build_pairing_algorithm_id(sources, changed_runtime) != baseline


def test_current_algorithm_id_is_versioned_and_path_free() -> None:
    algorithm_id = current_pairing_algorithm_id()
    assert re.fullmatch(r"autoretush-pairing-v1:[0-9a-f]{64}", algorithm_id)
    assert "/" not in algorithm_id
    assert "\\" not in algorithm_id
