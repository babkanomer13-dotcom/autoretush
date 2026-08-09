import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from autoretush.cli import app
from autoretush.dataset import (
    DatasetError,
    assign_group_splits,
    build_materialization_plan,
    materialize,
    select_pairs,
)


def _record(
    pair_id: str,
    group_id: str,
    before: Path,
    after: Path,
    decision: str = "accepted",
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "group_id": group_id,
        "before_path": str(before),
        "after_path": str(after),
        "decision": decision,
        "score": 0.91,
    }


def _geometry(
    homography: list[float] | None = None,
    *,
    height: int = 96,
    width: int = 128,
) -> dict[str, object]:
    return {
        "prepared_before_height": height,
        "prepared_before_width": width,
        "prepared_after_height": height,
        "prepared_after_width": width,
        "transform_valid": True,
        "homography": homography or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }


def _texture(seed: int, *, width: int = 128, height: int = 96) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _save_rgb(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_manual_review_overrides_automatic_decisions(tmp_path: Path) -> None:
    records = []
    for index, decision in enumerate(("accepted", "review", "accepted", "review")):
        records.append(
            _record(
                f"p_{index:020x}",
                f"g_{index}",
                tmp_path / f"b{index}.jpg",
                tmp_path / f"a{index}.jpg",
                decision,
            )
        )
    answers = {
        records[1]["pair_id"]: "match",
        records[2]["pair_id"]: "wrong",
        records[3]["pair_id"]: "unsure",
    }

    selected, excluded = select_pairs(records, review_answers=answers)

    assert [pair.pair_id for pair in selected] == [records[0]["pair_id"], records[1]["pair_id"]]
    assert excluded == {"manual_unsure": 1, "manual_wrong": 1}


def test_dataset_rejects_path_traversal_pair_id(tmp_path: Path) -> None:
    record = _record("../../outside", "g_one", tmp_path / "before", tmp_path / "after")

    with pytest.raises(DatasetError, match="safe pair_id"):
        select_pairs([record])


def test_dataset_rejects_review_answers_from_another_manifest(tmp_path: Path) -> None:
    record = _record(
        "p_00000000000000000001",
        "g_one",
        tmp_path / "before",
        tmp_path / "after",
    )

    with pytest.raises(DatasetError, match="absent from the candidate manifest"):
        select_pairs(
            [record],
            review_answers={"p_00000000000000000002": "wrong"},
        )


def test_group_split_never_separates_one_group(tmp_path: Path) -> None:
    records = [
        _record(
            f"p_{index:020x}", f"g_{index // 3}", tmp_path / f"b{index}", tmp_path / f"a{index}"
        )
        for index in range(30)
    ]
    selected, _ = select_pairs(records)

    first = assign_group_splits(selected, seed="fixed")
    second = assign_group_splits(reversed(selected), seed="fixed")

    by_group: dict[str, set[str]] = {}
    for pair in first:
        by_group.setdefault(pair.group_id, set()).add(pair.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert {pair.pair_id: pair.split for pair in first} == {
        pair.pair_id: pair.split for pair in second
    }


def test_materialization_copies_without_changing_sources(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before = archive / "group" / "before.png"
    after = archive / "group" / "pp" / "after.png"
    pixels = _texture(1)
    _save_rgb(before, pixels)
    _save_rgb(after, np.clip(pixels.astype(np.int16) * 0.9 + 12, 0, 255).astype(np.uint8))
    source_before = before.read_bytes()
    source_after = after.read_bytes()
    destination = tmp_path / "local" / "dataset"
    record = _record("p_00000000000000000001", "g_one", before, after)
    record["geometry"] = _geometry()
    records = [record]

    plan = build_materialization_plan(
        records,
        destination,
        [archive],
        reserve_fraction=0.0,
        minimum_reserve_bytes=0,
    )
    summary = materialize(plan)

    assert summary["pair_count"] == 1
    pair_dir = destination / "pairs" / "p_00000000000000000001"
    assert (pair_dir / "before.png").read_bytes() == source_before
    assert (pair_dir / "after.png").read_bytes() == source_after
    assert before.read_bytes() == source_before
    assert after.read_bytes() == source_after
    manifest = json.loads((destination / "manifests" / "pairs.jsonl").read_text())
    provenance = json.loads((destination / "manifests" / "provenance.jsonl").read_text())
    assert not Path(manifest["before"]).is_absolute()
    assert manifest["alignment"]["reference"] == "after"
    assert manifest["alignment"]["homography_before_to_after"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert provenance["before_source"] == str(before.resolve())


def test_destination_cannot_overlap_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before = archive / "before.jpg"
    after = archive / "pp" / "after.jpg"
    before.parent.mkdir(parents=True)
    after.parent.mkdir(parents=True)
    before.write_bytes(b"b")
    after.write_bytes(b"a")

    with pytest.raises(DatasetError, match="overlaps"):
        build_materialization_plan(
            [_record("p_00000000000000000001", "g_one", before, after)],
            archive / "dataset",
            [archive],
        )


def test_selected_pair_cannot_copy_a_file_from_outside_the_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before = archive / "institution" / "before.jpg"
    after = tmp_path / "unrelated-private-file.jpg"
    before.parent.mkdir(parents=True)
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    with pytest.raises(DatasetError, match="outside every archive root"):
        build_materialization_plan(
            [_record("p_00000000000000000001", "g_one", before, after)],
            tmp_path / "local" / "dataset",
            [archive],
            reserve_fraction=0.0,
            minimum_reserve_bytes=0,
        )


def test_selected_pair_cannot_cross_institution_boundaries(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before = archive / "institution-a" / "before.jpg"
    after = archive / "institution-b" / "pp" / "after.jpg"
    before.parent.mkdir(parents=True)
    after.parent.mkdir(parents=True)
    before.write_bytes(b"before")
    after.write_bytes(b"after")

    with pytest.raises(DatasetError, match="crosses archive or institution boundaries"):
        build_materialization_plan(
            [_record("p_00000000000000000001", "g_one", before, after)],
            tmp_path / "local" / "dataset",
            [archive],
            reserve_fraction=0.0,
            minimum_reserve_bytes=0,
        )


def test_same_institution_stays_together_across_seasons(tmp_path: Path) -> None:
    roots = (tmp_path / "season-a", tmp_path / "season-b")
    records = []
    for index, root in enumerate(roots):
        before = root / "same-institution" / "layout" / f"before-{index}.png"
        after = root / "same-institution" / "layout" / "pp" / f"after-{index}.png"
        pixels = _texture(index + 10)
        _save_rgb(before, pixels)
        _save_rgb(after, pixels)
        record = _record(f"p_{index:020x}", f"g_{index}", before, after)
        record["geometry"] = _geometry()
        records.append(record)

    plan = build_materialization_plan(
        records,
        tmp_path / "local" / "dataset",
        roots,
        reserve_fraction=0.0,
        minimum_reserve_bytes=0,
    )

    assert len({pair.split_group_id for pair in plan.pairs}) == 1
    assert len({pair.split for pair in plan.pairs}) == 1


def test_full_resolution_qc_accepts_good_texture_with_strong_geometry(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before = archive / "institution" / "layout" / "before.png"
    after = archive / "institution" / "layout" / "pp" / "after.png"
    pixels = _texture(40)
    dx, dy = 6, 4
    shifted = np.zeros_like(pixels)
    shifted[dy:, dx:] = pixels[:-dy, :-dx]
    _save_rgb(before, pixels)
    _save_rgb(after, shifted)
    record = _record("p_00000000000000000040", "g_good", before, after)
    record["geometry"] = _geometry([1.0, 0.0, float(dx), 0.0, 1.0, float(dy), 0.0, 0.0, 1.0])

    plan = build_materialization_plan(
        [record],
        tmp_path / "local" / "dataset",
        [archive],
        geometry_max_long_side=128,
        alignment_min_overlap_ratio=0.80,
        alignment_min_edge_correlation=0.90,
        reserve_fraction=0.0,
        minimum_reserve_bytes=0,
    )

    assert [pair.pair_id for pair in plan.pairs] == [record["pair_id"]]
    assert plan.excluded_counts == {}


def test_materialization_uses_stored_prepared_shapes_not_current_pairing_size(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    before = archive / "institution" / "layout" / "before.png"
    after = archive / "institution" / "layout" / "pp" / "after.png"
    pixels = _texture(45)
    _save_rgb(before, pixels)
    _save_rgb(after, pixels)
    record = _record("p_00000000000000000045", "g_stored_scale", before, after)
    record["geometry"] = _geometry()

    plan = build_materialization_plan(
        [record],
        tmp_path / "local" / "dataset",
        [archive],
        geometry_max_long_side=64,
        alignment_min_edge_correlation=0.90,
        reserve_fraction=0.0,
        minimum_reserve_bytes=0,
    )

    assert [pair.pair_id for pair in plan.pairs] == [record["pair_id"]]


def test_full_resolution_qc_rejects_mismatching_edges(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before = archive / "institution" / "layout" / "before.png"
    after = archive / "institution" / "layout" / "pp" / "after.png"
    _save_rgb(before, _texture(50))
    _save_rgb(after, _texture(51))
    record = _record("p_00000000000000000050", "g_wrong", before, after)
    record["geometry"] = _geometry()

    plan = build_materialization_plan(
        [record],
        tmp_path / "local" / "dataset",
        [archive],
        alignment_min_overlap_ratio=0.90,
        alignment_min_edge_correlation=0.80,
        reserve_fraction=0.0,
        minimum_reserve_bytes=0,
    )

    assert plan.pairs == ()
    assert plan.excluded_counts == {"alignment_qc_edges": 1}
    assert all(str(archive) not in reason for reason in plan.excluded_counts)
    with pytest.raises(DatasetError, match="empty"):
        materialize(plan)


def test_missing_invalid_and_unreadable_geometry_inputs_are_aggregated(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    records = []
    for index in range(3):
        before = archive / "institution" / f"layout-{index}" / "before.png"
        after = archive / "institution" / f"layout-{index}" / "pp" / "after.png"
        before.parent.mkdir(parents=True)
        after.parent.mkdir(parents=True)
        before.write_bytes(b"not-an-image")
        after.write_bytes(b"not-an-image")
        records.append(_record(f"p_{index + 60:020x}", f"g_{index}", before, after))
    records[1]["geometry"] = _geometry([1.0, 0.0])
    records[2]["geometry"] = _geometry()

    plan = build_materialization_plan(
        records,
        tmp_path / "local" / "dataset",
        [archive],
        reserve_fraction=0.0,
        minimum_reserve_bytes=0,
    )

    assert plan.pairs == ()
    assert plan.excluded_counts == {
        "alignment_read_error": 1,
        "invalid_geometry": 1,
        "missing_geometry": 1,
    }


def test_materialize_cli_is_dry_run_by_default(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before = archive / "before.jpg"
    after = archive / "pp" / "after.jpg"
    before.parent.mkdir(parents=True)
    after.parent.mkdir(parents=True)
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    workspace = tmp_path / "private"
    destination = workspace / "dataset"
    config = tmp_path / "local.yaml"
    config.write_text(
        f'archive_roots: ["{archive.as_posix()}"]\n'
        f'workspace: "{workspace.as_posix()}"\n'
        "materialize:\n"
        "  mode: copy\n"
        f'  destination: "{destination.as_posix()}"\n',
        encoding="utf-8",
    )
    candidates = workspace / "manifests" / "candidates.jsonl"
    candidates.parent.mkdir(parents=True)
    candidates.write_text(
        json.dumps(_record("p_00000000000000000001", "g_one", before, after)) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["materialize", "--config", str(config), "--candidates", str(candidates)],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run only" in result.output
    assert not destination.exists()
