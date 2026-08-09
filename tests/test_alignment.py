import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import autoretush.alignment as alignment_module
from autoretush.alignment import (
    AlignmentError,
    AlignmentQualityError,
    AlignmentThresholds,
    UnsafeCacheLocationError,
    align_confirmed_pair,
    full_resolution_homography,
    save_alignment_cache,
)
from autoretush.pairing.geometry import GeometryMetrics


def _pair_id(value: int) -> str:
    return f"p_{value:020x}"


def _scene(height: int = 240, width: int = 320, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(10, 55, (height, width, 3), dtype=np.uint8)
    for index in range(18):
        color = tuple(int(value) for value in rng.integers(70, 245, 3))
        center = tuple(
            int(value)
            for value in rng.integers([12, 12], [max(13, width - 12), max(13, height - 12)])
        )
        cv2.circle(image, center, int(rng.integers(4, 18)), color, -1)
        cv2.putText(
            image,
            str(index),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    assert success
    encoded.tofile(path)


def _read_unchanged(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    return decoded


def _scale_matrix(shape: tuple[int, int], max_long_side: int) -> np.ndarray:
    height, width = shape
    scale = min(1.0, max_long_side / max(height, width))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return np.diag([resized_width / width, resized_height / height, 1.0])


def _metrics_for_full_transform(
    full_matrix: np.ndarray,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    max_long_side: int,
) -> GeometryMetrics:
    before_scale = _scale_matrix(before_shape, max_long_side)
    after_scale = _scale_matrix(after_shape, max_long_side)
    prepared_matrix = after_scale @ full_matrix @ np.linalg.inv(before_scale)
    prepared_matrix /= prepared_matrix[2, 2]
    return GeometryMetrics(
        prepared_before_height=int(round(before_shape[0] * before_scale[1, 1])),
        prepared_before_width=int(round(before_shape[1] * before_scale[0, 0])),
        prepared_after_height=int(round(after_shape[0] * after_scale[1, 1])),
        prepared_after_width=int(round(after_shape[1] * after_scale[0, 0])),
        transform_valid=True,
        homography=tuple(float(value) for value in prepared_matrix.flat),
    )


def _identity_metrics() -> GeometryMetrics:
    return GeometryMetrics(
        transform_valid=True,
        homography=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )


def test_unicode_paths_and_scaled_homography_align_to_after_geometry(tmp_path: Path) -> None:
    archive = tmp_path / "архив 2025–2026" / "дети на разворот"
    before_path = archive / "до ретуши № 01.png"
    after_path = archive / "pp" / "готово № 01.png"
    before = _scene()
    after_shape = (260, 340)
    full_matrix = np.array([[1.0, 0.0, 13.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    after = cv2.warpPerspective(before, full_matrix, (after_shape[1], after_shape[0]))
    _write(before_path, before)
    _write(after_path, after)
    geometry_max_long_side = 157
    metrics = _metrics_for_full_transform(
        full_matrix,
        before.shape[:2],
        after.shape[:2],
        geometry_max_long_side,
    )

    recovered = full_resolution_homography(
        metrics,
        before_shape=before.shape[:2],
        after_shape=after.shape[:2],
        geometry_max_long_side=geometry_max_long_side,
    )
    recovered_with_different_current_limit = full_resolution_homography(
        metrics,
        before_shape=before.shape[:2],
        after_shape=after.shape[:2],
        geometry_max_long_side=83,
    )
    aligned = align_confirmed_pair(
        before_path,
        after_path,
        metrics,
        geometry_max_long_side=geometry_max_long_side,
        reference="after",
    )

    assert np.allclose(recovered, full_matrix, atol=1e-9)
    assert np.allclose(recovered_with_different_current_limit, full_matrix, atol=1e-9)
    assert aligned.before_bgr.shape == aligned.after_bgr.shape == after.shape
    assert aligned.valid_mask.shape == after.shape[:2]
    assert aligned.valid_mask.dtype == np.uint8
    assert set(np.unique(aligned.valid_mask)).issubset({0, 255})
    valid = aligned.valid_mask > 0
    assert np.max(np.abs(aligned.before_bgr[valid].astype(int) - after[valid].astype(int))) <= 1
    assert aligned.qc.overlap_ratio > 0.75
    assert aligned.qc.edge_correlation > 0.999
    assert aligned.qc.passed


def test_reference_before_warps_after_into_before_geometry(tmp_path: Path) -> None:
    archive = tmp_path / "архив"
    before_path = archive / "до.png"
    after_path = archive / "pp" / "после.png"
    image = _scene(180, 260)
    _write(before_path, image)
    _write(after_path, image)

    aligned = align_confirmed_pair(
        before_path,
        after_path,
        _identity_metrics(),
        geometry_max_long_side=100,
        reference="before",
        mask_border=0,
    )

    assert aligned.reference == "before"
    assert np.array_equal(aligned.before_bgr, image)
    assert np.array_equal(aligned.after_bgr, image)
    assert np.all(aligned.valid_mask == 255)
    assert aligned.qc.passed


def test_overlap_and_edge_quality_failures_are_reported(tmp_path: Path) -> None:
    archive = tmp_path / "архив"
    before_path = archive / "до.png"
    after_path = archive / "pp" / "после.png"
    _write(before_path, _scene(seed=2))
    _write(after_path, _scene(seed=99))
    translated = np.array([[1.0, 0.0, 250.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    metrics = _metrics_for_full_transform(translated, (240, 320), (240, 320), 160)

    aligned = align_confirmed_pair(
        before_path,
        after_path,
        metrics,
        geometry_max_long_side=160,
        thresholds=AlignmentThresholds(
            min_overlap_ratio=0.50,
            min_edge_correlation=0.90,
        ),
    )

    assert aligned.qc.overlap_ratio < 0.25
    assert set(aligned.qc.failures) == {"overlap_ratio", "edge_correlation"}
    assert not aligned.qc.passed


@pytest.mark.parametrize("image_format, extension", [("png", ".png"), ("tif", ".tiff")])
def test_lossless_cache_is_atomic_and_outside_archive(
    tmp_path: Path,
    image_format: str,
    extension: str,
) -> None:
    archive = tmp_path / "исходный архив"
    cache_root = tmp_path / "локальный кэш"
    before_path = archive / "кадр.png"
    after_path = archive / "pp" / "результат.png"
    image = _scene(120, 160)
    _write(before_path, image)
    _write(after_path, cv2.convertScaleAbs(image, alpha=1.02, beta=3))
    aligned = align_confirmed_pair(
        before_path,
        after_path,
        _identity_metrics(),
        geometry_max_long_side=100,
        mask_border=0,
    )
    source_bytes = (before_path.read_bytes(), after_path.read_bytes())

    cached = save_alignment_cache(
        aligned,
        pair_id=_pair_id(1 if image_format == "png" else 2),
        cache_root=cache_root,
        archive_roots=(archive,),
        image_format=image_format,
    )

    assert cached.directory.parent == cache_root.resolve()
    assert cached.before.suffix == cached.after.suffix == cached.valid_mask.suffix == extension
    assert np.array_equal(_read_unchanged(cached.before), aligned.before_bgr)
    assert np.array_equal(_read_unchanged(cached.after), aligned.after_bgr)
    assert np.array_equal(_read_unchanged(cached.valid_mask), aligned.valid_mask)
    assert (before_path.read_bytes(), after_path.read_bytes()) == source_bytes
    metadata = json.loads(cached.metadata.read_text(encoding="utf-8"))
    assert metadata["qc"]["passed"] is True
    assert metadata["files"]["valid_mask"] == f"valid_mask{extension}"
    assert str(before_path) not in cached.metadata.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_alignment_cache(
            aligned,
            pair_id=_pair_id(1 if image_format == "png" else 2),
            cache_root=cache_root,
            archive_roots=(archive,),
            image_format=image_format,
        )


def test_cache_inside_archive_and_failed_qc_are_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    before_path = archive / "before.png"
    after_path = archive / "pp" / "after.png"
    image = _scene(100, 140)
    _write(before_path, image)
    _write(after_path, image)
    aligned = align_confirmed_pair(
        before_path,
        after_path,
        _identity_metrics(),
        geometry_max_long_side=100,
    )

    with pytest.raises(UnsafeCacheLocationError):
        save_alignment_cache(
            aligned,
            pair_id=_pair_id(3),
            cache_root=archive / "cache",
            archive_roots=(archive,),
        )
    assert not (archive / "cache").exists()
    with pytest.raises(UnsafeCacheLocationError):
        save_alignment_cache(
            aligned,
            pair_id=_pair_id(4),
            cache_root=tmp_path,
            archive_roots=(archive,),
        )

    failed = align_confirmed_pair(
        before_path,
        after_path,
        _identity_metrics(),
        geometry_max_long_side=100,
        thresholds=AlignmentThresholds(min_edge_correlation=1.0),
    )
    assert not failed.qc.passed
    with pytest.raises(AlignmentQualityError):
        save_alignment_cache(
            failed,
            pair_id=_pair_id(5),
            cache_root=tmp_path / "cache",
            archive_roots=(archive,),
        )


def test_partial_staging_is_removed_when_encoding_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    before_path = archive / "before.png"
    after_path = archive / "pp" / "after.png"
    image = _scene(100, 140)
    _write(before_path, image)
    _write(after_path, image)
    aligned = align_confirmed_pair(
        before_path,
        after_path,
        _identity_metrics(),
        geometry_max_long_side=100,
    )
    cache_root = tmp_path / "cache"
    original_write = alignment_module._write_lossless
    calls = 0

    def fail_on_second_write(path: Path, data: np.ndarray, image_format: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AlignmentError("simulated encoder failure")
        original_write(path, data, image_format)  # type: ignore[arg-type]

    monkeypatch.setattr(alignment_module, "_write_lossless", fail_on_second_write)

    with pytest.raises(AlignmentError, match="simulated"):
        save_alignment_cache(
            aligned,
            pair_id=_pair_id(6),
            cache_root=cache_root,
            archive_roots=(archive,),
        )

    assert not (cache_root / _pair_id(6)).exists()
    assert list(cache_root.iterdir()) == []


@pytest.mark.parametrize(
    "metrics",
    [
        GeometryMetrics(),
        GeometryMetrics(transform_valid=True, homography=None),
        GeometryMetrics(transform_valid=True, homography=(1.0,) * 8),
        GeometryMetrics(
            transform_valid=True,
            homography=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    ],
)
def test_invalid_geometry_is_not_used_for_alignment(
    tmp_path: Path,
    metrics: GeometryMetrics,
) -> None:
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    image = _scene(80, 120)
    _write(before_path, image)
    _write(after_path, image)

    with pytest.raises(AlignmentError):
        align_confirmed_pair(
            before_path,
            after_path,
            metrics,
            geometry_max_long_side=100,
        )
