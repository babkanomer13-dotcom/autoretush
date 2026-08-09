from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from autoretush.identifiers import validate_pair_id
from autoretush.pairing.fingerprint import read_bgr
from autoretush.pairing.geometry import GeometryMetrics

ReferenceFrame = Literal["before", "after"]
LosslessFormat = Literal["png", "tiff"]


class AlignmentError(RuntimeError):
    """Raised when a confirmed pair cannot be aligned safely."""


class AlignmentQualityError(AlignmentError):
    """Raised when materialization is attempted for an alignment that failed QC."""


class UnsafeCacheLocationError(AlignmentError):
    """Raised when cache output could overlap a source archive."""


@dataclass(frozen=True)
class AlignmentThresholds:
    min_overlap_ratio: float = 0.50
    min_edge_correlation: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_overlap_ratio <= 1.0:
            raise ValueError("min_overlap_ratio must be between 0 and 1")
        if not -1.0 <= self.min_edge_correlation <= 1.0:
            raise ValueError("min_edge_correlation must be between -1 and 1")


@dataclass(frozen=True)
class AlignmentQC:
    overlap_ratio: float
    edge_correlation: float
    valid_pixels: int
    output_pixels: int
    min_overlap_ratio: float
    min_edge_correlation: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class AlignedPair:
    before_path: Path
    after_path: Path
    reference: ReferenceFrame
    homography_before_to_after: tuple[float, ...]
    qc: AlignmentQC
    before_bgr: np.ndarray = field(repr=False)
    after_bgr: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class AlignmentCache:
    directory: Path
    before: Path
    after: Path
    valid_mask: Path
    metadata: Path
    qc: AlignmentQC


def _normalized_homography(metrics: GeometryMetrics) -> np.ndarray:
    if not metrics.transform_valid:
        raise AlignmentError("GeometryMetrics does not contain a validated transform")
    if metrics.error is not None:
        raise AlignmentError(f"GeometryMetrics contains an error: {metrics.error}")
    if metrics.homography is None:
        raise AlignmentError("GeometryMetrics does not contain a homography")

    matrix = np.asarray(metrics.homography, dtype=np.float64)
    if matrix.size != 9:
        raise AlignmentError("GeometryMetrics homography must contain exactly 9 values")
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all() or abs(float(matrix[2, 2])) < 1e-12:
        raise AlignmentError("GeometryMetrics homography is not finite or normalizable")
    matrix = matrix / matrix[2, 2]
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise AlignmentError("GeometryMetrics homography is singular")
    return matrix


def _comparison_scale(shape: tuple[int, int], max_long_side: int) -> np.ndarray:
    if max_long_side <= 0:
        raise ValueError("geometry_max_long_side must be positive")
    height, width = shape
    if height <= 0 or width <= 0:
        raise AlignmentError("Image dimensions must be positive")
    scale = min(1.0, max_long_side / max(height, width))
    prepared_width = max(1, round(width * scale))
    prepared_height = max(1, round(height * scale))
    return np.diag([prepared_width / width, prepared_height / height, 1.0]).astype(np.float64)


def _stored_comparison_scale(
    full_shape: tuple[int, int],
    prepared_shape: tuple[int, int],
) -> np.ndarray:
    full_height, full_width = full_shape
    prepared_height, prepared_width = prepared_shape
    if min(full_height, full_width, prepared_height, prepared_width) <= 0:
        raise AlignmentError("Stored prepared image dimensions must be positive")
    width_scale = prepared_width / full_width
    height_scale = prepared_height / full_height
    rounding_tolerance = max(1.0 / full_width, 1.0 / full_height) + 1e-9
    if width_scale > 1.0 + rounding_tolerance or height_scale > 1.0 + rounding_tolerance:
        raise AlignmentError("Stored prepared image dimensions exceed the source image")
    if abs(width_scale - height_scale) > rounding_tolerance:
        raise AlignmentError("Stored prepared image dimensions do not preserve aspect ratio")
    return np.diag([width_scale, height_scale, 1.0]).astype(np.float64)


def full_resolution_homography(
    metrics: GeometryMetrics,
    *,
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    geometry_max_long_side: int,
) -> np.ndarray:
    """Convert the stored before-to-after homography to original image coordinates."""

    prepared_matrix = _normalized_homography(metrics)
    stored_dimensions = (
        metrics.prepared_before_height,
        metrics.prepared_before_width,
        metrics.prepared_after_height,
        metrics.prepared_after_width,
    )
    if all(value > 0 for value in stored_dimensions):
        before_scale = _stored_comparison_scale(
            before_shape,
            (metrics.prepared_before_height, metrics.prepared_before_width),
        )
        after_scale = _stored_comparison_scale(
            after_shape,
            (metrics.prepared_after_height, metrics.prepared_after_width),
        )
    elif any(value != 0 for value in stored_dimensions):
        raise AlignmentError("Stored prepared image dimensions are incomplete")
    else:
        before_scale = _comparison_scale(before_shape, geometry_max_long_side)
        after_scale = _comparison_scale(after_shape, geometry_max_long_side)
    matrix = np.linalg.inv(after_scale) @ prepared_matrix @ before_scale
    if not np.isfinite(matrix).all() or abs(float(matrix[2, 2])) < 1e-12:
        raise AlignmentError("Full-resolution homography is not finite or normalizable")
    matrix = matrix / matrix[2, 2]
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        raise AlignmentError("Full-resolution homography is singular")
    return matrix


def _valid_source_mask(
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    output_size: tuple[int, int],
    *,
    mask_border: int,
) -> np.ndarray:
    if mask_border < 0:
        raise ValueError("mask_border must not be negative")
    source_mask = np.full(source_shape, 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        source_mask,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if mask_border:
        size = 2 * mask_border + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        mask = cv2.erode(
            mask,
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _edge_correlation(left_bgr: np.ndarray, right_bgr: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if int(valid.sum()) < 16:
        return 0.0

    def laplacian(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 0.8)
        return cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

    left = laplacian(left_bgr)[valid].astype(np.float64)
    right = laplacian(right_bgr)[valid].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _quality_control(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: AlignmentThresholds,
) -> AlignmentQC:
    output_pixels = int(valid_mask.size)
    valid_pixels = int(np.count_nonzero(valid_mask))
    overlap_ratio = valid_pixels / output_pixels if output_pixels else 0.0
    edge_correlation = _edge_correlation(before_bgr, after_bgr, valid_mask)
    failures: list[str] = []
    if overlap_ratio < thresholds.min_overlap_ratio:
        failures.append("overlap_ratio")
    if edge_correlation < thresholds.min_edge_correlation:
        failures.append("edge_correlation")
    return AlignmentQC(
        overlap_ratio=float(overlap_ratio),
        edge_correlation=edge_correlation,
        valid_pixels=valid_pixels,
        output_pixels=output_pixels,
        min_overlap_ratio=thresholds.min_overlap_ratio,
        min_edge_correlation=thresholds.min_edge_correlation,
        failures=tuple(failures),
    )


def align_confirmed_pair(
    before_path: Path,
    after_path: Path,
    geometry: GeometryMetrics,
    *,
    geometry_max_long_side: int,
    reference: ReferenceFrame = "after",
    thresholds: AlignmentThresholds | None = None,
    mask_border: int = 2,
) -> AlignedPair:
    """Align a previously confirmed same-frame pair without modifying either source."""

    if reference not in ("before", "after"):
        raise ValueError("reference must be 'before' or 'after'")
    thresholds = thresholds or AlignmentThresholds()
    try:
        before_bgr = read_bgr(before_path)
        after_bgr = read_bgr(after_path)
    except Exception as exc:
        raise AlignmentError(f"Cannot read confirmed pair: {exc}") from exc

    before_shape = before_bgr.shape[:2]
    after_shape = after_bgr.shape[:2]
    before_to_after = full_resolution_homography(
        geometry,
        before_shape=before_shape,
        after_shape=after_shape,
        geometry_max_long_side=geometry_max_long_side,
    )

    if reference == "after":
        output_height, output_width = after_shape
        aligned_before = cv2.warpPerspective(
            before_bgr,
            before_to_after,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aligned_after = after_bgr.copy()
        valid_mask = _valid_source_mask(
            before_shape,
            before_to_after,
            (output_width, output_height),
            mask_border=mask_border,
        )
    else:
        try:
            after_to_before = np.linalg.inv(before_to_after)
        except np.linalg.LinAlgError as exc:
            raise AlignmentError("Full-resolution homography cannot be inverted") from exc
        output_height, output_width = before_shape
        aligned_before = before_bgr.copy()
        aligned_after = cv2.warpPerspective(
            after_bgr,
            after_to_before,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = _valid_source_mask(
            after_shape,
            after_to_before,
            (output_width, output_height),
            mask_border=mask_border,
        )

    qc = _quality_control(aligned_before, aligned_after, valid_mask, thresholds)
    return AlignedPair(
        before_path=before_path,
        after_path=after_path,
        reference=reference,
        homography_before_to_after=tuple(float(value) for value in before_to_after.flat),
        qc=qc,
        before_bgr=aligned_before,
        after_bgr=aligned_after,
        valid_mask=valid_mask,
    )


def _is_within(path: Path, root: Path) -> bool:
    normalized_path = os.path.normcase(str(path.resolve(strict=False)))
    normalized_root = os.path.normcase(str(root.resolve(strict=False)))
    try:
        return os.path.commonpath((normalized_path, normalized_root)) == normalized_root
    except ValueError:  # Different Windows drives cannot overlap.
        return False


def _validate_cache_location(
    aligned: AlignedPair,
    cache_root: Path,
    archive_roots: Iterable[Path],
) -> tuple[Path, tuple[Path, ...]]:
    roots = tuple(Path(root).resolve(strict=False) for root in archive_roots)
    if not roots:
        raise UnsafeCacheLocationError("At least one source archive root must be declared")
    resolved_cache = cache_root.resolve(strict=False)
    if any(_is_within(resolved_cache, root) or _is_within(root, resolved_cache) for root in roots):
        raise UnsafeCacheLocationError(
            f"Alignment cache must not overlap any source archive: {resolved_cache}"
        )
    for source in (aligned.before_path, aligned.after_path):
        resolved_source = source.resolve(strict=False)
        if not any(_is_within(resolved_source, root) for root in roots):
            raise UnsafeCacheLocationError(
                f"Source is outside the declared archive roots: {resolved_source}"
            )
    return resolved_cache, roots


def _normalize_format(image_format: str) -> tuple[LosslessFormat, str]:
    normalized = image_format.casefold().lstrip(".")
    if normalized == "png":
        return "png", ".png"
    if normalized in {"tif", "tiff"}:
        return "tiff", ".tiff"
    raise ValueError("image_format must be PNG or TIFF")


def _write_lossless(path: Path, image: np.ndarray, image_format: LosslessFormat) -> None:
    if image_format == "png":
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        extension = ".png"
    else:
        parameters = [cv2.IMWRITE_TIFF_COMPRESSION, 5]
        extension = ".tiff"
    try:
        success, encoded = cv2.imencode(extension, image, parameters)
    except cv2.error as exc:
        raise AlignmentError(f"Cannot encode lossless cache image: {path.name}") from exc
    if not success:
        raise AlignmentError(f"Cannot encode lossless cache image: {path.name}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())
        handle.flush()
        os.fsync(handle.fileno())


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def save_alignment_cache(
    aligned: AlignedPair,
    *,
    pair_id: str,
    cache_root: Path,
    archive_roots: Iterable[Path],
    image_format: str = "png",
    require_qc: bool = True,
) -> AlignmentCache:
    """Atomically publish an immutable lossless pair cache outside source archives."""

    pair_id = validate_pair_id(pair_id)
    if require_qc and not aligned.qc.passed:
        rendered = ", ".join(aligned.qc.failures)
        raise AlignmentQualityError(f"Alignment failed quality control: {rendered}")
    normalized_format, extension = _normalize_format(image_format)
    resolved_cache, roots = _validate_cache_location(aligned, cache_root, archive_roots)
    resolved_cache.mkdir(parents=True, exist_ok=True)
    resolved_cache = resolved_cache.resolve(strict=True)
    if any(_is_within(resolved_cache, root) or _is_within(root, resolved_cache) for root in roots):
        raise UnsafeCacheLocationError(
            f"Alignment cache resolved overlapping a source archive: {resolved_cache}"
        )

    destination = resolved_cache / pair_id
    if destination.exists():
        raise FileExistsError(f"Alignment cache already exists: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{pair_id}.", dir=resolved_cache))
    try:
        before_name = f"before{extension}"
        after_name = f"after{extension}"
        mask_name = f"valid_mask{extension}"
        _write_lossless(staging / before_name, aligned.before_bgr, normalized_format)
        _write_lossless(staging / after_name, aligned.after_bgr, normalized_format)
        _write_lossless(staging / mask_name, aligned.valid_mask, normalized_format)
        metadata_payload: dict[str, object] = {
            "schema_version": 1,
            "reference": aligned.reference,
            "image_format": normalized_format,
            "width": int(aligned.before_bgr.shape[1]),
            "height": int(aligned.before_bgr.shape[0]),
            "homography_before_to_after": list(aligned.homography_before_to_after),
            "qc": {**asdict(aligned.qc), "passed": aligned.qc.passed},
            "files": {
                "before": before_name,
                "after": after_name,
                "valid_mask": mask_name,
            },
        }
        _write_metadata(staging / "metadata.json", metadata_payload)
        if destination.exists():
            raise FileExistsError(f"Alignment cache already exists: {destination}")
        os.rename(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return AlignmentCache(
        directory=destination,
        before=destination / before_name,
        after=destination / after_name,
        valid_mask=destination / mask_name,
        metadata=destination / "metadata.json",
        qc=aligned.qc,
    )


def align_and_cache(
    before_path: Path,
    after_path: Path,
    geometry: GeometryMetrics,
    *,
    geometry_max_long_side: int,
    pair_id: str,
    cache_root: Path,
    archive_roots: Iterable[Path],
    reference: ReferenceFrame = "after",
    thresholds: AlignmentThresholds | None = None,
    mask_border: int = 2,
    image_format: str = "png",
) -> AlignmentCache:
    """Align one confirmed pair and atomically save it when QC passes."""

    aligned = align_confirmed_pair(
        before_path,
        after_path,
        geometry,
        geometry_max_long_side=geometry_max_long_side,
        reference=reference,
        thresholds=thresholds,
        mask_border=mask_border,
    )
    return save_alignment_cache(
        aligned,
        pair_id=pair_id,
        cache_root=cache_root,
        archive_roots=archive_roots,
        image_format=image_format,
    )
