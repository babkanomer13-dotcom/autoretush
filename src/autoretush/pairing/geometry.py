from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from autoretush.pairing.fingerprint import read_bgr, resize_long_side


@dataclass(frozen=True)
class GeometryMetrics:
    prepared_before_height: int = 0
    prepared_before_width: int = 0
    prepared_after_height: int = 0
    prepared_after_width: int = 0
    keypoints_before: int = 0
    keypoints_after: int = 0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    source_coverage: float = 0.0
    target_coverage: float = 0.0
    edge_correlation: float = 0.0
    transform_valid: bool = False
    score: float = 0.0
    homography: tuple[float, ...] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedFrame:
    gray: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None


def prepare_frame(path: Path, max_long_side: int) -> PreparedFrame:
    image = resize_long_side(read_bgr(path), max_long_side)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02, edgeThreshold=15)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return PreparedFrame(gray=gray, keypoints=keypoints, descriptors=descriptors)


def _coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3 or width <= 0 or height <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / (width * height))


def _valid_transform(
    matrix: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> bool:
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False
    if abs(float(matrix[2, 2])) < 1e-9:
        return False
    matrix = matrix / matrix[2, 2]
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    corners = np.float32(
        [[0, 0], [source_w - 1, 0], [source_w - 1, source_h - 1], [0, source_h - 1]]
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    signed_area = float(cv2.contourArea(transformed.astype(np.float32), oriented=True))
    source_area = max(1.0, float(source_w * source_h))
    target_area = max(1.0, float(target_w * target_h))
    area_ratio = abs(signed_area) / min(source_area, target_area)
    if signed_area <= 0 or not 0.20 <= area_ratio <= 4.0:
        return False
    return not (abs(float(matrix[2, 0])) > 0.01 or abs(float(matrix[2, 1])) > 0.01)


def _edge_correlation(
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    homography: np.ndarray,
) -> float:
    target_h, target_w = target_gray.shape
    warped = cv2.warpPerspective(source_gray, homography, (target_w, target_h))
    valid = cv2.warpPerspective(
        np.full(source_gray.shape, 255, dtype=np.uint8), homography, (target_w, target_h)
    )
    valid_mask = valid > 250
    if int(valid_mask.sum()) < target_h * target_w * 0.15:
        return 0.0

    source_edges = cv2.Laplacian(warped, cv2.CV_32F, ksize=3)
    target_edges = cv2.Laplacian(target_gray, cv2.CV_32F, ksize=3)
    left = source_edges[valid_mask].astype(np.float64)
    right = target_edges[valid_mask].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-9:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def compare_prepared(source: PreparedFrame, target: PreparedFrame) -> GeometryMetrics:
    source_gray = source.gray
    target_gray = target.gray
    source_keypoints = source.keypoints
    target_keypoints = target.keypoints
    source_descriptors = source.descriptors
    target_descriptors = target.descriptors
    base = {
        "prepared_before_height": int(source_gray.shape[0]),
        "prepared_before_width": int(source_gray.shape[1]),
        "prepared_after_height": int(target_gray.shape[0]),
        "prepared_after_width": int(target_gray.shape[1]),
        "keypoints_before": len(source_keypoints),
        "keypoints_after": len(target_keypoints),
    }
    if source_descriptors is None or target_descriptors is None:
        return GeometryMetrics(**base, error="No SIFT descriptors")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(source_descriptors, target_descriptors, k=2)
    good = [first for first, second in raw_matches if first.distance < 0.76 * second.distance]
    if len(good) < 4:
        return GeometryMetrics(**base, good_matches=len(good), error="Too few keypoint matches")

    source_points = np.float32([source_keypoints[item.queryIdx].pt for item in good])
    target_points = np.float32([target_keypoints[item.trainIdx].pt for item in good])
    homography, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 4.0)
    if homography is None or mask is None:
        return GeometryMetrics(**base, good_matches=len(good), error="RANSAC failed")

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    source_inliers = source_points[inlier_mask]
    target_inliers = target_points[inlier_mask]
    source_coverage = _coverage(source_inliers, source_gray.shape[1], source_gray.shape[0])
    target_coverage = _coverage(target_inliers, target_gray.shape[1], target_gray.shape[0])
    transform_valid = _valid_transform(homography, source_gray.shape, target_gray.shape)
    edge_correlation = (
        _edge_correlation(source_gray, target_gray, homography) if transform_valid else 0.0
    )

    inlier_strength = min(1.0, inliers / 80.0)
    coverage_strength = min(1.0, min(source_coverage, target_coverage) / 0.25)
    edge_strength = float(np.clip((edge_correlation + 0.10) / 1.10, 0.0, 1.0))
    score = (
        0.30 * inlier_strength
        + 0.30 * inlier_ratio
        + 0.20 * coverage_strength
        + 0.20 * edge_strength
    )
    if not transform_valid:
        score *= 0.25

    return GeometryMetrics(
        **base,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=round(inlier_ratio, 6),
        source_coverage=round(source_coverage, 6),
        target_coverage=round(target_coverage, 6),
        edge_correlation=round(edge_correlation, 6),
        transform_valid=transform_valid,
        score=round(float(score), 6),
        homography=tuple(round(float(value), 9) for value in homography.flatten()),
    )


def compare_geometry(before: Path, after: Path, *, max_long_side: int) -> GeometryMetrics:
    try:
        source = prepare_frame(before, max_long_side)
        target = prepare_frame(after, max_long_side)
    except Exception as exc:  # OpenCV errors are converted into a local report.
        return GeometryMetrics(error=str(exc))
    return compare_prepared(source, target)
