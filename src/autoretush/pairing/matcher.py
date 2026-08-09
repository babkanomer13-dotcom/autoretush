from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from autoretush.config import PairingConfig
from autoretush.inventory import PairGroup
from autoretush.pairing.fingerprint import FrameFingerprint, fingerprint, hamming_distance
from autoretush.pairing.geometry import (
    GeometryMetrics,
    PreparedFrame,
    compare_prepared,
    prepare_frame,
)


@dataclass(frozen=True)
class PairCandidate:
    pair_id: str
    group_id: str
    before_path: Path
    after_path: Path
    decision: str
    score: float
    phash_distance: int
    aspect_ratio_delta: float
    geometry: GeometryMetrics

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["before_path"] = str(self.before_path)
        result["after_path"] = str(self.after_path)
        return result


def _pair_id(group_id: str, before: Path, after: Path) -> str:
    payload = f"{group_id}\0{before}\0{after}".encode()
    return "p_" + hashlib.sha256(payload).hexdigest()[:20]


def _decision(metrics: GeometryMetrics, phash_distance: int, config: PairingConfig) -> str:
    strict_geometry = (
        metrics.transform_valid
        and metrics.inliers >= config.min_keypoint_inliers
        and metrics.inlier_ratio >= config.min_inlier_ratio
        and min(metrics.source_coverage, metrics.target_coverage) >= config.min_source_coverage
        and phash_distance <= config.max_phash_distance
    )
    if strict_geometry and metrics.score >= config.auto_accept_score:
        return "accepted"
    if metrics.transform_valid and metrics.score >= config.review_score:
        return "review"
    return "rejected"


def _fingerprints(paths: list[Path]) -> tuple[dict[Path, FrameFingerprint], dict[Path, str]]:
    records: dict[Path, FrameFingerprint] = {}
    errors: dict[Path, str] = {}
    for path in paths:
        try:
            records[path] = fingerprint(path)
        except Exception as exc:
            errors[path] = str(exc)
    return records, errors


def _prepared(
    paths: list[Path], *, max_long_side: int
) -> tuple[dict[Path, PreparedFrame], dict[Path, str]]:
    records: dict[Path, PreparedFrame] = {}
    errors: dict[Path, str] = {}
    for path in paths:
        try:
            records[path] = prepare_frame(path, max_long_side)
        except Exception as exc:
            errors[path] = str(exc)
    return records, errors


def match_group(group: PairGroup, config: PairingConfig) -> list[PairCandidate]:
    before_paths = [item.path for item in group.before]
    after_paths = [item.path for item in group.after]
    before_fp, _ = _fingerprints(before_paths)
    after_fp, _ = _fingerprints(after_paths)
    usable_before = [path for path in before_paths if path in before_fp]
    usable_after = [path for path in after_paths if path in after_fp]
    if not usable_before or not usable_after:
        return []

    prepared_before, _ = _prepared(usable_before, max_long_side=config.max_long_side)
    prepared_after, _ = _prepared(usable_after, max_long_side=config.max_long_side)
    usable_before = [path for path in usable_before if path in prepared_before]
    usable_after = [path for path in usable_after if path in prepared_after]
    if not usable_before or not usable_after:
        return []

    score_matrix = np.full((len(usable_before), len(usable_after)), -1.0, dtype=np.float64)
    computed: dict[tuple[int, int], tuple[int, float, GeometryMetrics]] = {}

    for before_index, before_path in enumerate(usable_before):
        source = before_fp[before_path]
        rankings: list[tuple[int, float, int]] = []
        for after_index, after_path in enumerate(usable_after):
            target = after_fp[after_path]
            ratio_delta = abs(source.aspect_ratio - target.aspect_ratio) / max(
                source.aspect_ratio, target.aspect_ratio
            )
            distance = hamming_distance(source.phash, target.phash)
            rankings.append((distance, ratio_delta, after_index))

        rankings.sort(key=lambda item: (item[0], item[1]))
        for phash_distance, ratio_delta, after_index in rankings[: config.shortlist_size]:
            if ratio_delta > 0.25 or phash_distance > max(40, config.max_phash_distance + 8):
                continue
            after_path = usable_after[after_index]
            geometry = compare_prepared(
                prepared_before[before_path],
                prepared_after[after_path],
            )
            # pHash is a shortlist feature; geometry dominates the final score.
            phash_similarity = max(0.0, 1.0 - phash_distance / 64.0)
            combined_score = 0.90 * geometry.score + 0.10 * phash_similarity
            if geometry.transform_valid:
                score_matrix[before_index, after_index] = combined_score
            computed[(before_index, after_index)] = (
                phash_distance,
                ratio_delta,
                GeometryMetrics(**{**geometry.to_dict(), "score": round(combined_score, 6)}),
            )

    if not np.any(score_matrix >= 0.0):
        return []

    row_indices, column_indices = linear_sum_assignment(-score_matrix)
    results: list[PairCandidate] = []
    for before_index, after_index in zip(
        row_indices.tolist(), column_indices.tolist(), strict=True
    ):
        key = (before_index, after_index)
        if key not in computed or score_matrix[before_index, after_index] < 0:
            continue
        phash_distance, ratio_delta, geometry = computed[key]
        before_path = usable_before[before_index]
        after_path = usable_after[after_index]
        results.append(
            PairCandidate(
                pair_id=_pair_id(group.group_id, before_path, after_path),
                group_id=group.group_id,
                before_path=before_path,
                after_path=after_path,
                decision=_decision(geometry, phash_distance, config),
                score=geometry.score,
                phash_distance=phash_distance,
                aspect_ratio_delta=round(ratio_delta, 6),
                geometry=geometry,
            )
        )
    return sorted(results, key=lambda item: item.score, reverse=True)
