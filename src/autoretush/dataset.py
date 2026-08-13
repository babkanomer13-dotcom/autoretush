from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from autoretush.alignment import AlignmentError, AlignmentThresholds, align_confirmed_pair
from autoretush.identifiers import validate_pair_id
from autoretush.pairing.fingerprint import ImageReadError
from autoretush.pairing.geometry import GeometryMetrics


class DatasetError(ValueError):
    """Raised when a dataset operation would be incomplete or unsafe."""


REVIEW_VALUES = frozenset({"match", "wrong", "unsure"})
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SelectedPair:
    pair_id: str
    group_id: str
    before_path: Path
    after_path: Path
    score: float
    source_decision: str
    review_decision: str | None
    geometry: GeometryMetrics | None = None
    homography_before_to_after: tuple[float, ...] | None = None
    split_group_id: str = ""
    split: str = ""


@dataclass(frozen=True)
class MaterializationPlan:
    destination: Path
    pairs: tuple[SelectedPair, ...]
    source_bytes: int
    required_free_bytes: int
    geometry_max_long_side: int
    alignment_min_overlap_ratio: float
    alignment_min_edge_correlation: float
    split_counts: dict[str, int]
    excluded_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "destination": str(self.destination),
            "pair_count": len(self.pairs),
            "source_bytes": self.source_bytes,
            "required_free_bytes": self.required_free_bytes,
            "geometry_max_long_side": self.geometry_max_long_side,
            "alignment_qc": {
                "min_overlap_ratio": self.alignment_min_overlap_ratio,
                "min_edge_correlation": self.alignment_min_edge_correlation,
            },
            "split_counts": self.split_counts,
            "excluded_counts": self.excluded_counts,
        }


def load_review_answers(path: Path) -> dict[str, str]:
    """Load the small operator-review JSON without accepting unknown values."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"Review result does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid review JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DatasetError("Unsupported review result schema")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise DatasetError("Review result must contain an answers object")
    result: dict[str, str] = {}
    for raw_pair_id, value in answers.items():
        if not isinstance(value, str):
            raise DatasetError("Review answers must map string pair IDs to string decisions")
        try:
            pair_id = validate_pair_id(raw_pair_id)
        except ValueError as exc:
            raise DatasetError("Review result contains an unsafe pair_id") from exc
        if value not in REVIEW_VALUES:
            raise DatasetError(f"Unsupported review decision for {pair_id}: {value}")
        result[pair_id] = value
    return result


def _record_path(record: Mapping[str, object], field: str) -> Path:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise DatasetError(f"Candidate has no usable {field}")
    return Path(value)


def _geometry(record: Mapping[str, object], pair_id: str) -> GeometryMetrics | None:
    value = record.get("geometry")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DatasetError(f"Candidate {pair_id} geometry must be an object")
    allowed = set(GeometryMetrics.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        raise DatasetError(f"Candidate {pair_id} geometry has unsupported fields")
    try:
        return GeometryMetrics(**dict(value))
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"Candidate {pair_id} geometry is invalid") from exc


def select_pairs(
    records: Iterable[Mapping[str, object]],
    *,
    review_answers: Mapping[str, str] | None = None,
) -> tuple[tuple[SelectedPair, ...], dict[str, int]]:
    """Select strict auto-accepts plus explicitly confirmed review candidates.

    A manual ``wrong`` or ``unsure`` decision always overrides an automatic accept.
    An unreviewed borderline candidate is never selected.
    """
    answers = review_answers or {}
    excluded: Counter[str] = Counter()
    selected: list[SelectedPair] = []
    seen_ids: set[str] = set()
    used_before: set[Path] = set()
    used_after: set[Path] = set()

    for record in records:
        raw_pair_id = record.get("pair_id")
        group_id = record.get("group_id")
        decision = record.get("decision")
        try:
            pair_id = validate_pair_id(raw_pair_id)
        except ValueError as exc:
            raise DatasetError("Candidate has no safe pair_id") from exc
        if pair_id in seen_ids:
            raise DatasetError(f"Duplicate pair_id in candidate manifest: {pair_id}")
        seen_ids.add(pair_id)
        if not isinstance(group_id, str) or not group_id:
            raise DatasetError(f"Candidate {pair_id} has no usable group_id")
        if decision not in {"accepted", "review", "rejected"}:
            raise DatasetError(f"Candidate {pair_id} has unsupported decision: {decision}")

        manual = answers.get(pair_id)
        if manual is not None and manual not in REVIEW_VALUES:
            raise DatasetError(f"Unsupported review decision for {pair_id}: {manual}")
        if manual in {"wrong", "unsure"}:
            excluded[f"manual_{manual}"] += 1
            continue
        if decision == "rejected":
            excluded["automatic_rejected"] += 1
            continue
        if decision == "review" and manual != "match":
            excluded["awaiting_review"] += 1
            continue

        before_path = _record_path(record, "before_path")
        after_path = _record_path(record, "after_path")
        try:
            score = float(record.get("score", 0.0))
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Candidate {pair_id} has a non-numeric score") from exc
        try:
            geometry = _geometry(record, pair_id)
        except DatasetError:
            excluded["invalid_geometry"] += 1
            continue
        before_key = before_path.resolve(strict=False)
        after_key = after_path.resolve(strict=False)
        if before_key in used_before:
            raise DatasetError(
                f"A before image occurs in more than one selected pair: {before_path}"
            )
        if after_key in used_after:
            raise DatasetError(
                f"An after image occurs in more than one selected pair: {after_path}"
            )
        used_before.add(before_key)
        used_after.add(after_key)
        selected.append(
            SelectedPair(
                pair_id=pair_id,
                group_id=group_id,
                before_path=before_path,
                after_path=after_path,
                score=score,
                source_decision=str(decision),
                review_decision=manual,
                geometry=geometry,
            )
        )

    unknown_answers = set(answers) - seen_ids
    if unknown_answers:
        raise DatasetError("Review result contains pair IDs absent from the candidate manifest")
    return tuple(selected), dict(sorted(excluded.items()))


def assign_group_splits(
    pairs: Iterable[SelectedPair],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: str = "autoretush-v1",
) -> tuple[SelectedPair, ...]:
    """Assign whole candidate groups to deterministic, pair-balanced splits."""
    if not 0.0 < train_ratio < 1.0:
        raise DatasetError("train_ratio must be between 0 and 1")
    if not 0.0 <= validation_ratio < 1.0 or train_ratio + validation_ratio >= 1.0:
        raise DatasetError("validation_ratio must leave a positive test ratio")
    pair_list = tuple(pairs)
    grouped: defaultdict[str, list[SelectedPair]] = defaultdict(list)
    for pair in pair_list:
        grouped[pair.split_group_id or pair.group_id].append(pair)
    total_pairs = len(pair_list)
    targets = {
        "train": total_pairs * train_ratio,
        "validation": total_pairs * validation_ratio,
        "test": total_pairs * (1.0 - train_ratio - validation_ratio),
    }
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            -len(grouped[key]),
            hashlib.sha256(f"{seed}\0{key}".encode()).digest(),
        ),
    )
    group_splits: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for index, split_key in enumerate(ordered_keys):
        empty = [split for split in SPLITS if split not in group_splits.values()]
        remaining_units = len(ordered_keys) - index
        candidates = empty if remaining_units == len(empty) else list(SPLITS)
        split = max(
            candidates,
            key=lambda name: (targets[name] - counts[name], -SPLITS.index(name)),
        )
        group_splits[split_key] = split
        counts[split] += len(grouped[split_key])

    result: list[SelectedPair] = []
    for pair in pair_list:
        split_key = pair.split_group_id or pair.group_id
        split = group_splits[split_key]
        result.append(
            SelectedPair(
                pair_id=pair.pair_id,
                group_id=pair.group_id,
                before_path=pair.before_path,
                after_path=pair.after_path,
                score=pair.score,
                source_decision=pair.source_decision,
                review_decision=pair.review_decision,
                geometry=pair.geometry,
                homography_before_to_after=pair.homography_before_to_after,
                split_group_id=pair.split_group_id,
                split=split,
            )
        )
    return tuple(result)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assign_archive_split_units(
    pairs: Iterable[SelectedPair], archive_roots: Iterable[Path]
) -> tuple[SelectedPair, ...]:
    """Keep the same top-level institution folder together across seasons."""
    roots = tuple(root.resolve(strict=False) for root in archive_roots)

    def archive_relative(path: Path) -> tuple[Path, tuple[str, ...]]:
        resolved = path.resolve(strict=False)
        matching = [root for root in roots if _inside(resolved, root)]
        if not matching:
            raise DatasetError(f"Selected source is outside every archive root: {path}")
        archive = max(matching, key=lambda root: len(root.parts))
        relative = resolved.relative_to(archive)
        if not relative.parts:
            raise DatasetError(f"Selected source cannot be an archive root: {path}")
        return archive, tuple(part.casefold() for part in relative.parts)

    result: list[SelectedPair] = []
    for pair in pairs:
        before_archive, before_relative = archive_relative(pair.before_path)
        after_archive, after_relative = archive_relative(pair.after_path)
        before_unit = before_relative[0] if len(before_relative) > 1 else pair.group_id
        crosses_unit = len(before_relative) > 1 and before_unit != after_relative[0]
        if before_archive != after_archive or crosses_unit:
            raise DatasetError(
                f"Selected pair {pair.pair_id} crosses archive or institution boundaries"
            )
        unit_label = before_unit
        split_group_id = "sg_" + hashlib.sha256(unit_label.encode("utf-8")).hexdigest()[:16]
        result.append(
            SelectedPair(
                pair_id=pair.pair_id,
                group_id=pair.group_id,
                before_path=pair.before_path,
                after_path=pair.after_path,
                score=pair.score,
                source_decision=pair.source_decision,
                review_decision=pair.review_decision,
                geometry=pair.geometry,
                homography_before_to_after=pair.homography_before_to_after,
                split_group_id=split_group_id,
            )
        )
    return tuple(result)


def validate_destination(destination: Path, archive_roots: Iterable[Path]) -> Path:
    """Refuse destinations that could overlap any read-only source archive."""
    resolved = destination.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise DatasetError("Dataset destination cannot be a drive or filesystem root")
    for root in archive_roots:
        archive = root.resolve(strict=False)
        if _inside(resolved, archive) or _inside(archive, resolved):
            raise DatasetError(f"Dataset destination overlaps source archive: {root}")
    return resolved


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise DatasetError(f"No existing ancestor for destination: {path}")
        current = parent
    return current


def _geometry_is_structurally_valid(geometry: GeometryMetrics) -> bool:
    """Perform a path-free check before the full-resolution alignment call."""
    if geometry.transform_valid is not True or geometry.error is not None:
        return False
    prepared_dimensions = (
        geometry.prepared_before_height,
        geometry.prepared_before_width,
        geometry.prepared_after_height,
        geometry.prepared_after_width,
    )
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in prepared_dimensions
    ):
        return False
    homography = geometry.homography
    if not isinstance(homography, (list, tuple)) or len(homography) != 9:
        return False
    try:
        values = tuple(float(value) for value in homography)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in values) or abs(values[8]) < 1e-12:
        return False
    a, b, c, d, e, f, g, h, i = values
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return abs(determinant) >= 1e-12


def _alignment_qc_reason(failures: tuple[str, ...]) -> str:
    failed = set(failures)
    if {"overlap_ratio", "edge_correlation"}.issubset(failed):
        return "alignment_qc_overlap_and_edges"
    if "overlap_ratio" in failed:
        return "alignment_qc_overlap"
    if "edge_correlation" in failed:
        return "alignment_qc_edges"
    return "alignment_qc_failed"


def _filter_by_full_resolution_alignment(
    pairs: Iterable[SelectedPair],
    *,
    geometry_max_long_side: int,
    thresholds: AlignmentThresholds,
) -> tuple[tuple[SelectedPair, ...], Counter[str]]:
    """Keep only pairs that pass a full-resolution overlap and edge check."""
    accepted: list[SelectedPair] = []
    excluded: Counter[str] = Counter()
    for pair in pairs:
        geometry = pair.geometry
        if geometry is None:
            excluded["missing_geometry"] += 1
            continue
        if not _geometry_is_structurally_valid(geometry):
            excluded["invalid_geometry"] += 1
            continue
        try:
            aligned = align_confirmed_pair(
                pair.before_path,
                pair.after_path,
                geometry,
                geometry_max_long_side=geometry_max_long_side,
                reference="after",
                thresholds=thresholds,
            )
        except AlignmentError as exc:
            if isinstance(exc.__cause__, ImageReadError):
                excluded["alignment_read_error"] += 1
            else:
                excluded["invalid_geometry"] += 1
            continue
        if not aligned.qc.passed:
            excluded[_alignment_qc_reason(aligned.qc.failures)] += 1
            continue
        accepted.append(
            SelectedPair(
                pair_id=pair.pair_id,
                group_id=pair.group_id,
                before_path=pair.before_path,
                after_path=pair.after_path,
                score=pair.score,
                source_decision=pair.source_decision,
                review_decision=pair.review_decision,
                geometry=pair.geometry,
                homography_before_to_after=aligned.homography_before_to_after,
                split_group_id=pair.split_group_id,
                split=pair.split,
            )
        )
    return tuple(accepted), excluded


def build_materialization_plan(
    records: Iterable[Mapping[str, object]],
    destination: Path,
    archive_roots: Iterable[Path],
    *,
    review_answers: Mapping[str, str] | None = None,
    geometry_max_long_side: int = 1200,
    alignment_min_overlap_ratio: float = 0.50,
    alignment_min_edge_correlation: float = 0.15,
    reserve_fraction: float = 0.10,
    minimum_reserve_bytes: int = 1 << 30,
) -> MaterializationPlan:
    roots = tuple(archive_roots)
    resolved = validate_destination(destination, roots)
    if not 0.0 <= reserve_fraction <= 1.0:
        raise DatasetError("reserve_fraction must be between 0 and 1")
    if minimum_reserve_bytes < 0:
        raise DatasetError("minimum_reserve_bytes cannot be negative")
    if geometry_max_long_side <= 0:
        raise DatasetError("geometry_max_long_side must be positive")
    try:
        thresholds = AlignmentThresholds(
            min_overlap_ratio=alignment_min_overlap_ratio,
            min_edge_correlation=alignment_min_edge_correlation,
        )
    except ValueError as exc:
        raise DatasetError(f"Invalid alignment QC threshold: {exc}") from exc
    selected, excluded = select_pairs(records, review_answers=review_answers)
    institution_pairs = _assign_archive_split_units(selected, roots)
    qc_pairs, qc_excluded = _filter_by_full_resolution_alignment(
        institution_pairs,
        geometry_max_long_side=geometry_max_long_side,
        thresholds=thresholds,
    )
    excluded_counts = Counter(excluded)
    excluded_counts.update(qc_excluded)
    split_pairs = assign_group_splits(qc_pairs)
    source_bytes = 0
    for pair in split_pairs:
        for path in (pair.before_path, pair.after_path):
            if not path.is_file():
                raise DatasetError(f"Selected source file is missing: {path}")
            source_bytes += path.stat().st_size
    reserve = max(minimum_reserve_bytes, int(source_bytes * reserve_fraction))
    split_counts = Counter(pair.split for pair in split_pairs)
    return MaterializationPlan(
        destination=resolved,
        pairs=split_pairs,
        source_bytes=source_bytes,
        required_free_bytes=source_bytes + reserve,
        geometry_max_long_side=geometry_max_long_side,
        alignment_min_overlap_ratio=thresholds.min_overlap_ratio,
        alignment_min_edge_correlation=thresholds.min_edge_correlation,
        split_counts={split: split_counts[split] for split in SPLITS},
        excluded_counts=dict(sorted(excluded_counts.items())),
    )


def _copy_and_hash(source: Path, destination: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        input_handle = input_handle  # Narrow BinaryIO for type checkers.
        output_handle = output_handle
        while chunk := input_handle.read(1024 * 1024):
            output_handle.write(chunk)
            digest.update(chunk)
            count += len(chunk)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    return digest.hexdigest(), count


def _write_jsonl(handle: BinaryIO, item: Mapping[str, object]) -> None:
    handle.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))


def _alignment_payload(pair: SelectedPair) -> dict[str, object]:
    matrix = pair.homography_before_to_after
    if matrix is None or len(matrix) != 9:
        raise DatasetError(f"Selected pair {pair.pair_id} has no approved alignment")
    return {
        "reference": "after",
        "homography_before_to_after": [float(value) for value in matrix],
    }


def materialize(plan: MaterializationPlan) -> dict[str, object]:
    """Copy a preflighted plan into a new directory and atomically publish it."""
    if not plan.pairs:
        raise DatasetError("Refusing to publish an empty materialized dataset")
    destination = plan.destination
    if destination.exists():
        raise DatasetError(f"Dataset destination already exists: {destination}")
    ancestor = _existing_ancestor(destination)
    free_bytes = shutil.disk_usage(ancestor).free
    if free_bytes < plan.required_free_bytes:
        raise DatasetError(
            f"Insufficient free space: need {plan.required_free_bytes} bytes, have {free_bytes}"
        )

    staging = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex[:12]}")
    if staging.exists():
        raise DatasetError(f"Temporary dataset path already exists: {staging}")
    copied_bytes = 0
    try:
        pairs_dir = staging / "pairs"
        manifests_dir = staging / "manifests"
        reports_dir = staging / "reports"
        pairs_dir.mkdir(parents=True)
        manifests_dir.mkdir()
        reports_dir.mkdir()
        public_manifest = (manifests_dir / "pairs.jsonl").open("xb")
        provenance = (manifests_dir / "provenance.jsonl").open("xb")
        try:
            for pair in plan.pairs:
                pair_dir = pairs_dir / pair.pair_id
                pair_dir.mkdir()
                before_name = f"before{pair.before_path.suffix.casefold()}"
                after_name = f"after{pair.after_path.suffix.casefold()}"
                before_target = pair_dir / before_name
                after_target = pair_dir / after_name
                before_hash, before_bytes = _copy_and_hash(pair.before_path, before_target)
                after_hash, after_bytes = _copy_and_hash(pair.after_path, after_target)
                copied_bytes += before_bytes + after_bytes
                alignment = _alignment_payload(pair)
                manifest_record: dict[str, object] = {
                    "schema_version": 1,
                    "pair_id": pair.pair_id,
                    "group_id": pair.group_id,
                    "split_group_id": pair.split_group_id,
                    "split": pair.split,
                    "before": str(Path("pairs") / pair.pair_id / before_name),
                    "after": str(Path("pairs") / pair.pair_id / after_name),
                    "source_decision": pair.source_decision,
                    "review_decision": pair.review_decision,
                    "score": pair.score,
                }
                manifest_record["alignment"] = alignment
                _write_jsonl(
                    public_manifest,
                    manifest_record,
                )
                _write_jsonl(
                    provenance,
                    {
                        "schema_version": 1,
                        "pair_id": pair.pair_id,
                        "before_source": str(pair.before_path.resolve()),
                        "after_source": str(pair.after_path.resolve()),
                        "before_sha256": before_hash,
                        "after_sha256": after_hash,
                        "before_bytes": before_bytes,
                        "after_bytes": after_bytes,
                    },
                )
        finally:
            public_manifest.close()
            provenance.close()

        summary = {
            **plan.to_dict(),
            "copied_bytes": copied_bytes,
            "status": "complete",
        }
        (reports_dir / "materialization_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(destination)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def split_summary(pairs: Iterable[SelectedPair]) -> dict[str, dict[str, int]]:
    """Return compact split/group counts without exposing private paths."""
    pair_counts: Counter[str] = Counter()
    group_sets: defaultdict[str, set[str]] = defaultdict(set)
    split_unit_sets: defaultdict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        pair_counts[pair.split] += 1
        group_sets[pair.split].add(pair.group_id)
        split_unit_sets[pair.split].add(pair.split_group_id or pair.group_id)
    return {
        split: {
            "pairs": pair_counts[split],
            "groups": len(group_sets[split]),
            "split_units": len(split_unit_sets[split]),
        }
        for split in SPLITS
    }
