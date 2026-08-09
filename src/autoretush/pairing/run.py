"""Crash-safe orchestration for long, group-oriented pairing runs.

The run fingerprint deliberately excludes group payloads and output locations.  Callers
must provide pseudonymous ``group_id`` values, an identity for the exact algorithm
implementation/runtime, and an ordered content snapshot for every group.  Shards and
the final manifest are private runtime artifacts and must be written outside the public
repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path, PurePath
from typing import Protocol, TypeVar, cast

_SCHEMA_VERSION = 2
_METADATA_NAME = "run.json"
_TEMP_PREFIX = ".autoretush-tmp-"
_FINAL_TEMP_PREFIX = ".autoretush-final-"
_ALGORITHM_ID_PATTERN = re.compile(r"^autoretush-pairing-v[1-9][0-9]*:[0-9a-f]{64}$")
_SNAPSHOT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

GroupT = TypeVar("GroupT")


class GroupDescriptor(Protocol):
    """Minimum descriptor accepted by :func:`run_pairing`."""

    group_id: str


class PairingRunError(RuntimeError):
    """Base class for resumable pairing run integrity errors."""


class StagingMismatchError(PairingRunError):
    """The staging directory belongs to a different run specification."""


class CorruptStagingError(PairingRunError):
    """A staging file is incomplete, malformed, or has failed its integrity check."""


class DuplicateRecordError(PairingRunError):
    """The same canonical JSON record occurred in more than one output position."""


@dataclass(frozen=True)
class PairingRunResult:
    """Summary returned after the final JSONL has been installed."""

    output_path: Path
    staging_path: Path
    fingerprint: str
    group_count: int
    processed_group_count: int
    resumed_group_count: int
    record_count: int
    staging_retained: bool


def _normalise_fingerprint_value(value: object) -> object:
    """Convert deterministic configuration values to a JSON-compatible form."""

    if isinstance(value, PurePath):
        raise ValueError("Run fingerprint parameters must not contain filesystem paths")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Run fingerprint parameters must contain finite numbers")
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalise_fingerprint_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Run fingerprint mapping keys must be strings")
            result[key] = _normalise_fingerprint_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalise_fingerprint_value(item) for item in value]
    raise TypeError(f"Unsupported run fingerprint value type: {type(value).__name__}")


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Pairing run data must be JSON serializable") from exc
    return text.encode("utf-8")


def build_run_fingerprint(
    group_ids: Sequence[str],
    pairing_config: object,
    selection_config: object,
    *,
    algorithm_id: str,
    input_snapshots: Sequence[str],
) -> str:
    """Hash reproducibility inputs without incorporating source or output paths.

    Group descriptors, raw source paths, the callback, staging path, and final output
    path are intentionally excluded.  ``Path`` values in either configuration are
    rejected to prevent accidentally incorporating private source locations.
    """

    ids = _validate_group_ids(group_ids)
    validated_algorithm_id = _validate_algorithm_id(algorithm_id)
    snapshots = _validate_input_snapshots(input_snapshots, expected_count=len(ids))
    payload = {
        "algorithm_id": validated_algorithm_id,
        "groups": [
            {"group_id": group_id, "input_snapshot": snapshot}
            for group_id, snapshot in zip(ids, snapshots, strict=True)
        ],
        "pairing": _normalise_fingerprint_value(pairing_config),
        "schema": _SCHEMA_VERSION,
        "selection": _normalise_fingerprint_value(selection_config),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _validate_algorithm_id(algorithm_id: str) -> str:
    if not isinstance(algorithm_id, str) or not _ALGORITHM_ID_PATTERN.fullmatch(algorithm_id):
        raise ValueError(
            "algorithm_id must be a versioned autoretush pairing ID with a SHA-256 digest"
        )
    return algorithm_id


def _validate_input_snapshots(
    input_snapshots: Sequence[str],
    *,
    expected_count: int,
) -> list[str]:
    if isinstance(input_snapshots, (str, bytes)):
        raise TypeError("input_snapshots must be an ordered sequence of digests")
    snapshots = list(input_snapshots)
    if len(snapshots) != expected_count:
        raise ValueError("input_snapshots must contain exactly one digest per pairing group")
    if any(
        not isinstance(snapshot, str) or not _SNAPSHOT_PATTERN.fullmatch(snapshot)
        for snapshot in snapshots
    ):
        raise ValueError("Every input snapshot must use the sha256:<64 lowercase hex> format")
    return snapshots


def _validate_group_ids(group_ids: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group_id in group_ids:
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("Every pairing group must have a non-empty string group_id")
        if any(ord(character) < 32 for character in group_id):
            raise ValueError("Pairing group_id values must not contain control characters")
        if group_id in seen:
            raise ValueError(f"Duplicate pairing group_id: {group_id}")
        seen.add(group_id)
        result.append(group_id)
    return result


def _default_group_id(group: object) -> str:
    try:
        return cast(GroupDescriptor, group).group_id
    except AttributeError as exc:
        raise TypeError("Pairing group descriptors must expose group_id") from exc


def _staging_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.pairing-staging")


def _shard_name(index: int, group_id: str) -> str:
    group_digest = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return f"{index:08d}-{group_digest}.jsonl"


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_directory(path: Path) -> None:
    """Persist directory entries where the host supports directory fsync."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, chunks: Iterable[bytes]) -> None:
    with path.open("wb") as handle:
        for chunk in chunks:
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())


def _new_temp_path(directory: Path, prefix: str = _TEMP_PREFIX) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    os.close(descriptor)
    return Path(name)


def _atomic_write(path: Path, chunks: Iterable[bytes]) -> None:
    temporary = _new_temp_path(path.parent)
    try:
        _write_fsynced(temporary, chunks)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if _path_exists(temporary):
            temporary.unlink()


def _metadata(fingerprint: str, group_count: int) -> dict[str, object]:
    return {
        "fingerprint": fingerprint,
        "group_count": group_count,
        "kind": "autoretush_pairing_run",
        "schema": _SCHEMA_VERSION,
    }


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptStagingError(f"Invalid staging metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise CorruptStagingError(f"Staging metadata is not an object: {path.name}")
    return value


def _remove_abandoned_temps(staging_path: Path) -> None:
    for entry in staging_path.iterdir():
        if not entry.name.startswith((_TEMP_PREFIX, _FINAL_TEMP_PREFIX)):
            continue
        if entry.is_symlink() or not entry.is_file():
            raise CorruptStagingError("Unsafe temporary entry in pairing staging")
        entry.unlink()
    _fsync_directory(staging_path)


def _prepare_staging(
    staging_path: Path,
    *,
    fingerprint: str,
    group_count: int,
    expected_shards: set[str],
    resume: bool,
) -> None:
    existed = _path_exists(staging_path)
    if existed:
        if staging_path.is_symlink() or not staging_path.is_dir():
            raise CorruptStagingError("Pairing staging path is not a real directory")
        if not resume:
            raise PairingRunError("Pairing staging already exists and resume is disabled")
    else:
        staging_path.mkdir()
        _fsync_directory(staging_path.parent)

    _remove_abandoned_temps(staging_path)
    metadata_path = staging_path / _METADATA_NAME
    entries = list(staging_path.iterdir())
    if not _path_exists(metadata_path):
        if entries:
            raise CorruptStagingError("Pairing staging has files but no run metadata")
        data = _canonical_json(_metadata(fingerprint, group_count)) + b"\n"
        _atomic_write(metadata_path, [data])
    else:
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise CorruptStagingError("Pairing run metadata is not a regular file")
        actual = _load_json_object(metadata_path)
        expected = _metadata(fingerprint, group_count)
        if actual != expected:
            raise StagingMismatchError("Pairing staging fingerprint or run shape does not match")

    allowed = expected_shards | {_METADATA_NAME}
    for entry in staging_path.iterdir():
        if entry.name not in allowed:
            raise CorruptStagingError(f"Unexpected pairing staging entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file():
            raise CorruptStagingError(f"Unsafe pairing staging entry: {entry.name}")


def _record_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        record = dict(value)
    else:
        converter = getattr(value, "to_dict", None)
        if not callable(converter):
            raise TypeError("Pairing callbacks must yield mappings or objects with to_dict()")
        converted = converter()
        if not isinstance(converted, Mapping):
            raise TypeError("Pairing record to_dict() must return a mapping")
        record = dict(converted)
    if not all(isinstance(key, str) for key in record):
        raise TypeError("Pairing record keys must be strings")
    # Round-trip once so shards never contain values the final JSONL cannot reproduce.
    try:
        decoded = json.loads(_canonical_json(record).decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypeError("Pairing callback returned an invalid JSON record") from exc
    if not isinstance(decoded, dict):
        raise TypeError("Pairing callback records must be JSON objects")
    return cast(dict[str, object], decoded)


def _write_shard(
    path: Path,
    *,
    fingerprint: str,
    group_id: str,
    group_index: int,
    records: Iterable[object],
) -> int:
    temporary = _new_temp_path(path.parent)
    count = 0
    digest = hashlib.sha256()
    header = {
        "fingerprint": fingerprint,
        "group_id": group_id,
        "group_index": group_index,
        "kind": "header",
        "schema": _SCHEMA_VERSION,
    }
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_json(header) + b"\n")
            for raw_record in records:
                wrapper = {"kind": "record", "record": _record_dict(raw_record)}
                line = _canonical_json(wrapper) + b"\n"
                handle.write(line)
                digest.update(line)
                count += 1
            footer = {
                "kind": "footer",
                "record_count": count,
                "records_sha256": digest.hexdigest(),
            }
            handle.write(_canonical_json(footer) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if _path_exists(temporary):
            temporary.unlink()
    return count


def _decode_shard_line(line: bytes, name: str) -> dict[str, object]:
    if not line.endswith(b"\n"):
        raise CorruptStagingError(f"Truncated pairing shard: {name}")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptStagingError(f"Invalid JSON in pairing shard: {name}") from exc
    if not isinstance(value, dict):
        raise CorruptStagingError(f"Non-object line in pairing shard: {name}")
    return value


def _load_shard(
    path: Path,
    *,
    fingerprint: str,
    group_id: str,
    group_index: int,
) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise CorruptStagingError(f"Pairing shard is not a regular file: {path.name}")
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as exc:
        raise CorruptStagingError(f"Cannot read pairing shard: {path.name}") from exc
    if len(lines) < 2:
        raise CorruptStagingError(f"Incomplete pairing shard: {path.name}")

    expected_header = {
        "fingerprint": fingerprint,
        "group_id": group_id,
        "group_index": group_index,
        "kind": "header",
        "schema": _SCHEMA_VERSION,
    }
    if _decode_shard_line(lines[0], path.name) != expected_header:
        raise CorruptStagingError(f"Mismatched pairing shard header: {path.name}")

    records: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for line in lines[1:-1]:
        wrapper = _decode_shard_line(line, path.name)
        if set(wrapper) != {"kind", "record"} or wrapper.get("kind") != "record":
            raise CorruptStagingError(f"Invalid record envelope in pairing shard: {path.name}")
        record = wrapper.get("record")
        if not isinstance(record, dict):
            raise CorruptStagingError(f"Non-object record in pairing shard: {path.name}")
        digest.update(line)
        records.append(cast(dict[str, object], record))

    footer = _decode_shard_line(lines[-1], path.name)
    expected_footer = {
        "kind": "footer",
        "record_count": len(records),
        "records_sha256": digest.hexdigest(),
    }
    if footer != expected_footer:
        raise CorruptStagingError(f"Pairing shard integrity check failed: {path.name}")
    return records


def _install_final_no_replace(temporary: Path, output_path: Path) -> None:
    """Atomically publish a complete file without replacing an existing destination."""

    try:
        os.link(temporary, output_path)
    except FileExistsError:
        raise
    except OSError as exc:
        raise PairingRunError(
            "The destination filesystem does not support atomic no-replace publication"
        ) from exc
    _fsync_directory(output_path.parent)


def _assemble_final(
    output_path: Path,
    staging_path: Path,
    ordered_shards: Sequence[tuple[Path, str, int]],
    fingerprint: str,
) -> int:
    temporary = _new_temp_path(staging_path, f"{_FINAL_TEMP_PREFIX}{fingerprint[:12]}-")
    record_count = 0
    seen: set[bytes] = set()
    try:
        with temporary.open("wb") as handle:
            for shard_path, group_id, group_index in ordered_shards:
                records = _load_shard(
                    shard_path,
                    fingerprint=fingerprint,
                    group_id=group_id,
                    group_index=group_index,
                )
                for record in records:
                    line = _canonical_json(record) + b"\n"
                    if line in seen:
                        raise DuplicateRecordError("Duplicate record found while assembling JSONL")
                    seen.add(line)
                    handle.write(line)
                    record_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        _install_final_no_replace(temporary, output_path)
    finally:
        if _path_exists(temporary):
            temporary.unlink()
    return record_count


def _clean_staging(
    staging_path: Path,
    output_path: Path,
    *,
    expected_entries: set[str],
) -> None:
    if not output_path.is_file():
        raise PairingRunError("Refusing to clean staging before the final JSONL exists")
    if staging_path.parent.resolve() != output_path.parent.resolve():
        raise PairingRunError("Refusing to clean a non-sibling staging directory")
    if staging_path != _staging_path(output_path):
        raise PairingRunError("Refusing to clean an unexpected staging directory")
    if staging_path.is_symlink() or not staging_path.is_dir():
        raise CorruptStagingError("Refusing to clean an unsafe staging path")

    for entry in staging_path.iterdir():
        if entry.name not in expected_entries:
            raise CorruptStagingError("Refusing to clean staging with unexpected entries")
        if entry.is_symlink() or not entry.is_file():
            raise CorruptStagingError("Refusing to clean staging with unsafe entries")
    for entry in staging_path.iterdir():
        entry.unlink()
    _fsync_directory(staging_path)
    staging_path.rmdir()
    _fsync_directory(staging_path.parent)


def run_pairing(
    output_path: Path,
    groups: Sequence[GroupT],
    process_group: Callable[[GroupT], Iterable[object]],
    *,
    pairing_config: object,
    selection_config: object,
    algorithm_id: str,
    input_snapshots: Sequence[str],
    group_id_getter: Callable[[GroupT], str] | None = None,
    resume: bool = True,
    retain_staging: bool = True,
) -> PairingRunResult:
    """Run or resume group pairing and atomically create one private JSONL manifest.

    A validated shard is never processed twice.  An empty callback result still creates
    a completed shard, so zero-candidate groups are also skipped after a restart.  The
    final destination is never overwritten.  Cross-group duplicate records are treated
    as an integrity error instead of being silently discarded.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(output_path):
        raise FileExistsError(f"Refusing to overwrite final pairing JSONL: {output_path}")

    ordered_groups = list(groups)
    getter = group_id_getter or _default_group_id
    group_ids = _validate_group_ids([getter(group) for group in ordered_groups])
    fingerprint = build_run_fingerprint(
        group_ids,
        pairing_config,
        selection_config,
        algorithm_id=algorithm_id,
        input_snapshots=input_snapshots,
    )
    staging_path = _staging_path(output_path)
    shard_names = [_shard_name(index, group_id) for index, group_id in enumerate(group_ids)]
    _prepare_staging(
        staging_path,
        fingerprint=fingerprint,
        group_count=len(ordered_groups),
        expected_shards=set(shard_names),
        resume=resume,
    )

    processed = 0
    resumed = 0
    ordered_shards: list[tuple[Path, str, int]] = []
    for index, (group, group_id, shard_name) in enumerate(
        zip(ordered_groups, group_ids, shard_names, strict=True)
    ):
        shard_path = staging_path / shard_name
        ordered_shards.append((shard_path, group_id, index))
        if _path_exists(shard_path):
            _load_shard(
                shard_path,
                fingerprint=fingerprint,
                group_id=group_id,
                group_index=index,
            )
            resumed += 1
            continue
        _write_shard(
            shard_path,
            fingerprint=fingerprint,
            group_id=group_id,
            group_index=index,
            records=process_group(group),
        )
        processed += 1

    record_count = _assemble_final(output_path, staging_path, ordered_shards, fingerprint)
    if not retain_staging:
        _clean_staging(
            staging_path,
            output_path,
            expected_entries={_METADATA_NAME, *shard_names},
        )
    return PairingRunResult(
        output_path=output_path,
        staging_path=staging_path,
        fingerprint=fingerprint,
        group_count=len(ordered_groups),
        processed_group_count=processed,
        resumed_group_count=resumed,
        record_count=record_count,
        staging_retained=retain_staging,
    )
