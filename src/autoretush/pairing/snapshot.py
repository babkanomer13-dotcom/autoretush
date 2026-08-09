"""Content snapshots and algorithm identities for reproducible pairing resumes."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import re
import stat
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import BinaryIO, Protocol

import cv2
import numpy as np
import scipy

from autoretush.inventory import ImageRecord, PairGroup

_SNAPSHOT_SCHEMA = "autoretush-pair-group-snapshot-v1"
_ALGORITHM_SCHEMA = "autoretush-pairing-v1"
_ALGORITHM_COMPONENTS = ("fingerprint", "geometry", "matcher")
_RUNTIME_COMPONENTS = ("python", "numpy", "opencv", "scipy")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")
_CHUNK_SIZE = 1024 * 1024


class InputSnapshotError(RuntimeError):
    """A PairGroup input cannot be snapshotted without a race or unsafe path."""


class _Digest(Protocol):
    def update(self, data: bytes) -> object: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_pairing_algorithm_id(
    source_blobs: Mapping[str, bytes],
    runtime_versions: Mapping[str, str],
) -> str:
    """Build a path-free, versioned identity from code bytes and runtime versions."""

    if set(source_blobs) != set(_ALGORITHM_COMPONENTS):
        raise ValueError("Algorithm source components are incomplete or unexpected")
    if set(runtime_versions) != set(_RUNTIME_COMPONENTS):
        raise ValueError("Algorithm runtime components are incomplete or unexpected")

    source_hashes: dict[str, str] = {}
    for name in _ALGORITHM_COMPONENTS:
        blob = source_blobs[name]
        if not isinstance(blob, bytes) or not blob:
            raise TypeError("Algorithm source components must be non-empty bytes")
        source_hashes[name] = hashlib.sha256(blob).hexdigest()

    versions: dict[str, str] = {}
    for name in _RUNTIME_COMPONENTS:
        version = runtime_versions[name]
        if not isinstance(version, str) or not _VERSION_PATTERN.fullmatch(version):
            raise ValueError("Algorithm runtime versions must be path-free version strings")
        versions[name] = version

    payload = {
        "runtime_versions": versions,
        "schema": _ALGORITHM_SCHEMA,
        "source_sha256": source_hashes,
    }
    return f"{_ALGORITHM_SCHEMA}:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


def _module_source(module: ModuleType) -> bytes:
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError) as exc:
        raise RuntimeError("Pairing algorithm source is unavailable") from exc
    # Source loaders normalize newlines inconsistently across platforms.  Line ending
    # differences do not change the algorithm and therefore do not change its ID.
    return source.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def current_pairing_algorithm_id() -> str:
    """Return the identity of the currently imported whole-frame pairing algorithm."""

    sources = {
        name: _module_source(import_module(f"autoretush.pairing.{name}"))
        for name in _ALGORITHM_COMPONENTS
    }
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
    }
    return build_pairing_algorithm_id(sources, versions)


def _is_link_or_reparse(path: Path, metadata: os.stat_result | None = None) -> bool:
    try:
        info = metadata if metadata is not None else path.lstat()
    except OSError as exc:
        raise InputSnapshotError("Cannot inspect a pairing input path") from exc
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode)


def _state(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        *_identity(metadata),
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _stream_content(handle: BinaryIO, digest: _Digest, chunk_size: int) -> int:
    total = 0
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            return total
        digest.update(chunk)
        total += len(chunk)


def _relative_source(
    root_absolute: Path,
    root_resolved: Path,
    record: ImageRecord,
    *,
    role: str,
    index: int,
) -> tuple[Path, str]:
    label = f"{role}[{index}]"
    candidate_absolute = Path(os.path.abspath(record.path))
    try:
        relative = candidate_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise InputSnapshotError(f"{label} is outside the archive root") from exc
    if not relative.parts:
        raise InputSnapshotError(f"{label} does not name a file")

    current = root_absolute
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise InputSnapshotError(f"{label} cannot be inspected") from exc
        if _is_link_or_reparse(current, metadata):
            raise InputSnapshotError(f"{label} uses a symlink or reparse point")

    try:
        resolved = candidate_absolute.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise InputSnapshotError(f"{label} resolves outside the archive root") from exc
    return candidate_absolute, relative.as_posix()


def _hash_record(
    group_digest,
    root_absolute: Path,
    root_resolved: Path,
    record: ImageRecord,
    *,
    role: str,
    index: int,
    chunk_size: int,
) -> None:
    label = f"{role}[{index}]"
    path, relative = _relative_source(
        root_absolute,
        root_resolved,
        record,
        role=role,
        index=index,
    )
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise InputSnapshotError(f"{label} cannot be inspected") from exc
    if _is_link_or_reparse(path, before_path) or not stat.S_ISREG(before_path.st_mode):
        raise InputSnapshotError(f"{label} is not a regular source file")
    if before_path.st_size != record.size_bytes:
        raise InputSnapshotError(f"{label} changed after inventory")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InputSnapshotError(f"{label} cannot be opened safely") from exc

    content_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if _identity(opened) != _identity(before_path) or not stat.S_ISREG(opened.st_mode):
                raise InputSnapshotError(f"{label} was replaced while opening")
            byte_count = _stream_content(handle, content_digest, chunk_size)
            after_open = os.fstat(handle.fileno())
    except InputSnapshotError:
        raise
    except OSError as exc:
        raise InputSnapshotError(f"{label} could not be hashed") from exc

    try:
        after_path = path.lstat()
    except OSError as exc:
        raise InputSnapshotError(f"{label} disappeared during hashing") from exc
    final_path, final_relative = _relative_source(
        root_absolute,
        root_resolved,
        record,
        role=role,
        index=index,
    )
    if final_path != path or final_relative != relative:
        raise InputSnapshotError(f"{label} was redirected during hashing")
    if _is_link_or_reparse(path, after_path):
        raise InputSnapshotError(f"{label} became a symlink during hashing")
    if (
        _state(opened) != _state(after_open)
        or _identity(after_path) != _identity(opened)
        or _state(after_path) != _state(after_open)
        or byte_count != after_open.st_size
    ):
        raise InputSnapshotError(f"{label} was replaced or changed during hashing")

    descriptor_data = {
        "content_sha256": content_digest.hexdigest(),
        "index": index,
        "relative": relative,
        "role": role,
        "size": byte_count,
    }
    encoded = _canonical_json(descriptor_data)
    group_digest.update(len(encoded).to_bytes(8, "big"))
    group_digest.update(encoded)


def snapshot_pair_group(group: PairGroup, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Hash all ordered PairGroup inputs without returning or persisting raw paths."""

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    root_absolute = Path(os.path.abspath(group.archive_root))
    try:
        root_metadata = root_absolute.lstat()
        root_resolved = root_absolute.resolve(strict=True)
    except OSError as exc:
        raise InputSnapshotError("Archive root cannot be inspected") from exc
    if _is_link_or_reparse(root_absolute, root_metadata) or not stat.S_ISDIR(root_metadata.st_mode):
        raise InputSnapshotError("Archive root must be a real directory")

    digest = hashlib.sha256()
    digest.update(_SNAPSHOT_SCHEMA.encode("ascii") + b"\0")
    for role, records in (("before", group.before), ("after", group.after)):
        digest.update(role.encode("ascii") + len(records).to_bytes(8, "big"))
        for index, record in enumerate(records):
            _hash_record(
                digest,
                root_absolute,
                root_resolved,
                record,
                role=role,
                index=index,
                chunk_size=chunk_size,
            )

    try:
        root_after = root_absolute.lstat()
    except OSError as exc:
        raise InputSnapshotError("Archive root disappeared during hashing") from exc
    if _state(root_metadata) != _state(root_after):
        raise InputSnapshotError("Archive root changed during hashing")
    return f"sha256:{digest.hexdigest()}"
