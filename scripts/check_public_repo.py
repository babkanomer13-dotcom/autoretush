from __future__ import annotations

import argparse
import codecs
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FORBIDDEN_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
    ".psd",
    ".psb",
    ".raw",
    ".dng",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".orf",
    ".rw2",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".engine",
    ".trt",
    ".tflite",
    ".task",
    ".pb",
    ".npy",
    ".npz",
    ".p12",
    ".pem",
    ".pfx",
    ".pkl",
    ".pickle",
    ".key",
    ".jks",
    ".kdbx",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".jsonl",
    ".csv",
    ".tsv",
    ".parquet",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tgz",
}
FORBIDDEN_PARTS = {
    "artifacts",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "images",
    "inventory",
    "local",
    "manifests",
    "mlruns",
    "models",
    "outputs",
    "photos",
    "private",
    "provenance",
    "reviews",
    "runs",
    "secrets",
    "wandb",
    "weights",
}
FORBIDDEN_FILENAMES = {
    ".env",
    ".htpasswd",
    ".netrc",
    ".pypirc",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
FORBIDDEN_PRIVATE_FILENAMES = {"test_global_lut.py"}
MAX_FILE_BYTES = 1_500_000
ALLOWED_TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_TEXT_FILENAMES = {".gitattributes", ".gitignore", "license", "pre-commit"}

_BINARY_SIGNATURES = (
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG\r\n\x1a\n",
    b"GIF87a",
    b"GIF89a",
    b"II*\x00",  # little-endian TIFF
    b"MM\x00*",  # big-endian TIFF
    b"8BPS",  # Photoshop
    b"PK\x03\x04",  # ZIP and zip-based checkpoints
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07",
    b"SQLite format 3\x00",
    b"\x93NUMPY",
    b"%PDF-",
    b"\x1f\x8b",  # gzip
)

_SIMPLE_CONTENT_PATTERNS = (
    (
        "private key material",
        re.compile(
            r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
    ),
    ("GitHub access token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    (
        "OpenAI-style API key",
        re.compile(r"\bsk-(?:(?:proj|svcacct|ant)-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("Slack access token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    (
        "JSON web token",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    (
        "authorization header with inline credentials",
        re.compile(r"(?im)^\s*authorization\s*:\s*(?:basic|bearer)\s+[A-Za-z0-9+/_.=-]{12,}"),
    ),
    (
        "URL containing inline credentials",
        re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@", re.IGNORECASE),
    ),
    (
        "Windows user-home absolute path",
        re.compile(
            r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]"
            r"[^\\/\s\"'<>]+[\\/]",
            re.IGNORECASE,
        ),
    ),
    (
        "Unix user-home absolute path",
        re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users)/[^/\s\"'<>]+/"),
    ),
    (
        "WSL-mounted absolute path",
        re.compile(r"(?<![A-Za-z0-9_])/mnt/[A-Za-z]/"),
    ),
    (
        "UNC network path",
        re.compile(r"(?<![\\])\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+"),
    ),
)

_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?P<path>[A-Za-z]:[\\/](?![\\/])[^\s\"'<>]*)"
)
_RASTER_DATA_URI = re.compile(
    r"data:image/(?:avif|bmp|gif|heic|heif|jpeg|jpg|png|tiff|webp);base64,",
    re.IGNORECASE,
)
_UNIX_ABSOLUTE_PATH = re.compile(r"(?<![:A-Za-z0-9_.-])(?P<path>/(?:[^/\s\"'<>]+/)+[^\s\"'<>]*)")
_REAL_ARCHIVE_NAMES = re.compile(
    r"(?:\u0410\u043b\u044c\u0431\u043e\u043c\u044b\s+"
    r"\u043f\u0440\u043e\u0448\u043b\u044b\u0435\s+"
    r"\u0433\u043e\u0434\u0430|"
    r"\u0421\u0435\u0437\u043e\u043d\s+20\d{2}\s*[-\u2013\u2014]\s*20\d{2})",
    re.IGNORECASE,
)
_CONFIG_LIKE_SUFFIXES = {".cfg", ".conf", ".ini", ".json", ".toml", ".yaml", ".yml"}


def _example_path(drive: str, suffix: str) -> str:
    # Keep absolute path literals out of this scanner's own source blob.
    return f"{drive}:/{suffix}".casefold()


_ALLOWED_WINDOWS_EXAMPLES_BY_FILE = {
    "configs/local.example.yaml": frozenset(
        {
            _example_path("x", "private-archive/season-a"),
            _example_path("x", "autoretush-local"),
            _example_path("x", "autoretush-local/dataset"),
        }
    ),
    "docs/dataset.md": frozenset(
        {
            _example_path("x", "autoretush-local/manifests/candidates.jsonl"),
            _example_path("x", "autoretush-local/reviews/review-results.json"),
        }
    ),
    "tests/test_config.py": frozenset(
        {
            _example_path("e", "archive"),
            _example_path("e", "local"),
            _example_path("e", "local/dataset"),
        }
    ),
}
_LEGACY_HISTORY_FINDING_EXCEPTIONS = {
    (
        "2955c750545b86a65b4e1204e1fe64305ddf4b9b",
        "docs/DATASET.md",
        "non-placeholder Windows absolute path",
    ),
    (
        "ca20ff5341658ac936c9803a157b7a284515959a",
        "docs/PROJECT_MAP.md",
        "non-placeholder Windows absolute path",
    ),
}
_LEGACY_PERSONAL_EMAIL_COMMITS = {"ff493ed7c10a100a98ddc3f866652775278d9c7c"}
_EMAIL_HEADER = re.compile(rb"^(?:author|committer|tagger) .* <([^<>\r\n]+)>", re.MULTILINE)

_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)\b(?:password|passwd|pwd|client_secret|api[_-]?key|access[_-]?token|"
    r"auth[_-]?token)\s*[:=]\s*[\"']?(?P<value>[^\s\"'#;,}{]{8,})"
)
_PLACEHOLDER_MARKERS = {
    "changeme",
    "dummy",
    "example",
    "not-a-secret",
    "placeholder",
    "redacted",
    "replace-me",
    "sample",
    "your-key",
    "your-password",
    "your-secret",
    "your-token",
}
_PLACEHOLDER_PREFIXES = ("${", "$env:", "{{", "<", "env[", "os.environ", "secrets.")


@dataclass(frozen=True, order=True)
class Finding:
    scope: str
    path: str
    reason: str

    def render(self) -> str:
        # Deliberately exclude matching content, offsets, object IDs and secret fragments.
        return f"{self.scope}: {self.reason}: {self.path}"


@dataclass(frozen=True)
class IndexEntry:
    mode: str
    object_id: str
    path: str


@dataclass(frozen=True)
class BlobSnapshot:
    size: int
    data: bytes | None


class RepositoryScanError(RuntimeError):
    """Raised without including potentially sensitive command output."""


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            input=input_bytes,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RepositoryScanError("Git repository inspection failed.") from exc
    return result.stdout


def _normalise_git_path(path: str) -> str:
    return path.replace("\\", "/")


def path_reasons(path: str, size: int | None = None) -> set[str]:
    relative = PurePosixPath(_normalise_git_path(path))
    directory_parts = {part.casefold() for part in relative.parts[:-1]}
    suffixes = {suffix.casefold() for suffix in relative.suffixes}
    name = relative.name.casefold()
    reasons: set[str] = set()

    if directory_parts & FORBIDDEN_PARTS:
        reasons.add("forbidden private/data directory")
    if suffixes & FORBIDDEN_SUFFIXES:
        reasons.add("forbidden binary/data file type")
    if name in FORBIDDEN_FILENAMES or (name.startswith(".env.") and name != ".env.example"):
        reasons.add("forbidden credential filename")
    if name in FORBIDDEN_PRIVATE_FILENAMES:
        reasons.add("forbidden proprietary implementation")
    if "configs" in directory_parts and (
        name in {"local.yaml", "local.yml"}
        or (name.startswith("local.") and name not in {"local.example.yaml", "local.example.yml"})
    ):
        reasons.add("forbidden local configuration")
    if size is not None and size > MAX_FILE_BYTES:
        reasons.add(f"file exceeds public size limit ({MAX_FILE_BYTES} bytes)")
    if (
        relative.suffix.casefold() not in ALLOWED_TEXT_SUFFIXES
        and name not in ALLOWED_TEXT_FILENAMES
    ):
        reasons.add("unsupported public file type")
    return reasons


def _decode_text(data: bytes) -> str | None:
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig", errors="strict")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return None
    if data and data.count(b"\0") / len(data) > 0.05:
        return None
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def _is_placeholder(value: str) -> bool:
    normalised = value.strip().strip("\"'").casefold()
    if not normalised:
        return True
    if normalised.startswith(_PLACEHOLDER_PREFIXES):
        return True
    return normalised in _PLACEHOLDER_MARKERS or normalised.startswith("your_")


def _is_allowed_windows_example(
    path: str,
    source_path: str | None,
) -> bool:
    if source_path is None:
        return False
    normalised = path.replace("\\", "/").rstrip("`.,:;)]}").casefold()
    source = _normalise_git_path(source_path).casefold()
    return normalised in _ALLOWED_WINDOWS_EXAMPLES_BY_FILE.get(source, ())


def content_reasons(
    data: bytes,
    *,
    path: str | None = None,
) -> set[str]:
    reasons: set[str] = set()
    if data.startswith(_BINARY_SIGNATURES) or (
        len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    ):
        reasons.add("binary/media file signature")
    text = _decode_text(data)
    if text is None:
        reasons.add("binary or non-UTF-8 content")
        return reasons

    reasons.update(reason for reason, pattern in _SIMPLE_CONTENT_PATTERNS if pattern.search(text))
    if _REAL_ARCHIVE_NAMES.search(text):
        reasons.add("real archive or season name")
    if _RASTER_DATA_URI.search(text):
        reasons.add("embedded raster image data URI")
    if any(
        not _is_allowed_windows_example(match.group("path"), path)
        for match in _WINDOWS_ABSOLUTE_PATH.finditer(text)
    ):
        reasons.add("non-placeholder Windows absolute path")
    if (
        path is not None
        and PurePosixPath(_normalise_git_path(path)).suffix.casefold() in _CONFIG_LIKE_SUFFIXES
        and _UNIX_ABSOLUTE_PATH.search(text)
    ):
        reasons.add("absolute path in configuration")
    if any(
        not _is_placeholder(match.group("value")) for match in _CREDENTIAL_ASSIGNMENT.finditer(text)
    ):
        reasons.add("inline credential assignment")
    return reasons


def index_entries(root: Path) -> list[IndexEntry]:
    output = _git(root, "ls-files", "--stage", "-z")
    entries: list[IndexEntry] = []
    for record in output.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise RepositoryScanError("Git index inspection failed.")
        fields = metadata.decode("ascii").split()
        if len(fields) != 3:
            raise RepositoryScanError("Git index inspection failed.")
        mode, object_id, stage = fields
        if stage == "0":
            entries.append(IndexEntry(mode, object_id, raw_path.decode("utf-8", errors="replace")))
    return entries


def _blob(root: Path, object_id: str) -> bytes:
    return _git(root, "cat-file", "blob", object_id)


def _object_metadata(root: Path, object_ids: list[str]) -> dict[str, tuple[str, int]]:
    unique_ids = list(dict.fromkeys(object_ids))
    if not unique_ids:
        return {}
    output = _git(
        root,
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        input_bytes=("\n".join(unique_ids) + "\n").encode("ascii"),
    )
    metadata: dict[str, tuple[str, int]] = {}
    for line in output.splitlines():
        fields = line.decode("ascii").split()
        if len(fields) == 2 and fields[1] == "missing":
            metadata[fields[0]] = ("missing", 0)
            continue
        if len(fields) != 3:
            raise RepositoryScanError("Git object inspection failed.")
        object_id, object_type, raw_size = fields
        metadata[object_id] = (object_type, int(raw_size))
    return metadata


def _add_findings(
    findings: set[Finding],
    *,
    scope: str,
    path: str,
    size: int,
    data: bytes | None,
) -> None:
    display_path = _normalise_git_path(path)
    reasons = path_reasons(display_path, size)
    if data is not None:
        reasons.update(content_reasons(data, path=display_path))
    for reason in reasons:
        findings.add(Finding(scope, display_path, reason))


def scan_index(
    root: Path,
) -> tuple[set[Finding], list[IndexEntry], dict[str, BlobSnapshot]]:
    findings: set[Finding] = set()
    entries = index_entries(root)
    snapshots: dict[str, BlobSnapshot] = {}
    metadata = _object_metadata(root, [entry.object_id for entry in entries])
    for entry in entries:
        object_type, size = metadata[entry.object_id]
        data = (
            _blob(root, entry.object_id)
            if object_type == "blob" and size <= MAX_FILE_BYTES
            else None
        )
        snapshots[entry.path] = BlobSnapshot(size=size, data=data)
        _add_findings(findings, scope="index", path=entry.path, size=size, data=data)
    return findings, entries, snapshots


def scan_tracked_worktree(
    root: Path,
    entries: list[IndexEntry],
    index_snapshots: dict[str, BlobSnapshot],
) -> set[Finding]:
    findings: set[Finding] = set()
    resolved_root = root.resolve()
    for entry in entries:
        # A symlink's index blob is scanned above; following it could read outside the repository.
        if entry.mode in {"120000", "160000"}:
            continue
        relative = PurePosixPath(entry.path)
        candidate = root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
            continue
        try:
            size = resolved.stat().st_size
            data = resolved.read_bytes() if size <= MAX_FILE_BYTES else None
        except OSError as exc:
            raise RepositoryScanError("Tracked working-tree inspection failed.") from exc
        index_snapshot = index_snapshots[entry.path]
        if size == index_snapshot.size and data is not None and data == index_snapshot.data:
            continue
        if data is None and index_snapshot.size > MAX_FILE_BYTES:
            # Both copies already have the same publish-blocking size reason.
            continue
        _add_findings(findings, scope="worktree", path=entry.path, size=size, data=data)
    return findings


def _history_entries(root: Path) -> tuple[dict[str, set[str]], set[str]]:
    tree_ids = {
        line.decode("ascii")
        for line in _git(root, "log", "--all", "--format=%T").splitlines()
        if line
    }
    paths_by_blob: dict[str, set[str]] = {}
    all_paths: set[str] = set()
    for tree_id in tree_ids:
        output = _git(root, "ls-tree", "-r", "-z", "--full-tree", tree_id)
        for record in output.split(b"\0"):
            if not record:
                continue
            metadata, separator, raw_path = record.partition(b"\t")
            fields = metadata.decode("ascii").split()
            if not separator or len(fields) != 3:
                raise RepositoryScanError("Git history inspection failed.")
            _, object_type, object_id = fields
            path = raw_path.decode("utf-8", errors="replace")
            all_paths.add(path)
            if object_type == "blob":
                paths_by_blob.setdefault(object_id, set()).add(path)
    return paths_by_blob, all_paths


def scan_history(
    root: Path,
    *,
    exclude_entries: set[tuple[str, str]] | None = None,
) -> set[Finding]:
    findings: set[Finding] = set()
    excluded = exclude_entries or set()
    paths_by_blob, all_paths = _history_entries(root)
    metadata = _object_metadata(root, list(paths_by_blob))

    blob_paths = {path for paths in paths_by_blob.values() for path in paths}
    for path in all_paths - blob_paths:
        display_path = _normalise_git_path(path)
        for reason in path_reasons(display_path):
            findings.add(Finding("history", display_path, reason))

    for object_id, paths in paths_by_blob.items():
        _, size = metadata[object_id]
        data = _blob(root, object_id) if size <= MAX_FILE_BYTES else None
        for path in paths:
            display_path = _normalise_git_path(path)
            if (object_id, display_path) in excluded:
                continue
            reasons = path_reasons(display_path, size)
            if data is not None:
                reasons.update(content_reasons(data, path=display_path))
            for reason in reasons:
                if (object_id, display_path, reason) in _LEGACY_HISTORY_FINDING_EXCEPTIONS:
                    continue
                findings.add(Finding("history", display_path, reason))
    return findings


def _private_email_reason(raw_object: bytes, *, allow_legacy: bool) -> set[str]:
    if allow_legacy:
        return set()
    for raw_email in _EMAIL_HEADER.findall(raw_object):
        try:
            email = raw_email.decode("ascii").casefold()
        except UnicodeDecodeError:
            return {"non-private author or committer email"}
        if not (
            email == "noreply@github.com"
            or email.endswith("@users.noreply.github.com")
            or email.endswith(".invalid")
        ):
            return {"non-private author or committer email"}
    return set()


def _metadata_message(raw_object: bytes) -> bytes:
    _, separator, message = raw_object.partition(b"\n\n")
    return message if separator else b""


def scan_history_metadata(root: Path) -> set[Finding]:
    """Scan reachable commit/tag messages and email headers without logging their values."""
    findings: set[Finding] = set()
    commit_ids = [
        line.decode("ascii") for line in _git(root, "rev-list", "--all").splitlines() if line
    ]
    for commit_id in commit_ids:
        raw = _git(root, "cat-file", "commit", commit_id)
        reasons = _private_email_reason(
            raw,
            allow_legacy=commit_id in _LEGACY_PERSONAL_EMAIL_COMMITS,
        )
        reasons.update(content_reasons(_metadata_message(raw)))
        for reason in reasons:
            findings.add(Finding("history-metadata", "commit", reason))

    tag_lines = _git(root, "for-each-ref", "--format=%(objectname) %(objecttype)", "refs/tags")
    for line in tag_lines.splitlines():
        fields = line.decode("ascii").split()
        if len(fields) != 2:
            raise RepositoryScanError("Git tag inspection failed.")
        object_id, object_type = fields
        if object_type != "tag":
            continue
        raw = _git(root, "cat-file", "tag", object_id)
        reasons = _private_email_reason(raw, allow_legacy=False)
        reasons.update(content_reasons(_metadata_message(raw)))
        for reason in reasons:
            findings.add(Finding("history-metadata", "annotated tag", reason))
    return findings


def scan_repository(root: Path, *, include_history: bool = True) -> list[Finding]:
    root = root.resolve()
    findings, entries, index_snapshots = scan_index(root)
    findings.update(scan_tracked_worktree(root, entries, index_snapshots))
    if include_history:
        current_entries = {(entry.object_id, _normalise_git_path(entry.path)) for entry in entries}
        findings.update(scan_history(root, exclude_entries=current_entries))
        findings.update(scan_history_metadata(root))
    return sorted(findings)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a public repository is safe to publish."
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip old reachable Git blobs (the index and tracked worktree are still checked).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        findings = scan_repository(root, include_history=not args.no_history)
    except RepositoryScanError:
        print("Public repository safety check could not complete.", file=sys.stderr)
        return 2

    if findings:
        print("Public repository safety check failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding.render()}", file=sys.stderr)
        return 1
    print("Public repository safety check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
