from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FORBIDDEN_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".psd",
    ".psb",
    ".raw",
    ".dng",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".engine",
    ".tflite",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
}
FORBIDDEN_PARTS = {
    "data",
    "dataset",
    "datasets",
    "photos",
    "images",
    "manifests",
    "reviews",
    "provenance",
    "weights",
    "checkpoints",
    "outputs",
    "runs",
    "private",
    "local",
}
MAX_FILE_BYTES = 1_500_000


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    problems: list[str] = []
    for path in tracked_files(root):
        relative = path.relative_to(root)
        parts = {part.casefold() for part in relative.parts[:-1]}
        suffixes = {suffix.casefold() for suffix in path.suffixes}
        if parts & FORBIDDEN_PARTS:
            problems.append(f"forbidden directory: {relative}")
        if suffixes & FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden file type: {relative}")
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            problems.append(f"tracked file is missing: {relative}")
            continue
        if size > MAX_FILE_BYTES:
            problems.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative} ({size})")

    if problems:
        print("Public repository safety check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Public repository safety check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
