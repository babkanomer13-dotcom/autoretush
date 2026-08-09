from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class ImageReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameFingerprint:
    width: int
    height: int
    phash: int

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


def read_bgr(path: Path) -> np.ndarray:
    """Read non-ASCII Windows paths through imdecode."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"Cannot read image bytes: {path}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageReadError(f"OpenCV cannot decode image: {path}")
    return image


def resize_long_side(image: np.ndarray, max_long_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_long_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(small)[:8, :8]
    values = coefficients.flatten()
    threshold = float(np.median(values[1:]))
    bits = values > threshold
    bits[0] = False
    result = 0
    for index, enabled in enumerate(bits.tolist()):
        if enabled:
            result |= 1 << index
    return result


def fingerprint(path: Path, *, max_long_side: int = 512) -> FrameFingerprint:
    image = read_bgr(path)
    height, width = image.shape[:2]
    return FrameFingerprint(
        width=width,
        height=height,
        phash=perceptual_hash(resize_long_side(image, max_long_side)),
    )


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()
