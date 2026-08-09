from pathlib import Path

import cv2
import numpy as np

from autoretush.config import PairingConfig
from autoretush.inventory import ImageRecord, PairGroup
from autoretush.pairing.geometry import compare_geometry
from autoretush.pairing.matcher import match_group


def _scene(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 45, (720, 960, 3), dtype=np.uint8)
    for index in range(28):
        color = tuple(int(value) for value in rng.integers(55, 245, 3))
        center = tuple(int(value) for value in rng.integers([40, 40], [920, 680]))
        radius = int(rng.integers(8, 55))
        cv2.circle(image, center, radius, color, -1)
        cv2.putText(
            image,
            f"T{index}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    assert success
    encoded.tofile(path)


def _record(path: Path) -> ImageRecord:
    return ImageRecord(path=path, size_bytes=path.stat().st_size, extension=path.suffix)


def test_same_frame_survives_tone_and_local_retouch(tmp_path: Path) -> None:
    before = tmp_path / "source.jpg"
    after = tmp_path / "pp" / "result.jpg"
    source = _scene()
    retouched = cv2.convertScaleAbs(source, alpha=1.08, beta=12)
    local = retouched[240:500, 330:630]
    retouched[240:500, 330:630] = cv2.addWeighted(
        local, 0.78, cv2.GaussianBlur(local, (0, 0), 2.0), 0.22, 0
    )
    _write(before, source)
    _write(after, retouched)

    metrics = compare_geometry(before, after, max_long_side=900)

    assert metrics.transform_valid
    assert metrics.inliers >= 80
    assert metrics.inlier_ratio >= 0.8
    assert metrics.score >= 0.75


def test_group_assignment_matches_two_retouched_frames(tmp_path: Path) -> None:
    source_dir = tmp_path / "дети на разворот"
    processed_dir = source_dir / "pp"
    before_a = source_dir / "a.jpg"
    before_b = source_dir / "b.jpg"
    after_a = processed_dir / "unrelated-name-1.jpg"
    after_b = processed_dir / "unrelated-name-2.jpg"
    scene_a = _scene(11)
    scene_b = _scene(29)
    _write(before_a, scene_a)
    _write(before_b, scene_b)
    _write(after_a, cv2.convertScaleAbs(scene_b, alpha=0.96, beta=9))
    _write(after_b, cv2.convertScaleAbs(scene_a, alpha=1.04, beta=5))
    group = PairGroup(
        group_id="g_test",
        archive_root=tmp_path,
        source_dir=source_dir,
        processed_dir=processed_dir,
        before=(_record(before_a), _record(before_b)),
        after=(_record(after_a), _record(after_b)),
    )
    config = PairingConfig(
        shortlist_size=2,
        max_long_side=900,
        min_keypoint_inliers=25,
        min_inlier_ratio=0.55,
        min_source_coverage=0.05,
        max_phash_distance=30,
        auto_accept_score=0.72,
        review_score=0.55,
    )

    results = match_group(group, config)
    mapping = {item.before_path.name: item.after_path.name for item in results}

    assert mapping == {"a.jpg": "unrelated-name-2.jpg", "b.jpg": "unrelated-name-1.jpg"}
    assert {item.decision for item in results} == {"accepted"}
