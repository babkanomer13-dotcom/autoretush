import json
from pathlib import Path

import pytest
from PIL import Image

from autoretush.review import build_review_pack, select_review_sample


def test_review_sample_spreads_across_groups() -> None:
    records = [
        {
            "pair_id": f"p_{index}",
            "group_id": f"g_{index % 10}",
            "decision": "accepted" if index < 80 else "review",
            "score": 0.9,
        }
        for index in range(100)
    ]

    sample = select_review_sample(records, size=50, accepted_fraction=0.8, seed=1)

    assert len(sample) == 50
    assert sum(item["decision"] == "accepted" for item in sample) == 40
    assert len({item["group_id"] for item in sample}) == 10


def test_review_pack_uses_webp_without_exif(tmp_path: Path) -> None:
    before = tmp_path / "before.jpg"
    after = tmp_path / "after.jpg"
    Image.new("RGB", (640, 480), (120, 90, 60)).save(before, exif=b"Exif\x00\x00test")
    Image.new("RGB", (640, 480), (130, 100, 70)).save(after)
    record = {
        "pair_id": "p_0123456789abcdefabcd",
        "group_id": "g_test",
        "decision": "accepted",
        "score": 0.9,
        "before_path": str(before),
        "after_path": str(after),
        "geometry": {"inliers": 100, "inlier_ratio": 0.95},
    }

    private_manifest = tmp_path / "private" / "sample.json"
    index = build_review_pack(
        [record],
        tmp_path / "review",
        image_format="webp",
        private_manifest_path=private_manifest,
    )

    thumbnails = sorted((tmp_path / "review" / "assets").glob("*.webp"))
    assert len(thumbnails) == 2
    assert ".webp" in index.read_text(encoding="utf-8")
    assert all(len(Image.open(path).getexif()) == 0 for path in thumbnails)
    assert not (tmp_path / "review" / "sample_manifest.json").exists()
    assert json.loads(private_manifest.read_text())[0]["before_path"] == str(before)


def test_private_manifest_is_refused_inside_publishable_pack(tmp_path: Path) -> None:
    before = tmp_path / "before.jpg"
    after = tmp_path / "after.jpg"
    Image.new("RGB", (32, 32)).save(before)
    Image.new("RGB", (32, 32)).save(after)
    record = {
        "pair_id": "p_0123456789abcdefabcd",
        "before_path": str(before),
        "after_path": str(after),
    }

    with pytest.raises(ValueError, match="outside"):
        build_review_pack(
            [record],
            tmp_path / "review",
            private_manifest_path=tmp_path / "review" / "private.json",
        )


def test_review_pack_rejects_path_traversal_pair_id_before_writing(tmp_path: Path) -> None:
    record = {
        "pair_id": "../../outside",
        "before_path": str(tmp_path / "before.jpg"),
        "after_path": str(tmp_path / "after.jpg"),
    }

    with pytest.raises(ValueError, match="pair_id"):
        build_review_pack([record], tmp_path / "review")

    assert not (tmp_path / "review").exists()
    assert not (tmp_path / "outside_before.webp").exists()


def test_review_pack_refuses_non_empty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "review"
    output.mkdir()
    stale = output / "sample_manifest.json"
    stale.write_text("private", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        build_review_pack([], output)

    assert stale.read_text(encoding="utf-8") == "private"
