from pathlib import Path

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

    index = build_review_pack([record], tmp_path / "review", image_format="webp")

    thumbnails = sorted((tmp_path / "review" / "assets").glob("*.webp"))
    assert len(thumbnails) == 2
    assert ".webp" in index.read_text(encoding="utf-8")
    assert all(len(Image.open(path).getexif()) == 0 for path in thumbnails)
