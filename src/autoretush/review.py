from __future__ import annotations

import html
import json
import os
import random
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageOps

from autoretush.identifiers import validate_pair_id


def _round_robin_groups(records: list[dict[str, object]], *, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("group_id", "unknown"))].append(record)
    queues: list[deque[dict[str, object]]] = []
    for group_records in groups.values():
        rng.shuffle(group_records)
        queues.append(deque(group_records))
    rng.shuffle(queues)

    ordered: list[dict[str, object]] = []
    while queues:
        next_queues: list[deque[dict[str, object]]] = []
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return ordered


def select_review_sample(
    records: Iterable[dict[str, object]],
    *,
    size: int,
    accepted_fraction: float = 0.8,
    seed: int = 20260809,
) -> list[dict[str, object]]:
    """Select across folders and include boundary cases for threshold calibration."""
    accepted = [record for record in records if record.get("decision") == "accepted"]
    review = [record for record in records if record.get("decision") == "review"]
    accepted_target = min(len(accepted), round(size * accepted_fraction))
    review_target = min(len(review), size - accepted_target)
    if accepted_target + review_target < size:
        accepted_target = min(len(accepted), size - review_target)
    if accepted_target + review_target < size:
        review_target = min(len(review), size - accepted_target)

    accepted_order = _round_robin_groups(accepted, seed=seed)
    review_order = _round_robin_groups(review, seed=seed + 1)
    selected = accepted_order[:accepted_target] + review_order[:review_target]
    random.Random(seed + 2).shuffle(selected)
    return selected


def _thumbnail(
    source: Path,
    destination: Path,
    *,
    max_size: int,
    image_format: str,
) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if image_format == "webp":
            image.save(destination, format="WEBP", quality=72, method=6)
        else:
            image.save(destination, format="JPEG", quality=84, optimize=True)


def _card(record: dict[str, object], *, index: int, image_extension: str) -> str:
    pair_id = html.escape(str(record["pair_id"]))
    decision = html.escape(str(record.get("decision", "unknown")))
    score = float(record.get("score", 0.0))
    inliers = int(dict(record.get("geometry", {})).get("inliers", 0))
    ratio = float(dict(record.get("geometry", {})).get("inlier_ratio", 0.0))
    return f"""
    <article class="pair-card" data-pair-id="{pair_id}">
      <header>
        <span class="number">#{index:03d}</span>
        <code>{pair_id}</code>
        <span class="badge {decision}">{decision}</span>
        <span>score {score:.3f} · inliers {inliers} · ratio {ratio:.2f}</span>
      </header>
      <div class="images">
        <figure><img loading="lazy" src="assets/{pair_id}_before.{image_extension}" alt="До"><figcaption>ДО</figcaption></figure>
        <figure><img loading="lazy" src="assets/{pair_id}_after.{image_extension}" alt="После"><figcaption>ПОСЛЕ</figcaption></figure>
      </div>
      <div class="choices" role="group" aria-label="Оценка пары {pair_id}">
        <button data-value="match">✓ Совпало</button>
        <button data-value="wrong">✕ Не тот кадр</button>
        <button data-value="unsure">? Сомневаюсь</button>
      </div>
    </article>
    """


def build_review_pack(
    records: list[dict[str, object]],
    output_dir: Path,
    *,
    thumbnail_size: int = 900,
    image_format: str = "webp",
    private_manifest_path: Path | None = None,
) -> Path:
    image_format = image_format.casefold()
    if image_format not in {"webp", "jpeg"}:
        raise ValueError("image_format must be 'webp' or 'jpeg'")
    pair_ids: list[str] = []
    seen_pair_ids: set[str] = set()
    for record in records:
        pair_id = validate_pair_id(record.get("pair_id"))
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id in review pack: {pair_id}")
        seen_pair_ids.add(pair_id)
        pair_ids.append(pair_id)
    if private_manifest_path is not None:
        resolved_output = output_dir.resolve(strict=False)
        resolved_manifest = private_manifest_path.resolve(strict=False)
        try:
            resolved_manifest.relative_to(resolved_output)
        except ValueError:
            pass
        else:
            raise ValueError("Private manifest must be outside the publishable review directory")
        if os.path.lexists(private_manifest_path):
            raise FileExistsError(f"Private manifest already exists: {private_manifest_path}")
    if os.path.lexists(output_dir):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError("Review output must be a real directory")
        if any(output_dir.iterdir()):
            raise ValueError("Review output directory must be empty")
    else:
        output_dir.mkdir(parents=True)
    image_extension = "webp" if image_format == "webp" else "jpg"
    assets = output_dir / "assets"
    cards: list[str] = []
    private_manifest: list[dict[str, object]] = []

    for index, (record, pair_id) in enumerate(zip(records, pair_ids, strict=True), start=1):
        before = Path(str(record["before_path"]))
        after = Path(str(record["after_path"]))
        _thumbnail(
            before,
            assets / f"{pair_id}_before.{image_extension}",
            max_size=thumbnail_size,
            image_format=image_format,
        )
        _thumbnail(
            after,
            assets / f"{pair_id}_after.{image_extension}",
            max_size=thumbnail_size,
            image_format=image_format,
        )
        cards.append(_card(record, index=index, image_extension=image_extension))
        private_manifest.append(record)

    if private_manifest_path is not None:
        private_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with private_manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(private_manifest, handle, ensure_ascii=False, indent=2)
    page = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow,noarchive,nosnippet,noimageindex">
  <title>AutoRetush — проверка {len(records)} пар</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }}
    body {{ margin: 0; background: #111318; color: #f4f4f5; }}
    .top {{ position: sticky; top: 0; z-index: 10; display: flex; gap: 16px; align-items: center;
      padding: 12px 20px; background: #191c24ee; backdrop-filter: blur(12px); border-bottom: 1px solid #343846; }}
    .top strong {{ margin-right: auto; }}
    .top button {{ background: #5b7cfa; color: white; border: 0; border-radius: 8px; padding: 10px 15px; cursor: pointer; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 18px; display: grid; gap: 18px; }}
    .pair-card {{ background: #1a1d25; border: 2px solid #2d3240; border-radius: 14px; overflow: hidden; }}
    .pair-card[data-answer="match"] {{ border-color: #31b675; }}
    .pair-card[data-answer="wrong"] {{ border-color: #ef5b5b; }}
    .pair-card[data-answer="unsure"] {{ border-color: #e2b84b; }}
    header {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 12px 15px; }}
    .number {{ font-weight: 800; }}
    .badge {{ border-radius: 999px; padding: 3px 9px; background: #3b4254; }}
    .badge.accepted {{ background: #135c3b; }} .badge.review {{ background: #675016; }}
    .images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 3px; background: #07080a; }}
    figure {{ position: relative; margin: 0; min-height: 260px; display: grid; place-items: center; }}
    img {{ display: block; max-width: 100%; max-height: 78vh; object-fit: contain; }}
    figcaption {{ position: absolute; left: 10px; bottom: 10px; padding: 4px 9px; border-radius: 6px; background: #000b; font-weight: 800; }}
    .choices {{ display: flex; gap: 10px; padding: 12px 15px; }}
    .choices button {{ flex: 1; border: 1px solid #53596a; border-radius: 9px; padding: 11px; color: #eee; background: #272b36; cursor: pointer; }}
    .choices button.active {{ outline: 3px solid #8da2ff; }}
    @media (max-width: 800px) {{ .images {{ grid-template-columns: 1fr; }} .choices {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <div class="top">
    <strong>AutoRetush: проверка {len(records)} пар</strong>
    <span id="progress">0 / {len(records)}</span>
    <span id="server-status"></span>
    <button id="save-server">Сохранить на сервере</button>
    <button id="export">Скачать результаты JSON</button>
  </div>
  <main>{"".join(cards)}</main>
  <script>
    const storageKey = 'autoretush-review-{output_dir.name}';
    const answers = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
    const cards = [...document.querySelectorAll('.pair-card')];
    function render() {{
      cards.forEach(card => {{
        const value = answers[card.dataset.pairId];
        if (value) card.dataset.answer = value; else delete card.dataset.answer;
        card.querySelectorAll('button[data-value]').forEach(button =>
          button.classList.toggle('active', button.dataset.value === value));
      }});
      document.getElementById('progress').textContent = `${{Object.keys(answers).length}} / {len(records)}`;
      localStorage.setItem(storageKey, JSON.stringify(answers));
    }}
    cards.forEach(card => card.querySelectorAll('button[data-value]').forEach(button =>
      button.addEventListener('click', () => {{ answers[card.dataset.pairId] = button.dataset.value; render(); }})));
    document.getElementById('export').addEventListener('click', () => {{
      const payload = {{schema_version: 1, generated_at: new Date().toISOString(), answers}};
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
      const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
      link.download = 'review-results.json'; link.click(); URL.revokeObjectURL(link.href);
    }});
    document.getElementById('save-server').addEventListener('click', async () => {{
      const status = document.getElementById('server-status');
      status.textContent = 'Сохраняю…';
      const payload = {{schema_version: 1, generated_at: new Date().toISOString(), answers}};
      try {{
        const response = await fetch('save_review.php', {{
          method: 'POST',
          credentials: 'same-origin',
          headers: {{'Content-Type': 'application/json', 'X-Requested-With': 'AutoRetush'}},
          body: JSON.stringify(payload)
        }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        status.textContent = `Сохранено: ${{Object.keys(answers).length}}`;
      }} catch (error) {{
        status.textContent = 'Не удалось сохранить — скачай JSON';
      }}
    }});
    render();
  </script>
</body>
</html>
"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index
