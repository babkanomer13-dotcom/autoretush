from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer

from autoretush import __version__
from autoretush.config import AppConfig, ConfigError, load_config, validate_archive_roots
from autoretush.inventory import PairGroup, build_inventory, discover_groups, write_inventory
from autoretush.pairing.manifest import append_candidates, iter_records
from autoretush.pairing.matcher import match_group
from autoretush.review import build_review_pack, select_review_sample

app = typer.Typer(
    no_args_is_help=True,
    help="Local-first paired retouching dataset tools. Source archives are never modified.",
)


def _config(path: Path) -> AppConfig:
    try:
        config = load_config(path)
        validate_archive_roots(config)
        return config
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc


def _selected(group: PairGroup, config: AppConfig) -> bool:
    relative = str(group.source_dir).casefold()
    includes = config.selection.include_path_keywords
    excludes = config.selection.exclude_path_keywords
    return (not includes or any(word in relative for word in includes)) and not any(
        word in relative for word in excludes
    )


def _groups_by_root(groups: Iterable[PairGroup]) -> dict[Path, list[PairGroup]]:
    result: dict[Path, list[PairGroup]] = defaultdict(list)
    for group in groups:
        result[group.archive_root].append(group)
    return result


@app.callback()
def root(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command("inventory")
def inventory_command(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
) -> None:
    """Build a read-only structural inventory."""
    config = _config(config_path)
    destination = output or config.workspace / "inventory" / "inventory.json"
    report = build_inventory(config)
    write_inventory(report, destination)
    summary = {key: value for key, value in report.items() if key != "groups"}
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    typer.echo(f"Private report: {destination}")


@app.command("pair")
def pair_command(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
    target_per_root: Annotated[
        int,
        typer.Option(
            "--target-per-root",
            min=0,
            help="Stop each root after this many accepted/review pairs; 0 scans all.",
        ),
    ] = 0,
    max_groups_per_root: Annotated[
        int,
        typer.Option("--max-groups-per-root", min=0, help="Optional pilot cap per archive root."),
    ] = 0,
) -> None:
    """Match the same whole frame before/after; no face identity model is used."""
    config = _config(config_path)
    destination = output or config.workspace / "manifests" / "candidates.jsonl"
    if destination.exists():
        raise typer.BadParameter(
            f"Output already exists: {destination}. Choose a new file to avoid duplicate records.",
            param_hint="--output",
        )

    all_groups = [group for group in discover_groups(config) if _selected(group, config)]
    grouped = _groups_by_root(all_groups)
    totals: Counter[str] = Counter()
    for root_index, archive_root in enumerate(config.archive_roots, start=1):
        root_groups = grouped.get(archive_root, [])
        if max_groups_per_root:
            root_groups = root_groups[:max_groups_per_root]
        root_hits = 0
        typer.echo(
            f"Root {root_index}/{len(config.archive_roots)}: {len(root_groups)} selected groups"
        )
        for group_index, group in enumerate(root_groups, start=1):
            candidates = match_group(group, config.pairing)
            append_candidates(destination, candidates)
            decisions = Counter(item.decision for item in candidates)
            totals.update(decisions)
            root_hits += decisions["accepted"] + decisions["review"]
            typer.echo(
                f"  group {group_index}/{len(root_groups)} {group.group_id}: "
                f"accepted={decisions['accepted']} review={decisions['review']} "
                f"rejected={decisions['rejected']}"
            )
            if target_per_root and root_hits >= target_per_root:
                break

    typer.echo(json.dumps(dict(totals), ensure_ascii=False, indent=2))
    typer.echo(f"Private candidates: {destination}")


@app.command("review-sample")
def review_sample_command(
    candidates: Annotated[Path, typer.Option("--candidates", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    size: Annotated[int, typer.Option("--size", min=1, max=1000)] = 200,
    thumbnail_size: Annotated[int, typer.Option("--thumbnail-size", min=320, max=2000)] = 900,
    image_format: Annotated[str, typer.Option("--image-format")] = "webp",
) -> None:
    """Create an offline HTML review pack with pseudonymous pair IDs."""
    records = list(iter_records(candidates))
    selected = select_review_sample(records, size=size)
    if not selected:
        raise typer.BadParameter("No accepted/review candidates are available")
    try:
        index = build_review_pack(
            selected,
            output_dir,
            thumbnail_size=thumbnail_size,
            image_format=image_format,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--image-format") from exc
    typer.echo(f"Review pairs: {len(selected)}")
    typer.echo(f"Open locally: {index}")


if __name__ == "__main__":
    app()
