from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer

from autoretush import __version__
from autoretush.config import AppConfig, ConfigError, load_config, validate_archive_roots
from autoretush.dataset import (
    DatasetError,
    build_materialization_plan,
    load_review_answers,
    materialize,
    split_summary,
)
from autoretush.inventory import PairGroup, build_inventory, discover_groups, write_inventory
from autoretush.pairing.manifest import append_candidates, iter_records
from autoretush.pairing.matcher import match_group
from autoretush.pairing.run import PairingRunError, run_pairing
from autoretush.pairing.snapshot import (
    InputSnapshotError,
    current_pairing_algorithm_id,
    snapshot_pair_group,
)
from autoretush.paths import (
    UnsafePrivatePathError,
    validate_private_output,
    validate_private_workspace,
)
from autoretush.review import build_review_pack, select_review_sample

app = typer.Typer(
    no_args_is_help=True,
    help="Local-first paired retouching dataset tools. Source archives are never modified.",
)


def _config(path: Path) -> AppConfig:
    try:
        config = load_config(path)
        validate_archive_roots(config)
        validate_private_workspace(config.workspace, config.archive_roots)
        validate_private_output(
            config.materialize.destination,
            workspace=config.workspace,
            archive_roots=config.archive_roots,
        )
        return config
    except (ConfigError, UnsafePrivatePathError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc


def _private_path(path: Path, config: AppConfig, *, param_hint: str) -> Path:
    try:
        return validate_private_output(
            path,
            workspace=config.workspace,
            archive_roots=config.archive_roots,
        )
    except UnsafePrivatePathError as exc:
        raise typer.BadParameter(str(exc), param_hint=param_hint) from exc


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
    destination = _private_path(
        output or config.workspace / "inventory" / "inventory.json",
        config,
        param_hint="--output",
    )
    report = build_inventory(config)
    try:
        write_inventory(report, destination)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc), param_hint="--output") from exc
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
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse validated per-group shards after an interrupted full scan.",
        ),
    ] = True,
    clean_staging: Annotated[
        bool,
        typer.Option(
            "--clean-staging",
            help="Delete validated shards only after the final manifest is complete.",
        ),
    ] = False,
) -> None:
    """Match the same whole frame before/after; no face identity model is used."""
    config = _config(config_path)
    destination = _private_path(
        output or config.workspace / "manifests" / "candidates.jsonl",
        config,
        param_hint="--output",
    )
    if destination.exists():
        raise typer.BadParameter(
            f"Output already exists: {destination}. Choose a new file to avoid duplicate records.",
            param_hint="--output",
        )

    all_groups = [group for group in discover_groups(config) if _selected(group, config)]
    grouped = _groups_by_root(all_groups)
    selected_by_root: list[tuple[Path, list[PairGroup]]] = []
    for archive_root in config.archive_roots:
        root_groups = sorted(grouped.get(archive_root, []), key=lambda group: group.group_id)
        if max_groups_per_root:
            root_groups = root_groups[:max_groups_per_root]
        selected_by_root.append((archive_root, root_groups))

    if not target_per_root:
        selected_groups = [group for _, root_groups in selected_by_root for group in root_groups]
        positions = {id(group): index for index, group in enumerate(selected_groups, start=1)}
        run_group_ids: dict[int, str] = {}
        for root_index, (_archive_root, root_groups) in enumerate(selected_by_root):
            for group in root_groups:
                run_group_ids[id(group)] = f"r{root_index}_{group.group_id}"

        typer.echo(f"Fingerprinting {len(selected_groups)} groups before pairing")
        try:
            algorithm_id = current_pairing_algorithm_id()
            input_snapshots: list[str] = []
            for index, group in enumerate(selected_groups, start=1):
                typer.echo(f"  snapshot {index}/{len(selected_groups)} {group.group_id}")
                input_snapshots.append(snapshot_pair_group(group))
        except InputSnapshotError as exc:
            raise typer.BadParameter(str(exc), param_hint="--config") from exc

        def process_group(group: PairGroup):
            position = positions[id(group)]
            typer.echo(f"Group {position}/{len(selected_groups)} {group.group_id}")
            candidates = match_group(group, config.pairing)
            decisions = Counter(item.decision for item in candidates)
            typer.echo(
                f"  accepted={decisions['accepted']} review={decisions['review']} "
                f"rejected={decisions['rejected']}"
            )
            return candidates

        try:
            result = run_pairing(
                destination,
                selected_groups,
                process_group,
                pairing_config=config.pairing,
                selection_config={
                    "inventory": config.inventory,
                    "processed_folder_names": config.processed_folder_names,
                    "selection": config.selection,
                },
                algorithm_id=algorithm_id,
                input_snapshots=input_snapshots,
                group_id_getter=lambda group: run_group_ids[id(group)],
                resume=resume,
                retain_staging=not clean_staging,
            )
        except (PairingRunError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--output") from exc

        totals = Counter(
            str(record.get("decision", "unknown")) for record in iter_records(destination)
        )
        payload = {
            "groups": result.group_count,
            "processed_groups": result.processed_group_count,
            "resumed_groups": result.resumed_group_count,
            "records": result.record_count,
            "decisions": dict(sorted(totals.items())),
            "staging_retained": result.staging_retained,
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        typer.echo(f"Private candidates: {destination}")
        return

    typer.echo(
        "Pilot target mode is intentionally non-resumable; use --target-per-root 0 "
        "for a crash-safe full scan."
    )
    totals: Counter[str] = Counter()
    for root_index, (_archive_root, root_groups) in enumerate(selected_by_root, start=1):
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
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    candidates: Annotated[Path, typer.Option("--candidates", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    size: Annotated[int, typer.Option("--size", min=1, max=1000)] = 200,
    thumbnail_size: Annotated[int, typer.Option("--thumbnail-size", min=320, max=2000)] = 900,
    image_format: Annotated[str, typer.Option("--image-format")] = "webp",
    private_manifest: Annotated[
        Path | None,
        typer.Option(
            "--private-manifest",
            dir_okay=False,
            help="Optional local mapping; it must stay outside the publishable folder.",
        ),
    ] = None,
) -> None:
    """Create an offline HTML review pack with pseudonymous pair IDs."""
    config = _config(config_path)
    candidates = _private_path(candidates, config, param_hint="--candidates")
    output_dir = _private_path(output_dir, config, param_hint="--output-dir")
    if private_manifest is not None:
        private_manifest = _private_path(
            private_manifest,
            config,
            param_hint="--private-manifest",
        )
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
            private_manifest_path=private_manifest,
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Review pairs: {len(selected)}")
    typer.echo(f"Open locally: {index}")


@app.command("materialize")
def materialize_command(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    candidates: Annotated[Path, typer.Option("--candidates", exists=True, dir_okay=False)],
    review_results: Annotated[
        Path | None,
        typer.Option("--review-results", exists=True, dir_okay=False),
    ] = None,
    execute: Annotated[
        bool,
        typer.Option(
            "--execute",
            help="Copy files only after the dry-run summary has been checked.",
        ),
    ] = False,
) -> None:
    """Plan or safely copy selected pairs; the source archive is never modified."""
    config = _config(config_path)
    candidates = _private_path(candidates, config, param_hint="--candidates")
    if review_results is not None:
        review_results = _private_path(
            review_results,
            config,
            param_hint="--review-results",
        )
    try:
        answers = load_review_answers(review_results) if review_results else None
        plan = build_materialization_plan(
            iter_records(candidates),
            config.materialize.destination,
            config.archive_roots,
            review_answers=answers,
            geometry_max_long_side=config.pairing.max_long_side,
            alignment_min_overlap_ratio=config.materialize.min_overlap_ratio,
            alignment_min_edge_correlation=config.materialize.min_edge_correlation,
        )
        payload = plan.to_dict()
        payload["splits"] = split_summary(plan.pairs)
        if execute:
            payload = materialize(plan)
    except DatasetError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if execute:
        typer.echo(f"Private dataset: {plan.destination}")
    else:
        typer.echo("Dry run only: nothing was copied. Re-run with --execute after review.")


if __name__ == "__main__":
    app()
