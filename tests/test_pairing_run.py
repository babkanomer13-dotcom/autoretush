from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from autoretush.pairing.run import (
    CorruptStagingError,
    DuplicateRecordError,
    StagingMismatchError,
    build_run_fingerprint,
)
from autoretush.pairing.run import run_pairing as _run_pairing_impl


@dataclass(frozen=True)
class Group:
    group_id: str
    private_payload: str = ""


PAIRING = {"threshold": 0.81, "shortlist": 8}
SELECTION = {"decision": "accepted"}
ALGORITHM_ID = "autoretush-pairing-v1:" + "a" * 64


def _snapshot(group_id: str) -> str:
    return "sha256:" + hashlib.sha256(group_id.encode("utf-8")).hexdigest()


def run_pairing(output_path, groups, process_group, **kwargs):
    kwargs.setdefault("algorithm_id", ALGORITHM_ID)
    kwargs.setdefault("input_snapshots", [_snapshot(group.group_id) for group in groups])
    return _run_pairing_impl(output_path, groups, process_group, **kwargs)


def _staging(output: Path) -> Path:
    return output.with_name(f".{output.name}.pairing-staging")


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_crash_then_resume_skips_validated_shards_and_preserves_order(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    groups = [Group("g_a"), Group("g_b"), Group("g_c")]
    first_calls: list[str] = []

    def crash_on_second(group: Group):
        first_calls.append(group.group_id)
        if group.group_id == "g_b":
            raise RuntimeError("simulated interruption")
        return [{"pair_id": group.group_id}]

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_pairing(
            output,
            groups,
            crash_on_second,
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )
    assert first_calls == ["g_a", "g_b"]
    assert not output.exists()

    resumed_calls: list[str] = []

    def finish(group: Group):
        resumed_calls.append(group.group_id)
        return [{"pair_id": group.group_id}]

    result = run_pairing(
        output,
        groups,
        finish,
        pairing_config=PAIRING,
        selection_config=SELECTION,
    )

    assert resumed_calls == ["g_b", "g_c"]
    assert result.resumed_group_count == 1
    assert result.processed_group_count == 2
    assert _records(output) == [
        {"pair_id": "g_a"},
        {"pair_id": "g_b"},
        {"pair_id": "g_c"},
    ]


def test_zero_candidate_group_has_completed_empty_shard(tmp_path: Path) -> None:
    output = tmp_path / "empty.jsonl"
    calls = 0

    def empty(_group: Group):
        nonlocal calls
        calls += 1
        return []

    result = run_pairing(
        output,
        [Group("g_empty")],
        empty,
        pairing_config=PAIRING,
        selection_config=SELECTION,
    )

    assert calls == 1
    assert result.record_count == 0
    assert output.read_bytes() == b""
    shards = list(_staging(output).glob("*.jsonl"))
    assert len(shards) == 1
    assert b'"record_count":0' in shards[0].read_bytes()


def test_empty_completed_shard_is_skipped_after_interruption(tmp_path: Path) -> None:
    output = tmp_path / "resume-empty.jsonl"
    groups = [Group("g_empty"), Group("g_later")]

    def interrupt(group: Group):
        if group.group_id == "g_later":
            raise RuntimeError("stop")
        return []

    with pytest.raises(RuntimeError):
        run_pairing(
            output,
            groups,
            interrupt,
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )

    calls: list[str] = []
    result = run_pairing(
        output,
        groups,
        lambda group: calls.append(group.group_id) or [],
        pairing_config=PAIRING,
        selection_config=SELECTION,
    )
    assert calls == ["g_later"]
    assert result.resumed_group_count == 1
    assert output.read_bytes() == b""


def test_changed_fingerprint_rejects_existing_staging(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    groups = [Group("g_a"), Group("g_b")]

    def interrupt(group: Group):
        if group.group_id == "g_b":
            raise RuntimeError("stop")
        return [{"pair_id": group.group_id}]

    with pytest.raises(RuntimeError):
        run_pairing(
            output,
            groups,
            interrupt,
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )

    with pytest.raises(StagingMismatchError):
        run_pairing(
            output,
            groups,
            lambda group: [{"pair_id": group.group_id}],
            pairing_config={**PAIRING, "threshold": 0.9},
            selection_config=SELECTION,
        )


def test_corrupt_completed_shard_is_detected_not_recomputed(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    groups = [Group("g_a"), Group("g_b")]

    def interrupt(group: Group):
        if group.group_id == "g_b":
            raise RuntimeError("stop")
        return [{"pair_id": group.group_id}]

    with pytest.raises(RuntimeError):
        run_pairing(
            output,
            groups,
            interrupt,
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )
    shard = next(_staging(output).glob("*.jsonl"))
    shard.write_bytes(shard.read_bytes()[:-2] + b"\n")

    calls: list[str] = []
    with pytest.raises(CorruptStagingError):
        run_pairing(
            output,
            groups,
            lambda group: calls.append(group.group_id) or [],
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )
    assert calls == []
    assert not output.exists()


def test_final_is_not_visible_until_complete_and_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"

    def produce(group: Group):
        assert not output.exists()
        return [{"pair_id": group.group_id}]

    run_pairing(
        output,
        [Group("g_a")],
        produce,
        pairing_config=PAIRING,
        selection_config=SELECTION,
    )
    original = output.read_bytes()
    calls = 0

    def should_not_run(_group: Group):
        nonlocal calls
        calls += 1
        return []

    with pytest.raises(FileExistsError):
        run_pairing(
            output,
            [Group("g_a")],
            should_not_run,
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )
    assert calls == 0
    assert output.read_bytes() == original


def test_duplicate_records_abort_before_final_publication(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"

    with pytest.raises(DuplicateRecordError):
        run_pairing(
            output,
            [Group("g_a"), Group("g_b")],
            lambda _group: [{"same": True}],
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )

    assert not output.exists()


def test_fingerprint_excludes_payload_and_output_paths_and_rejects_path_config(
    tmp_path: Path,
) -> None:
    snapshots = [_snapshot("g_a")]
    first = build_run_fingerprint(
        ["g_a"],
        PAIRING,
        SELECTION,
        algorithm_id=ALGORITHM_ID,
        input_snapshots=snapshots,
    )
    second = build_run_fingerprint(
        ["g_a"],
        dict(reversed(list(PAIRING.items()))),
        SELECTION,
        algorithm_id=ALGORITHM_ID,
        input_snapshots=snapshots,
    )
    assert first == second

    output_a = tmp_path / "a.jsonl"
    output_b = tmp_path / "nested" / "b.jsonl"
    result_a = run_pairing(
        output_a,
        [Group("g_a", "private-location-a")],
        lambda group: [{"payload": group.private_payload}],
        pairing_config=PAIRING,
        selection_config=SELECTION,
    )
    result_b = run_pairing(
        output_b,
        [Group("g_a", "private-location-b")],
        lambda group: [{"payload": group.private_payload}],
        pairing_config=PAIRING,
        selection_config=SELECTION,
    )
    assert result_a.fingerprint == result_b.fingerprint == first

    with pytest.raises(ValueError, match="must not contain filesystem paths"):
        build_run_fingerprint(
            ["g_a"],
            {"source": tmp_path},
            SELECTION,
            algorithm_id=ALGORITHM_ID,
            input_snapshots=snapshots,
        )


def test_fingerprint_changes_with_algorithm_snapshot_config_and_order() -> None:
    ids = ["g_a", "g_b"]
    snapshots = [_snapshot(group_id) for group_id in ids]

    def fingerprint(*, algorithm_id=ALGORITHM_ID, inputs=snapshots, config=PAIRING):
        return build_run_fingerprint(
            ids,
            config,
            SELECTION,
            algorithm_id=algorithm_id,
            input_snapshots=inputs,
        )

    baseline = fingerprint()
    assert fingerprint(algorithm_id="autoretush-pairing-v1:" + "b" * 64) != baseline
    assert fingerprint(inputs=["sha256:" + "c" * 64, snapshots[1]]) != baseline
    assert fingerprint(config={**PAIRING, "threshold": 0.82}) != baseline
    assert fingerprint(inputs=list(reversed(snapshots))) != baseline


@pytest.mark.parametrize(
    ("algorithm_id", "snapshots", "error"),
    [
        ("pairing:" + "a" * 64, [_snapshot("g_a")], "algorithm_id"),
        (ALGORITHM_ID, [], "exactly one"),
        (ALGORITHM_ID, ["sha256:abc"], "lowercase hex"),
        (ALGORITHM_ID, ["sha256:" + "A" * 64], "lowercase hex"),
    ],
)
def test_invalid_algorithm_or_input_snapshot_is_rejected(
    algorithm_id: str,
    snapshots: list[str],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_run_fingerprint(
            ["g_a"],
            PAIRING,
            SELECTION,
            algorithm_id=algorithm_id,
            input_snapshots=snapshots,
        )


def test_snapshot_change_rejects_completed_staging(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    groups = [Group("g_a"), Group("g_b")]

    def interrupt(group: Group):
        if group.group_id == "g_b":
            raise RuntimeError("stop")
        return [{"pair_id": group.group_id}]

    with pytest.raises(RuntimeError):
        run_pairing(
            output,
            groups,
            interrupt,
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )

    changed = [_snapshot("changed-a"), _snapshot("g_b")]
    with pytest.raises(StagingMismatchError):
        run_pairing(
            output,
            groups,
            lambda _group: [],
            pairing_config=PAIRING,
            selection_config=SELECTION,
            input_snapshots=changed,
        )


def test_schema_one_staging_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    staging = _staging(output)
    staging.mkdir()
    (staging / "run.json").write_text(
        json.dumps(
            {
                "fingerprint": "0" * 64,
                "group_count": 1,
                "kind": "autoretush_pairing_run",
                "schema": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StagingMismatchError):
        run_pairing(
            output,
            [Group("g_a")],
            lambda _group: [],
            pairing_config=PAIRING,
            selection_config=SELECTION,
        )


def test_staging_metadata_contains_no_raw_input_or_output_paths(tmp_path: Path) -> None:
    output = tmp_path / "deep" / "pairs.jsonl"
    private_source = str(tmp_path / "private-source-name")
    run_pairing(
        output,
        [Group("g_a", private_source)],
        lambda _group: [{"pair_id": "p_a"}],
        pairing_config=PAIRING,
        selection_config=SELECTION,
    )

    staging = _staging(output)
    metadata_files = [staging / "run.json", *staging.glob("*.jsonl")]
    metadata_bytes = b"\n".join(path.read_bytes() for path in metadata_files)
    assert private_source.encode("utf-8") not in metadata_bytes
    assert str(tmp_path).encode("utf-8") not in metadata_bytes
    metadata = json.loads((staging / "run.json").read_text(encoding="utf-8"))
    assert set(metadata) == {"fingerprint", "group_count", "kind", "schema"}


def test_cleanup_happens_only_after_final_exists(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    result = run_pairing(
        output,
        [Group("g_a")],
        lambda group: [{"pair_id": group.group_id}],
        pairing_config=PAIRING,
        selection_config=SELECTION,
        retain_staging=False,
    )

    assert output.is_file()
    assert not result.staging_path.exists()
    assert not result.staging_retained


def test_cleanup_refuses_to_delete_an_unexpected_staging_file(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    staging = _staging(output)

    def produce(group: Group):
        (staging / "unrelated.txt").write_text("keep", encoding="utf-8")
        return [{"pair_id": group.group_id}]

    with pytest.raises(CorruptStagingError, match="unexpected entries"):
        run_pairing(
            output,
            [Group("g_a")],
            produce,
            pairing_config=PAIRING,
            selection_config=SELECTION,
            retain_staging=False,
        )

    assert output.is_file()
    assert (staging / "unrelated.txt").read_text(encoding="utf-8") == "keep"
