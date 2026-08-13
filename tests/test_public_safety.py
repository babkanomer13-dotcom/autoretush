from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_public_repo.py"
SPEC = importlib.util.spec_from_file_location("check_public_repo", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SAFETY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAFETY
SPEC.loader.exec_module(SAFETY)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _new_repository(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.name", "Safety Test")
    _git(root, "config", "user.email", "safety@example.invalid")


@pytest.mark.parametrize(
    ("path", "expected_reason"),
    [
        ("private" + "/notes.txt", "forbidden private/data directory"),
        ("sample." + "jpg", "forbidden binary/data file type"),
        ("face_landmarker." + "task", "forbidden binary/data file type"),
        ("certificate." + "p12", "forbidden binary/data file type"),
        ("credentials" + ".json", "forbidden credential filename"),
        ("configs/" + "local.yaml", "forbidden local configuration"),
        ("renamed-model.bin", "unsupported public file type"),
        ("src/autoretush/models/interface.py", "forbidden private/data directory"),
        ("tests/test_global_lut.py", "forbidden proprietary implementation"),
    ],
)
def test_path_reasons_reject_private_artifacts(path: str, expected_reason: str) -> None:
    assert expected_reason in SAFETY.path_reasons(path)


def test_content_reasons_detect_secrets_and_private_paths_without_returning_values() -> None:
    secret = "T3st-Only-Credential-4729"
    separator = chr(92)
    drive_path = "D:" + separator + "client-archive" + separator + "season"
    content = ("password" + "=" + secret + "\nsource=" + drive_path).encode()

    reasons = SAFETY.content_reasons(content)

    assert "inline credential assignment" in reasons
    assert "non-placeholder Windows absolute path" in reasons
    assert all(secret not in reason and drive_path not in reason for reason in reasons)


def test_placeholder_credentials_are_allowed() -> None:
    content = ("api_" + "key=${EXAMPLE_API_KEY}\npassword=<your-password>").encode()

    assert "inline credential assignment" not in SAFETY.content_reasons(content)


def test_binary_media_is_detected_even_with_a_text_extension() -> None:
    disguised_png = b"\x89PNG\r\n\x1a\n" + b"not-a-real-image"

    reasons = SAFETY.content_reasons(disguised_png, path="notes.txt")

    assert "binary/media file signature" in reasons
    assert "binary or non-UTF-8 content" in reasons


@pytest.mark.parametrize("mime", ["webp", "jpeg", "png"])
def test_embedded_raster_data_uri_is_rejected(mime: str) -> None:
    content = ("const preview = 'data:" + f"image/{mime};base64,AAAA';").encode()

    assert "embedded raster image data URI" in SAFETY.content_reasons(content)


@pytest.mark.parametrize(
    "payload",
    [
        "image/png;charset=utf-8;base64,AAAA",
        "image/svg+xml;base64,PHN2Zz4=",
        "image/png,iVBORw0KGgo",
    ],
)
def test_any_image_data_uri_variant_is_rejected(payload: str) -> None:
    content = ("const preview = 'data:" + payload + "';").encode()

    assert "embedded raster image data URI" in SAFETY.content_reasons(content)


@pytest.mark.parametrize(
    ("drive_path", "source_path"),
    [
        ("E:" + "/archive", "tests/test_config.py"),
        ("E:" + "/local/dataset", "tests/test_config.py"),
        (
            "X:" + "/autoretush-local/manifests/candidates.jsonl",
            "docs/DATASET.md",
        ),
        ("X:" + "/private-archive/season-a", "configs/local.example.yaml"),
    ],
)
def test_exact_documented_windows_examples_are_allowed(
    drive_path: str,
    source_path: str,
) -> None:
    assert "non-placeholder Windows absolute path" not in SAFETY.content_reasons(
        ("source=" + drive_path).encode(),
        path=source_path,
    )


@pytest.mark.parametrize(
    "example_root",
    [
        "E:" + "/archive",
        "E:" + "/local",
        "X:" + "/autoretush-local",
        "X:" + "/private-archive",
    ],
)
def test_real_suffix_under_example_root_is_rejected(example_root: str) -> None:
    private_path = example_root + "/real-school/child-name.jpg"

    assert "non-placeholder Windows absolute path" in SAFETY.content_reasons(
        ("source=" + private_path).encode(),
        path="configs/local.example.yaml",
    )


def test_exact_example_path_is_rejected_outside_its_documented_file() -> None:
    example = "X:" + "/private-archive/season-a"

    assert "non-placeholder Windows absolute path" in SAFETY.content_reasons(
        ("source=" + example).encode(),
        path="notes.json",
    )


def test_retired_local_workspace_path_is_rejected_in_current_content() -> None:
    path = "E:" + "/autoretush_local/reviews"

    assert "non-placeholder Windows absolute path" in SAFETY.content_reasons(
        ("source=" + path).encode(),
        path="docs/DATASET.md",
    )


def test_real_archive_and_season_names_are_rejected() -> None:
    archive_name = "".join(
        chr(codepoint) for codepoint in (1040, 1083, 1100, 1073, 1086, 1084, 1099)
    )
    archive_name += " "
    archive_name += "".join(
        chr(codepoint) for codepoint in (1087, 1088, 1086, 1096, 1083, 1099, 1077)
    )
    archive_name += " "
    archive_name += "".join(chr(codepoint) for codepoint in (1075, 1086, 1076, 1072))

    assert "real archive or season name" in SAFETY.content_reasons(archive_name.encode())


def test_scan_repository_reads_staged_blob_instead_of_only_worktree(tmp_path: Path) -> None:
    _new_repository(tmp_path)
    tracked = tmp_path / "settings.txt"
    tracked.write_text("safe=true\n", encoding="utf-8")
    _git(tmp_path, "add", "settings.txt")
    _git(tmp_path, "commit", "-m", "safe")

    secret = "Staged-Credential-5831"
    tracked.write_text("password" + "=" + secret + "\n", encoding="utf-8")
    _git(tmp_path, "add", "settings.txt")
    tracked.write_text("safe=true\n", encoding="utf-8")

    findings = SAFETY.scan_repository(tmp_path, include_history=False)

    assert any(
        finding.scope == "index"
        and finding.path == "settings.txt"
        and finding.reason == "inline credential assignment"
        for finding in findings
    )
    assert all(secret not in finding.render() for finding in findings)


def test_scan_repository_rejects_staged_disguised_image(tmp_path: Path) -> None:
    _new_repository(tmp_path)
    disguised = tmp_path / "notes.txt"
    disguised.write_bytes(b"\x89PNG\r\n\x1a\n" + b"private-photo-bytes")
    _git(tmp_path, "add", "notes.txt")

    findings = SAFETY.scan_repository(tmp_path, include_history=False)

    assert any(
        finding.scope == "index"
        and finding.path == "notes.txt"
        and finding.reason == "binary/media file signature"
        for finding in findings
    )


def test_scan_repository_checks_reachable_history(tmp_path: Path) -> None:
    _new_repository(tmp_path)
    tracked = tmp_path / "settings.txt"
    tracked.write_text(
        "source=" + "D:" + chr(92) + "private-archive\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", "settings.txt")
    _git(tmp_path, "commit", "-m", "old")
    tracked.write_text("safe=true\n", encoding="utf-8")
    _git(tmp_path, "add", "settings.txt")
    _git(tmp_path, "commit", "-m", "safe")

    findings = SAFETY.scan_repository(tmp_path, include_history=True)

    assert any(
        finding.scope == "history"
        and finding.path == "settings.txt"
        and finding.reason == "non-placeholder Windows absolute path"
        for finding in findings
    )


def test_history_keeps_old_path_when_blob_is_current_under_safe_name(tmp_path: Path) -> None:
    _new_repository(tmp_path)
    old_directory = tmp_path / "private"
    old_directory.mkdir()
    old_path = old_directory / "notes.txt"
    old_path.write_text("safe=true\n", encoding="utf-8")
    _git(tmp_path, "add", "private/notes.txt")
    _git(tmp_path, "commit", "-m", "old path")
    _git(tmp_path, "mv", "private/notes.txt", "notes.txt")
    _git(tmp_path, "commit", "-m", "safe path")

    findings = SAFETY.scan_repository(tmp_path, include_history=True)

    assert any(
        finding.scope == "history"
        and finding.path == "private/notes.txt"
        and finding.reason == "forbidden private/data directory"
        for finding in findings
    )


def test_history_metadata_rejects_personal_commit_email(tmp_path: Path) -> None:
    _new_repository(tmp_path)
    _git(tmp_path, "config", "user.email", "developer@example.com")
    tracked = tmp_path / "notes.txt"
    tracked.write_text("safe=true\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.txt")
    _git(tmp_path, "commit", "-m", "safe")

    findings = SAFETY.scan_repository(tmp_path, include_history=True)

    assert any(
        finding.scope == "history-metadata"
        and finding.path == "commit"
        and finding.reason == "non-private author or committer email"
        for finding in findings
    )


def test_history_metadata_rejects_secret_in_commit_and_tag_messages(tmp_path: Path) -> None:
    _new_repository(tmp_path)
    tracked = tmp_path / "notes.txt"
    tracked.write_text("safe=true\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.txt")
    commit_message = "password" + "=" + "CommitSecret4729"
    tag_message = "password" + "=" + "TagSecret5831"
    _git(tmp_path, "commit", "-m", commit_message)
    _git(tmp_path, "tag", "-a", "v1", "-m", tag_message)

    findings = SAFETY.scan_repository(tmp_path, include_history=True)

    assert any(
        finding.scope == "history-metadata"
        and finding.path == "commit"
        and finding.reason == "inline credential assignment"
        for finding in findings
    )
    assert any(
        finding.scope == "history-metadata"
        and finding.path == "annotated tag"
        and finding.reason == "inline credential assignment"
        for finding in findings
    )
