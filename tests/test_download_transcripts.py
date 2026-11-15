"""Tests for the resumable transcript downloader (all network calls mocked)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_transcripts as downloader  # noqa: E402


def configure_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(downloader, "ROOT", tmp_path)
    monkeypatch.setattr(downloader, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(downloader, "LOGS_DIR", tmp_path / "logs")


def transcript(segments: int = 5) -> dict[str, object]:
    return {"segments": [{"text": str(i), "start": i, "duration": 1} for i in range(segments)]}


def test_playlist_missing_detects_invalid_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_paths(monkeypatch, tmp_path)
    folder = downloader.TRANSCRIPTS_DIR / "dsa"
    folder.mkdir(parents=True)
    (folder / "ok.json").write_text(json.dumps(transcript()), encoding="utf-8")
    (folder / "bad.json").write_text("{broken", encoding="utf-8")
    videos = [{"id": "ok", "title": "Ok", "channel": "c"}, {"id": "bad", "title": "Bad", "channel": "c"}]

    assert downloader.playlist_missing("dsa", videos) == [videos[1]]
    assert json.loads((tmp_path / "missing_dsa.json").read_text(encoding="utf-8"))[0]["id"] == "bad"


def test_json3_parsing_and_duplicate_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "captions.json3"
    source.write_text(json.dumps({"events": [
        {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello world\n"}]},
        {"tStartMs": 1000, "dDurationMs": 1000, "segs": [{"utf8": "hello world"}]},
        {"tStartMs": 2000, "dDurationMs": 1000, "segs": [{"utf8": "world again"}]},
    ]}), encoding="utf-8")

    parsed = downloader.clean_segments(downloader._json3_segments(source))
    assert [segment["text"] for segment in parsed] == ["hello world", "again"]
    assert parsed[0]["start"] == 0.0


def test_valid_file_is_never_overwritten(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_paths(monkeypatch, tmp_path)
    path = downloader.TRANSCRIPTS_DIR / "dsa" / "v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(transcript()), encoding="utf-8")
    monkeypatch.setattr(downloader, "fetch_transcript", lambda *_: pytest.fail("must not fetch"))

    status, _ = downloader.process_video("dsa", {"id": "v1", "title": "T", "channel": "C"}, 1, 1)
    assert status == "skipped"


def test_circuit_breaker_stops_after_three_rate_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_paths(monkeypatch, tmp_path)
    queue = [{"id": f"v{i}", "title": f"Video {i}", "channel": "C"} for i in range(4)]
    downloader.write_missing("dsa", queue)
    calls: list[str] = []

    def blocked(video_id: str, *_: object) -> dict[str, object]:
        calls.append(video_id)
        raise downloader.TranscriptsUnavailable("HTTP 429 too many requests")

    monkeypatch.setattr(downloader, "fetch_transcript", blocked)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    downloader.download_domain("dsa", "unused", None, retry_failed=True, min_delay=0, max_delay=0)

    assert calls == ["v0", "v1", "v2"]
