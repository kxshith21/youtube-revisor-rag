from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_playlist_full as full  # noqa: E402


def test_json3_parsing(tmp_path: Path) -> None:
    path = tmp_path / "x.json3"
    path.write_text(json.dumps({"events": [{"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": "hello\nworld"}]}]}), encoding="utf-8")
    assert full.json3_segments(path) == [{"text": "hello world", "start": 1.0, "duration": 0.5}]


def test_priority_order() -> None:
    videos = [{"id": "a", "title": "Misc", "channel": "c"}, {"id": "b", "title": "Sliding Window", "channel": "c"}]
    assert [v["id"] for v in full.priority(videos, None)] == ["b", "a"]


def test_fallback_switches_to_whisper_after_block(monkeypatch) -> None:
    args = Namespace(whisper_only=False, captions_only=False, whisper_model="small", whisper_language="en")
    monkeypatch.setattr(full, "api_captions", lambda _: (_ for _ in ()).throw(full.CaptionBlocked("429")))
    monkeypatch.setattr(full, "ytdlp_captions", lambda _: (_ for _ in ()).throw(full.CaptionBlocked("429")))
    monkeypatch.setattr(full, "whisper_model", lambda _: "model")
    monkeypatch.setattr(full, "whisper_captions", lambda *_: {"language": "en", "is_generated": True, "segments": [{"text": "x", "start": i, "duration": 1} for i in range(5)]})
    method, _, is_blocked, _ = full.run_video({"id": "x", "title": "x", "channel": "c"}, args, None, False)
    assert (method, is_blocked) == ("whisper", True)


def test_valid_file_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "v.json"; path.write_text(json.dumps({"segments": [{"text": "x"}] * 5}), encoding="utf-8")
    assert full.valid(path)
