"""Resumable full-playlist DSA transcript recovery with caption and Whisper fallbacks."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "transcripts" / "dsa"
PROGRESS = ROOT / "progress.json"
FAILURES = ROOT / "failed_dsa.json"
PLAYLIST_URL_DSA = "https://www.youtube.com/playlist?list=PLgUwDviBIf0oF6QL8m22w1hIDC1vJ_BHz"
LANGUAGES = ["en", "hi"]
KEYWORDS = ("binary search", "sliding window", "two pointers", "linked list", "stack", "queue", "tree", "graph", "bfs", "dfs", "dp", "recursion", "array", "sorting")
PROMPT = "Kadane, Dijkstra, monotonic stack, two pointers, sliding window, binary search, DFS, BFS, DP, memoization, tabulation, trie, heap, union find"
MIN_SEGMENTS, MAX_ATTEMPTS, RECHECK_EVERY = 5, 2, 25


class CaptionBlocked(Exception):
    pass


def blocked(exc: BaseException | str) -> bool:
    text = str(exc).casefold()
    return any(token in text for token in ("429", "ipblocked", "requestblocked", "too many requests", "blocked"))


def valid(path: Path) -> bool:
    try:
        return path.stat().st_size > 0 and len(json.loads(path.read_text(encoding="utf-8")).get("segments", [])) >= MIN_SEGMENTS
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_progress() -> dict[str, Any]:
    try:
        data = json.loads(PROGRESS.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"done": {}, "failed": {}}
    except (OSError, ValueError, json.JSONDecodeError):
        return {"done": {}, "failed": {}}


def json3_segments(path: Path) -> list[dict[str, Any]]:
    events = json.loads(path.read_text(encoding="utf-8")).get("events", [])
    result: list[dict[str, Any]] = []
    for event in events:
        text = " ".join("".join(part.get("utf8", "") for part in event.get("segs", [])).split())
        if text and event.get("tStartMs") is not None:
            result.append({"text": text, "start": float(event["tStartMs"]) / 1000, "duration": float(event.get("dDurationMs", 0)) / 1000})
    return result


def playlist() -> list[dict[str, str]]:
    options = {"extract_flat": True, "download": False, "skip_download": True, "ignoreerrors": True, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(PLAYLIST_URL_DSA, download=False) or {}
    channel = info.get("channel") or info.get("uploader") or "take U forward"
    rows = []
    for entry in info.get("entries") or []:
        if not entry or not entry.get("id") or entry.get("live_status") in {"is_live", "is_upcoming"}:
            continue
        rows.append({"id": str(entry["id"]), "title": str(entry.get("title") or entry["id"]), "channel": str(entry.get("channel") or entry.get("uploader") or channel)})
    return rows


def priority(videos: list[dict[str, str]], priority_file: Path | None) -> list[dict[str, str]]:
    ids: set[str] = set()
    if priority_file:
        ids = {line.strip() for line in priority_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    first = [v for v in videos if v["id"] in ids or any(word in v["title"].casefold() for word in KEYWORDS)]
    first_ids = {v["id"] for v in first}
    return first + [v for v in videos if v["id"] not in first_ids]


def api_captions(video_id: str) -> dict[str, Any]:
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=LANGUAGES)
        return {"language": getattr(fetched, "language", "en"), "is_generated": bool(getattr(fetched, "is_generated", False)), "segments": [{"text": s.text, "start": float(s.start), "duration": float(s.duration)} for s in fetched]}
    except Exception as exc:
        if blocked(exc):
            raise CaptionBlocked(str(exc)) from exc
        raise


def ytdlp_captions(video_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dsa-subs-") as directory:
        opts = {"skip_download": True, "writesubtitles": True, "writeautomaticsub": True, "subtitleslangs": ["en.*", "hi"], "subtitlesformat": "json3", "sleep_interval_subtitles": 5, "retries": 3, "outtmpl": str(Path(directory) / "%(id)s.%(language)s.%(ext)s"), "quiet": True, "no_warnings": True}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as exc:
            if blocked(exc):
                raise CaptionBlocked(str(exc)) from exc
            raise
        files = sorted(Path(directory).glob(f"{video_id}.*.json3"))
        if not files:
            raise RuntimeError("no JSON3 subtitles")
        segments = json3_segments(files[0])
        if len(segments) < MIN_SEGMENTS:
            raise RuntimeError("fewer than five subtitle segments")
        return {"language": "en", "is_generated": True, "segments": segments}


def audio(video_id: str, directory: Path) -> Path:
    if shutil.disk_usage(directory).free < 1_000_000_000:
        raise RuntimeError("less than 1 GB free disk space for Whisper audio")
    opts = {"format": "bestaudio/best", "outtmpl": str(directory / "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        return Path(ydl.prepare_filename(info))


def whisper_captions(video_id: str, model: Any, language: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dsa-audio-") as directory:
        file = audio(video_id, Path(directory))
        segments, info = model.transcribe(str(file), language=language, vad_filter=True, beam_size=5, initial_prompt=PROMPT)
        rows = [{"text": s.text.strip(), "start": float(s.start), "duration": float(s.end - s.start)} for s in segments if s.text.strip()]
        if len(rows) < MIN_SEGMENTS:
            raise RuntimeError("Whisper returned fewer than five segments")
        return {"language": getattr(info, "language", language or "unknown"), "is_generated": True, "segments": rows}


def whisper_model(name: str) -> Any:
    from faster_whisper import WhisperModel
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    return WhisperModel(name, device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")


def run_video(video: dict[str, str], args: argparse.Namespace, model: Any, captions_blocked: bool) -> tuple[str, dict[str, Any], bool, Any]:
    methods = ["whisper"] if args.whisper_only or captions_blocked else (["api", "ytdlp"] if args.captions_only else ["api", "ytdlp", "whisper"])
    errors: list[str] = []
    for method in methods:
        try:
            if method == "api": data = api_captions(video["id"])
            elif method == "ytdlp": data = ytdlp_captions(video["id"])
            else:
                if model is None: model = whisper_model(args.whisper_model)
                data = whisper_captions(video["id"], model, args.whisper_language)
            if len(data["segments"]) >= MIN_SEGMENTS:
                return method, data, captions_blocked, model
        except CaptionBlocked as exc:
            captions_blocked, errors = True, errors + [f"{method}: {exc}"]
        except Exception as exc:
            errors.append(f"{method}: {exc}")
    raise RuntimeError("; ".join(errors) or "no transcript method succeeded")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Recover the full Striver A2Z DSA playlist.")
    parser.add_argument("--limit", type=int); parser.add_argument("--whisper-only", action="store_true"); parser.add_argument("--captions-only", action="store_true")
    parser.add_argument("--whisper-model", default="small"); parser.add_argument("--whisper-language"); parser.add_argument("--priority-file", type=Path)
    parser.add_argument("--min-delay", type=float, default=5); parser.add_argument("--max-delay", type=float, default=15); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.whisper_only and args.captions_only: parser.error("--whisper-only and --captions-only cannot be combined")
    videos = priority(playlist(), args.priority_file)
    if args.limit: videos = videos[:args.limit]
    progress, model, captions_blocked = load_progress(), None, False
    pending = [v for v in videos if not valid(OUT / f"{v['id']}.json") and progress["failed"].get(v["id"], {}).get("attempts", 0) < MAX_ATTEMPTS]
    if args.dry_run:
        print("\n".join(f"{v['id']} {v['title']}" for v in pending)); return
    totals: dict[str, list[float]] = {"api": [], "ytdlp": [], "whisper": []}
    for index, video in enumerate(pending, 1):
        started = time.monotonic(); method = "failed"
        print(f"[{index}/{len(pending)}] {method} {video['title']}")
        try:
            method, data, captions_blocked, model = run_video(video, args, model, captions_blocked)
            record = {"video_id": video["id"], "title": video["title"], "channel": video["channel"], "domain": "dsa", "language": data["language"], "is_generated": data["is_generated"], "url": f"https://youtu.be/{video['id']}", "source": method, "segments": data["segments"]}
            save_json(OUT / f"{video['id']}.json", record); elapsed = time.monotonic() - started; totals[method].append(elapsed)
            progress["done"][video["id"]] = {"method": method, "seconds": elapsed}; progress["failed"].pop(video["id"], None)
        except Exception as exc:
            old = progress["failed"].get(video["id"], {}); progress["failed"][video["id"]] = {"title": video["title"], "error": str(exc), "attempts": old.get("attempts", 0) + 1}
        save_json(PROGRESS, progress); save_json(FAILURES, [{"video_id": k, **v} for k, v in progress["failed"].items()])
        remaining = len(pending) - index; avgs = {k: sum(v) / len(v) for k, v in totals.items() if v}; estimate = remaining * (sum(avgs.values()) / len(avgs) if avgs else 0)
        print(f"  saved={len(progress['done'])} failed={len(progress['failed'])} remaining={remaining} ETA={estimate / 60:.1f}m averages={{{', '.join(f'{k}:{v:.1f}s' for k,v in avgs.items())}}}")
        if index < len(pending): time.sleep(random.uniform(args.min_delay, args.max_delay))
        if captions_blocked and index % RECHECK_EVERY == 0: captions_blocked = False
    print({method: len(times) for method, times in totals.items()})


if __name__ == "__main__": main()
