"""Step 1 of the YouTube Reviser RAG pipeline.

Downloads YouTube captions (with timestamps) for the two source playlists and
saves one JSON file per video under transcripts/<domain>/<video_id>.json.

Usage (from the project root):
    python scripts/download_transcripts.py --domain all --limit 5   # test run
    python scripts/download_transcripts.py --domain all             # full run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import yt_dlp

# youtube-transcript-api 1.x uses the instance API (get_transcript() was removed).
from youtube_transcript_api import YouTubeTranscriptApi

from youtube_transcript_api._errors import (  # type: ignore[no-redef]
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)

# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
PLAYLIST_URL_DSA = "https://www.youtube.com/playlist?list=PLgUwDviBIf0oF6QL8m22w1hIDC1vJ_BHz"
PLAYLIST_URL_SD = "https://www.youtube.com/playlist?list=PLMCXHnjXnTnvo6alSjVkgxV-VH6EPyvoX"

DOMAINS: dict[str, str] = {
    "dsa": PLAYLIST_URL_DSA,
    "system_design": PLAYLIST_URL_SD,
}

ROOT = Path(__file__).resolve().parents[1]  # project root, CWD-independent
TRANSCRIPTS_DIR = ROOT / "transcripts"
LOGS_DIR = ROOT / "logs"

LANGUAGES = ["en", "hi"]  # order matters: Hinglish videos often have both
MAX_RETRIES = 3
SLEEP_MIN, SLEEP_MAX = 5.0, 10.0
BATCH_SLEEP_MIN, BATCH_SLEEP_MAX = 60.0, 120.0
MIN_SEGMENTS = 5

# Errors worth retrying (rate limit / network).
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    YouTubeRequestFailed,  # rate limit / blocked request
    ConnectionError,
    TimeoutError,
    OSError,
)

# No captions available: log these and move on (faster-whisper later).
NO_CAPTION_ERRORS: tuple[type[BaseException], ...] = (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)


class TranscriptsUnavailable(Exception):
    """A video has no usable captions (or never became fetchable)."""


def is_rate_limited(error: BaseException | str) -> bool:
    """Recognise provider wording for a temporary block without HTTP coupling."""
    message = str(error).casefold()
    return "429" in message or "too many requests" in message or "blocked" in message


# --------------------------------------------------------------------------- #
# Playlist listing
# --------------------------------------------------------------------------- #
def get_playlist_videos(url: str) -> list[dict[str, str]]:
    """List a playlist's videos (id, title, channel) without downloading media."""
    options = {
        "extract_flat": True,      # metadata only
        "download": False,         # never touch audio/video
        "skip_download": True,
        "ignoreerrors": True,      # private/deleted entries become None
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    # Flat entries lack the channel name; fall back to the playlist's channel.
    playlist_channel = (info or {}).get("channel") or (info or {}).get("uploader") or ""

    videos: list[dict[str, str]] = []
    for entry in (info or {}).get("entries") or []:
        if not entry or not entry.get("id"):
            continue  # private / deleted / unavailable video
        videos.append(
            {
                "id": str(entry["id"]),
                "title": entry.get("title") or str(entry["id"]),
                "channel": entry.get("channel") or entry.get("uploader") or playlist_channel,
            }
        )
    return videos


# --------------------------------------------------------------------------- #
# Caption fetching
# --------------------------------------------------------------------------- #
def _json3_segments(path: Path) -> list[dict[str, Any]]:
    """Convert yt-dlp's JSON3 subtitle format into our timestamped segments."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        text = "".join(part.get("utf8", "") for part in event.get("segs", []))
        text = " ".join(text.split())
        start_ms = event.get("tStartMs")
        if start_ms is None or not text.strip():
            continue
        duration_ms = event.get("dDurationMs", 0)
        segments.append({
            "text": text.strip(),
            "start": round(float(start_ms) / 1000, 3),
            "duration": round(float(duration_ms) / 1000, 3),
        })
    return segments


def clean_segments(segments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated caption events and trim an overlapping text prefix."""
    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        if cleaned:
            previous = cleaned[-1]["text"]
            if text.casefold() == previous.casefold():
                continue
            words = text.split()
            previous_words = previous.split()
            max_overlap = min(len(words), len(previous_words))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if [word.casefold() for word in words[:size]] == [
                    word.casefold() for word in previous_words[-size:]
                ]:
                    overlap = size
                    break
            if overlap:
                text = " ".join(words[overlap:])
                if not text:
                    continue
        cleaned.append({**segment, "text": text})
    return cleaned


def fetch_with_ytdlp(
    video_id: str, cookies_file: Path | None = None
) -> dict[str, Any]:
    """Fetch English captions through yt-dlp when the transcript API is blocked.

    The fallback downloads subtitle metadata only--never the video or audio.  An
    ``cookies_file`` is optional and is passed directly to yt-dlp; browser
    profiles are never read by this script.
    """
    with tempfile.TemporaryDirectory(prefix="yt-reviser-subs-") as temp_dir:
        options: dict[str, Any] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en.*", "hi"],
            "subtitlesformat": "json3",
            "sleep_interval_requests": 2,
            "sleep_interval_subtitles": 5,
            "retries": 3,
            "outtmpl": str(Path(temp_dir) / "%(id)s.%(language)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        if cookies_file:
            options["cookiefile"] = str(cookies_file)
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        subtitle_files = sorted(Path(temp_dir).glob(f"{video_id}.*.json3"))
        if not subtitle_files:
            raise TranscriptsUnavailable("yt-dlp found no English subtitles")
        segments = _json3_segments(subtitle_files[0])
        segments = clean_segments(segments)
        if len(segments) < MIN_SEGMENTS:
            raise TranscriptsUnavailable("yt-dlp subtitle file contained fewer than five timed captions")
        return {"language": "en", "is_generated": True, "segments": segments, "source": "ytdlp"}


def fetch_transcript(
    video_id: str, cookies_file: Path | None = None
) -> dict[str, Any]:
    """Fetch captions with timestamps; returns language, is_generated, segments.

    Retries rate-limit/network errors with exponential backoff (up to MAX_RETRIES).
    Raises TranscriptsUnavailable when captions simply do not exist.
    """
    last_error: BaseException | None = None
    for attempt in range(MAX_RETRIES):
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=LANGUAGES)
            segments = [
                {
                    "text": snippet.text,
                    "start": round(float(snippet.start), 3),
                    "duration": round(float(snippet.duration), 3),
                }
                for snippet in fetched
            ]
            segments = clean_segments(segments)
            if len(segments) < MIN_SEGMENTS:
                last_error = ValueError("transcript contained fewer than five timed captions")
                break
            return {
                "language": getattr(fetched, "language", None) or LANGUAGES[0],
                "is_generated": bool(getattr(fetched, "is_generated", False)),
                "segments": segments,
                "source": "api",
            }
        except RETRYABLE_ERRORS as exc:  # rate limit / network -> backoff
            last_error = exc
            wait = (2**attempt) * random.uniform(0.5, 1.5)
            print(f"      retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s ({exc})")
            time.sleep(wait)
        except NO_CAPTION_ERRORS as exc:
            last_error = exc
            if is_rate_limited(exc) and attempt < MAX_RETRIES - 1:
                wait = (2**attempt) * random.uniform(0.5, 1.5)
                print(f"      retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s ({exc})")
                time.sleep(wait)
                continue
            break

    # youtube-transcript-api is often blocked while yt-dlp's subtitle route is
    # still available.  Keep this as a fallback rather than making the project
    # depend on either provider alone.
    try:
        print("      transcript API unavailable; trying yt-dlp subtitles")
        return fetch_with_ytdlp(video_id, cookies_file)
    except Exception as fallback_error:
        raise TranscriptsUnavailable(
            f"transcript API: {last_error}; yt-dlp fallback: {fallback_error}"
        ) from fallback_error


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_transcript(path: Path, data: dict[str, Any]) -> None:
    """Write one transcript JSON file as utf-8 with readable (non-escaped) text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def valid_transcript(path: Path) -> bool:
    """A saved transcript is reusable only when it has meaningful structure."""
    try:
        if path.stat().st_size == 0:
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data.get("segments"), list) and len(data["segments"]) >= MIN_SEGMENTS
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def missing_path(domain: str) -> Path:
    return ROOT / ("missing_dsa.json" if domain == "dsa" else f"missing_{domain}.json")


def write_missing(domain: str, videos: list[dict[str, str]]) -> None:
    """Persist a small, resumable queue of playlist videos without valid files."""
    missing_path(domain).write_text(
        json.dumps([{"id": v["id"], "title": v["title"], "channel": v.get("channel", "")} for v in videos], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_missing(domain: str) -> list[dict[str, str]]:
    try:
        records = json.loads(missing_path(domain).read_text(encoding="utf-8"))
        return records if isinstance(records, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def playlist_missing(domain: str, videos: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return playlist entries lacking a valid local transcript and update queue."""
    missing = [
        video for video in videos
        if not valid_transcript(TRANSCRIPTS_DIR / domain / f"{video['id']}.json")
    ]
    write_missing(domain, missing)
    print(f"  saved {len(videos) - len(missing)}, missing {len(missing)}")
    return missing


def write_failure_log(domain: str, failures: list[dict[str, Any]]) -> None:
    """Persist the remaining retry queue to logs/failed_<domain>.json."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"failed_{domain}.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(failures, fh, ensure_ascii=False, indent=2)
    print(f"  failure log: {log_path.relative_to(ROOT)} ({len(failures)} videos)")


def load_failure_log(domain: str) -> list[dict[str, str]]:
    """Load the retry queue; a broken/missing log is equivalent to an empty one."""
    path = LOGS_DIR / f"failed_{domain}.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        return records if isinstance(records, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def prioritise(
    videos: list[dict[str, str]], priority_file: Path | None, priority_keywords: str
) -> list[dict[str, str]]:
    """Move explicit IDs and matching titles to the front without changing order."""
    ids: set[str] = set()
    if priority_file:
        try:
            ids = {line.strip() for line in priority_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        except OSError as exc:
            print(f"  WARNING: could not read priority file ({exc})")
    keywords = [item.strip().casefold() for item in priority_keywords.split(",") if item.strip()]
    front = [
        video for video in videos
        if video["id"] in ids or any(keyword in video["title"].casefold() for keyword in keywords)
    ]
    front_ids = {video["id"] for video in front}
    return front + [video for video in videos if video["id"] not in front_ids]


# --------------------------------------------------------------------------- #
# Per-video processing
# --------------------------------------------------------------------------- #
def process_video(
    domain: str,
    video: dict[str, str],
    index: int,
    total: int,
    cookies_file: Path | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Handle one video. Returns (status, failure_record).

    status is one of: "saved", "skipped", "failed".
    """
    video_id = video["id"]
    out_path = TRANSCRIPTS_DIR / domain / f"{video_id}.json"
    title = video["title"]

    if valid_transcript(out_path):
        print(f"[{index}/{total}] Skipping (exists): {title}")
        return "skipped", None

    print(f"[{index}/{total}] Downloading: {title}")
    try:
        transcript = fetch_transcript(video_id, cookies_file)
    except TranscriptsUnavailable as exc:
        print(f"      no captions: {exc}")
        return "failed", {"video_id": video_id, "title": title, "error": str(exc), "attempts": MAX_RETRIES}
    except Exception as exc:  # one bad video never stops the run
        print(f"      error: {type(exc).__name__}: {exc}")
        return "failed", {"video_id": video_id, "title": title, "error": f"{type(exc).__name__}: {exc}", "attempts": MAX_RETRIES}

    data = {
        "video_id": video_id,
        "title": title,
        "channel": video["channel"],
        "domain": domain,
        "language": transcript["language"],
        "is_generated": transcript["is_generated"],
        "url": f"https://youtu.be/{video_id}",
        "segments": transcript["segments"],
        "source": transcript["source"],
    }
    save_transcript(out_path, data)
    method = str(transcript["source"])
    print(f"      saved via {method}: {len(transcript['segments'])} segments "
          f"({transcript['language']}, generated={transcript['is_generated']})")
    return method, None


def download_domain(
    domain: str,
    url: str,
    limit: int | None,
    retry_failed: bool = False,
    cookies_file: Path | None = None,
    video_ids: set[str] | None = None,
    batch_size: int = 25,
    min_delay: float = SLEEP_MIN,
    max_delay: float = SLEEP_MAX,
    priority_file: Path | None = None,
    priority_keywords: str = "",
    dry_run: bool = False,
) -> None:
    """Run the download loop for one playlist and print a summary."""
    started = time.monotonic()
    print(f"\n=== {domain} ===")
    if retry_failed:
        videos = read_missing(domain)
        if not videos:
            print("  no missing videos queued for retry; list the playlist first")
            return
    else:
        try:
            videos = get_playlist_videos(url)
        except Exception as exc:
            print(f"  could not list playlist: {type(exc).__name__}: {exc}")
            sys.exit(1)
        videos = playlist_missing(domain, videos)

    if video_ids:
        videos = [video for video in videos if video["id"] in video_ids]
        missing = video_ids - {video["id"] for video in videos}
        if missing:
            print(f"  requested video IDs not in the retry queue/playlist: {', '.join(sorted(missing))}")

    videos = prioritise(videos, priority_file, priority_keywords)
    if limit is not None:
        videos = videos[:limit]

    total = len(videos)
    print(f"  videos in playlist: {total}" + (f" (limited to {limit})" if limit else ""))

    if dry_run:
        for video in videos:
            print(f"  would download: {video['id']}  {video['title']}")
        return

    counts = {"api": 0, "ytdlp": 0, "skipped": 0, "failed": 0}
    failures: list[dict[str, Any]] = []
    consecutive_limited = 0

    for i, video in enumerate(videos, start=1):
        status, failure = process_video(
            domain, video, i, total, cookies_file
        )
        counts[status] += 1
        if failure is not None:
            failures.append(failure)
            consecutive_limited = consecutive_limited + 1 if is_rate_limited(failure["error"]) else 0
        else:
            consecutive_limited = 0
            # Remove each success immediately so interruption is resumable.
            write_missing(domain, [entry for entry in read_missing(domain) if entry.get("id") != video["id"]])
        print(f"      running: api={counts['api']} ytdlp={counts['ytdlp']} failed={counts['failed']} remaining={total - i}")
        if consecutive_limited >= 3:
            print("YouTube is rate limiting this IP. Wait a few hours, or switch network (home Wi-Fi or phone hotspot, no VPN), then re-run with --retry-failed.")
            break
        if i < total:
            if i % batch_size == 0:
                time.sleep(random.uniform(BATCH_SLEEP_MIN, BATCH_SLEEP_MAX))
            else:
                time.sleep(random.uniform(min_delay, max_delay))

    write_failure_log(domain, failures)
    saved_total = sum(1 for path in (TRANSCRIPTS_DIR / domain).glob("*.json") if valid_transcript(path))
    remaining = len(read_missing(domain))
    print(f"  summary [{domain}]: saved this run={counts['api'] + counts['ytdlp']}, total saved={saved_total}, still missing={remaining}, failed={counts['failed']}, api={counts['api']}, ytdlp={counts['ytdlp']}, elapsed={time.monotonic() - started:.1f}s")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube captions for the YouTube Reviser playlists."
    )
    parser.add_argument(
        "-domain", "--domain",
        choices=["dsa", "system_design", "all"],
        default="all",
        help="which playlist to process (default: all)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry only videos listed in logs/failed_<domain>.json",
    )
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--min-delay", type=float, default=SLEEP_MIN)
    parser.add_argument("--max-delay", type=float, default=SLEEP_MAX)
    parser.add_argument("--priority-file", type=Path, metavar="PATH")
    parser.add_argument(
        "--priority-keywords",
        default="binary search,sliding window,two pointers,linked list,stack,queue,tree,graph,bfs,dfs,dp,recursion",
    )
    parser.add_argument("--cookies-file", type=Path, metavar="PATH",
                        help="optional Netscape cookies.txt passed to yt-dlp only")
    parser.add_argument("--dry-run", action="store_true", help="only list videos that would be downloaded")
    parser.add_argument(
        "--video-id",
        action="append",
        metavar="ID",
        help="download only this video ID (repeatable; useful for targeted recovery)",
    )
    parser.add_argument(
        "-limit", "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only process the first N videos per playlist (for testing)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # Video titles contain emoji/unicode; the Windows console defaults to cp1252.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    args = parse_args(argv)
    if args.batch_size < 1 or args.min_delay < 0 or args.max_delay < args.min_delay:
        parser_error = "--batch-size must be >= 1 and delays must satisfy 0 <= min <= max"
        raise SystemExit(parser_error)
    if args.cookies_file and not args.cookies_file.is_file():
        raise SystemExit(f"cookies file not found: {args.cookies_file}")
    domains = list(DOMAINS) if args.domain == "all" else [args.domain]

    started = time.time()
    for domain in domains:
        download_domain(
            domain,
            DOMAINS[domain],
            args.limit,
            retry_failed=args.retry_failed,
            cookies_file=args.cookies_file,
            video_ids=set(args.video_id) if args.video_id else None,
            batch_size=args.batch_size,
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            priority_file=args.priority_file,
            priority_keywords=args.priority_keywords,
            dry_run=args.dry_run,
        )
    print(f"\nDone in {time.time() - started:.1f}s.")


if __name__ == "__main__":
    main()
