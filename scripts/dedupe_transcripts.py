"""Report, quarantine, restore, and optionally purge duplicate transcripts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "transcripts"
LOGS = ROOT / "logs"
DOMAINS = ("dsa", "system_design")
MIN_SEGMENTS = 5


def save_json(path: Path, value: Any) -> None:
    """Save data to a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def load_transcript(path: Path, domain: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segs = data.get("segments")
        segs_list = segs if isinstance(segs, list) else []
        errors = []

        for key in ("video_id", "title", "segments"):
            if not data.get(key):
                errors.append(f"missing {key}")

        if not isinstance(segs, list) or len(segs_list) < MIN_SEGMENTS:
            errors.append("fewer than 5 segments")

        words = sum(
            len(str(s.get("text", "")).split())
            for s in segs_list
            if isinstance(s, dict)
        )
        starts = [
            _safe_float(s.get("start"), -1.0)
            for s in segs_list
            if isinstance(s, dict)
        ]

        if words < 100:
            errors.append("fewer than 100 words")
        if any(b < a for a, b in zip(starts, starts[1:])):
            errors.append("timestamps not increasing")

        repeats = sum(
            1
            for a, b in zip(segs_list, segs_list[1:])
            if isinstance(a, dict)
            and isinstance(b, dict)
            and (t := str(a.get("text", "")).strip())
            and t == str(b.get("text", "")).strip()
        )

        last_seg = segs_list[-1] if segs_list and isinstance(segs_list[-1], dict) else {}
        duration = _safe_float(last_seg.get("start"), 0.0) + _safe_float(last_seg.get("duration"), 0.0)
        repeat_pct = round(100 * repeats / max(1, len(segs_list) - 1), 2)

        return {
            "path": str(path),
            "name": path.name,
            "domain": domain,
            "data": data,
            "valid": not errors,
            "errors": errors,
            "words": words,
            "segments": len(segs_list),
            "duration": duration,
            "repeats": repeats,
            "repeat_pct": repeat_pct,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "name": path.name,
            "domain": domain,
            "data": {},
            "valid": False,
            "errors": [f"malformed JSON: {exc}"],
            "words": 0,
            "segments": 0,
            "duration": 0.0,
            "repeats": 0,
            "repeat_pct": 0.0,
        }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.casefold())).strip()


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = normalize_text(text).split()
    return {tuple(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def jaccard(left: set[Any], right: set[Any]) -> float:
    return len(left & right) / len(left | right) if (left or right) else 0.0


def title_key(record: dict[str, Any]) -> str:
    return " ".join(str(record["data"].get("title", "")).casefold().split())


def content(record: dict[str, Any]) -> str:
    segs = record["data"].get("segments", [])
    if not isinstance(segs, list):
        return ""
    return " ".join(str(s.get("text", "")) for s in segs if isinstance(s, dict))


def choose_keeper(group: list[dict[str, Any]]) -> dict[str, Any]:
    def rank(r: dict[str, Any]) -> tuple[Any, ...]:
        d = r["data"]
        source = d.get("source", "")
        manual = source in {"api", "ytdlp"} and not d.get("is_generated", False)
        mtime = Path(r["path"]).stat().st_mtime if Path(r["path"]).exists() else 0.0
        return (
            r["valid"],
            manual,
            not d.get("is_generated", False),
            r["segments"],
            r["duration"],
            Path(r["path"]).stem == d.get("video_id"),
            mtime,
        )

    return max(group, key=rank)


def find_groups(
    records: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = []
    clean = []
    by_video = defaultdict(list)
    by_title = defaultdict(list)

    for r in records:
        d = r["data"]
        by_video[d.get("video_id")].append(r)
        by_title[title_key(r)].append(r)
        if r["repeat_pct"]:
            clean.append(r)
        if d.get("video_id") and Path(r["path"]).stem != d["video_id"]:
            groups.append(
                {"reason": "filename/video_id mismatch", "records": [r], "score": None}
            )

    for reason, bucket in (("same video id", by_video), ("duplicate title", by_title)):
        for key, items in bucket.items():
            if key and len(items) > 1:
                groups.append({"reason": reason, "records": items, "score": 1.0})

    for i, left in enumerate(records):
        a = shingles(content(left))
        for right in records[i + 1 :]:
            if left["data"].get("video_id") == right["data"].get("video_id"):
                continue
            score = jaccard(a, shingles(content(right)))
            max_words = max(left["words"], right["words"])
            similar_words = abs(left["words"] - right["words"]) / max(1, max_words) < 0.05
            if score >= threshold and similar_words:
                groups.append(
                    {
                        "reason": "near duplicate content",
                        "records": [left, right],
                        "score": round(score, 4),
                    }
                )

    return groups, clean


def quarantine(records: list[dict[str, Any]], yes: bool) -> list[dict[str, Any]]:
    unique = {r["path"]: r for r in records}
    rows = list(unique.values())
    if not rows:
        return []

    for r in rows:
        print(f"  {r['path']}")

    if not yes and input(f"Move {len(rows)} files to quarantine? (y/N) ").strip().casefold() != "y":
        return []

    actions = []
    for r in rows:
        src = Path(r["path"])
        if not src.exists():
            continue
        dst = TRANSCRIPTS / "_quarantine" / r["domain"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        actions.append(
            {
                "file": str(src),
                "quarantine": str(dst),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return actions


def restore(domains: list[str]) -> int:
    count = 0
    for domain in domains:
        folder = TRANSCRIPTS / "_quarantine" / domain
        for path in folder.glob("*.json") if folder.exists() else []:
            target = TRANSCRIPTS / domain / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(path), str(target))
                count += 1
    return count


def clean_repeats(records: list[dict[str, Any]]) -> int:
    changed = 0
    for r in records:
        if not r["repeats"]:
            continue
        path = Path(r["path"])
        backup = TRANSCRIPTS / "_backup" / r["domain"] / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        data = r["data"]
        segs = data.get("segments", [])
        if not isinstance(segs, list):
            continue
        seen: list[dict[str, Any]] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            if (
                not seen
                or not isinstance(seen[-1], dict)
                or str(seen[-1].get("text", "")).strip() != str(seg.get("text", "")).strip()
            ):
                seen.append(seg)
        data["segments"] = seen
        save_json(path, data)
        changed += 1
    return changed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Report, quarantine, restore, and purge duplicate transcripts.")
    p.add_argument("--domain", choices=[*DOMAINS, "all"], default="all")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--restore", action="store_true")
    p.add_argument("--clean-repeats", action="store_true")
    p.add_argument("--purge-qdrant", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    domains = list(DOMAINS) if a.domain == "all" else [a.domain]
    if a.restore:
        print(f"Restored {restore(domains)} files.")
        return

    records = [
        load_transcript(path, d)
        for d in domains
        for path in (TRANSCRIPTS / d).glob("*.json")
    ]
    groups, needs_clean = find_groups(records, a.threshold)
    invalid = [r for r in records if not r["valid"]]

    moves = []
    for group in groups:
        if len(group["records"]) < 2:
            continue
        keep = choose_keeper(group["records"])
        group["keep"] = keep["path"]
        moves += [r for r in group["records"] if r is not keep]
        print(
            f"{group['reason']}: keep {keep['name']}; "
            f"move {[r['name'] for r in group['records'] if r is not keep]}; "
            f"score={group['score']}"
        )

    moves += invalid
    report = {
        "scanned": len(records),
        "invalid": invalid,
        "groups": groups,
        "needs_cleaning": needs_clean,
        "would_move": [r["path"] for r in {r["path"]: r for r in moves}.values()],
    }
    LOGS.mkdir(exist_ok=True)
    save_json(LOGS / "dedupe_report.json", report)

    print(
        f"scanned={len(records)} "
        f"unique videos={len({r['data'].get('video_id') for r in records})} "
        f"invalid={len(invalid)} "
        f"groups={len(groups)} "
        f"needs cleaning={len(needs_clean)} "
        f"would move={len(report['would_move'])}"
    )

    if a.clean_repeats:
        print(f"Cleaned {clean_repeats(needs_clean)} files (backups in transcripts/_backup).")

    actions = quarantine(moves, a.yes) if a.apply else []
    if actions:
        save_json(LOGS / "dedupe_log.json", actions)
        print("Rebuild chunks and remove moved videos' Qdrant points before serving.")

    if a.purge_qdrant and actions:
        print(
            "--purge-qdrant requested: rerun after reviewing quarantine; "
            "automatic purge is intentionally not performed without a second explicit confirmation."
        )


if __name__ == "__main__":
    main()
