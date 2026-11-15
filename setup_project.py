# setup_project.py
"""Create the youtube-reviser project structure in one go.

Works on Windows, macOS, and Linux:  python setup_project.py
Idempotent: existing files are never overwritten.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "transcripts/dsa",
    "transcripts/system_design",
    "chunks",
    "db",
    "logs",
    "scripts",
    "tests",
]

# Folders that should be tracked by git even when empty.
GITKEEP_FOLDERS = [
    "transcripts/dsa",
    "transcripts/system_design",
    "chunks",
    "logs",
]

# Files that get created only if they do not already exist.
FILES = {
    "scripts/download_transcripts.py": (
        "# Download raw caption JSON per video (DSA / system design) into transcripts/.\n"
    ),
    "scripts/chunk_transcripts.py": (
        "# Split transcripts into overlapping chunks and write chunks/*.jsonl files.\n"
    ),
    "scripts/embed_and_store.py": (
        "# Embed chunks with sentence-transformers and store them in the Chroma DB (db/).\n"
    ),
    "scripts/query.py": (
        "# Retrieve relevant chunks from Chroma and answer placement-prep questions.\n"
    ),
    "tests/test_chunking.py": (
        "# Tests for the transcript chunking logic in scripts/chunk_transcripts.py.\n"
    ),
    "glossary.json": "{}\n",
    "requirements.txt": (
        "yt-dlp\n"
        "youtube-transcript-api\n"
        "sentence-transformers\n"
        "chromadb\n"
        "pytest\n"
    ),
}


def main() -> None:
    created = 0

    for rel in FOLDERS:
        path = ROOT / rel
        if not path.exists():
            path.mkdir(parents=True)
            print(f"[folder]  {rel}/")
            created += 1

    for rel in GITKEEP_FOLDERS:
        keep = ROOT / rel / ".gitkeep"
        if not keep.exists():
            keep.write_text("")
            print(f"[gitkeep] {rel}/.gitkeep")
            created += 1

    for rel, content in FILES.items():
        path = ROOT / rel
        if path.exists():
            print(f"[exists]  {rel}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            print(f"[file]    {rel}")
            created += 1

    print(f"\nDone. {created} item(s) created.")


if __name__ == "__main__":
    main()
