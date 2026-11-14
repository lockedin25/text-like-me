"""Fail closed on common publication mistakes. Never print matching private values."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY_SUFFIXES = {
    ".sqlite",
    ".sqlite3",
    ".db",
    ".safetensors",
    ".pt",
    ".pem",
    ".key",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".pdf",
    ".docx",
    ".zip",
}
PATTERNS = [
    ("absolute-home", re.compile(r"/(?:Users|home)/[^/\s]+/")),
    ("private-key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("github-token", re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}")),
    ("provider-token", re.compile(r"\bsk-[A-Za-z0-9_-]{24,}")),
]


def files():
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if proc.returncode == 0 and proc.stdout:
        return [ROOT / p for p in proc.stdout.decode().split("\0") if p]
    return [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(
            s in p.parts
            for s in (
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "dist",
            )
        )
    ]


def main():
    failures = []
    items = files()
    for p in items:
        rel = str(p.relative_to(ROOT))
        if p.is_symlink():
            failures.append((rel, "symlink"))
            continue
        if p.suffix.lower() in BINARY_SUFFIXES:
            failures.append((rel, "private/binary artifact"))
            continue
        if p.stat().st_size > 1_000_000:
            failures.append((rel, "unexpected large file"))
            continue
        text = p.read_text(encoding="utf-8")
        # The scanner necessarily contains its own denylist; exempt only its literal patterns.
        if p.resolve() == Path(__file__).resolve():
            continue
        for name, pattern in PATTERNS:
            if pattern.search(text):
                failures.append((rel, name))
        if p.suffix == ".jsonl" and not rel.startswith("examples/"):
            failures.append((rel, "non-fixture JSONL"))
    if failures:
        for path, rule in failures:
            print(f"{path}: blocked by {rule}")
        raise SystemExit(1)
    print(
        f"Publication scan passed: {len(items)} text source/config/fixture files. Manual context review still required."
    )


if __name__ == "__main__":
    main()
