"""Parse Android/iOS text exports without dropping languages or splitting real messages."""

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

# The export format is locale-dependent: unsupported date formats fail explicitly.
STAMP = re.compile(
    r"^\[?(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]?\s*(?:-\s*)?(.*)$"
)
INVISIBLE = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
CONTROL = re.compile(r"<\|[^>]*\|>|<\|[^>]*>|<[^>]*\|>|<bos>|<eos>")
EXPLICIT = re.compile(
    r"\b(sex|sext\w*|nude\w*|naked|porn\w*|penis|vagina|pussy|cock|boob\w*|cumming|orgasm\w*|masturbat\w*|horny|blowjob|erection)\b",
    re.IGNORECASE,
)
PII = [
    (r"https?://\S+", "[link]"),
    (r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[email]"),
    (r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)", "[phone or identifier]"),
    (r"\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}\b", "[secret]"),
]


def redact(text, replacements=None):
    for original, alias in sorted(
        (replacements or {}).items(), key=lambda x: -len(x[0])
    ):
        if original:
            text = re.sub(
                re.escape(original),
                lambda _, alias=alias: alias,
                text,
                flags=re.IGNORECASE,
            )
    for pattern, replacement in PII:
        text = re.sub(pattern, replacement, text)
    return CONTROL.sub("[control token]", text)


def parse_export(path, cfg):
    rows, stats, current = [], Counter(), None
    senders = {cfg["self_sender"]: "self", cfg["other_sender"]: "other"}

    def flush():
        if current is not None:
            rows.append(dict(current))

    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = INVISIBLE.sub("", raw).replace("\u202f", " ").replace("\u00a0", " ")
        match = STAMP.match(line)
        if not match:
            if current is not None:
                current["text"] += "\n" + line
                stats["continuations"] += 1
            elif line.strip():
                stats["unparsed_lines"] += 1
            continue
        flush()
        current = None
        date, clock, rest = match.groups()
        day, month, year = map(int, re.split("[/.-]", date))
        if cfg["date_order"] == "MDY":
            day, month = month, day
        year = year + 2000 if year < 100 else year
        clock = clock.strip().upper()
        fmt = (
            "%I:%M:%S %p"
            if clock.endswith(("AM", "PM")) and clock.count(":") == 2
            else "%I:%M %p"
            if clock.endswith(("AM", "PM"))
            else "%H:%M:%S"
            if clock.count(":") == 2
            else "%H:%M"
        )
        # Some exports have no space before AM/PM.
        clock = re.sub(r"(?<=\d)(AM|PM)$", r" \1", clock)
        tm = datetime.strptime(clock, fmt).time()
        ts = datetime.combine(datetime(year, month, day), tm).isoformat()
        sender = next((s for s in senders if rest.startswith(s + ": ")), None)
        if sender is None:
            # Unknown sender is not silently treated as a system notice.
            if ": " in rest:
                stats["unknown_sender_lines"] += 1
            else:
                stats["system_lines"] += 1
            continue
        current = {
            "id": f"m{len(rows):07}",
            "ts": ts,
            "sender": senders[sender],
            "text": rest[len(sender) + 2 :],
        }
    flush()
    if not rows:
        raise ValueError(
            "No messages parsed. Check export format and exact sender names."
        )
    if stats["unknown_sender_lines"] or stats["unparsed_lines"]:
        raise ValueError(
            "Unrecognized lines/senders found; inspect locally before continuing"
        )
    if any(a["ts"] > b["ts"] for a, b in zip(rows, rows[1:], strict=False)):
        raise ValueError("Export is not chronological; verify date_order")
    return rows, dict(stats)


def media_kind(text):
    low = text.lower()
    if low.strip() in ("this message was deleted", "you deleted this message", "null"):
        return "deleted"
    if "omitted>" in low or "(file attached)" in low or "<attached:" in low:
        if "stk-" in low or "sticker" in low:
            return "sticker"
        if (
            re.search(r"\.(jpg|jpeg|png|heic|gif|webp)\b", low)
            or "image omitted" in low
        ):
            return "photo"
        if re.search(r"\.(opus|mp3|m4a|aac)\b", low) or "audio omitted" in low:
            return "voice"
        if re.search(r"\.(mp4|mov|3gp)\b", low) or "video omitted" in low:
            return "video"
        return "media"
    if low.startswith("location:"):
        return "location"
    return "text"


def language(text):
    """Conservative heuristic for the fallback/audits, never a filter on the main stream."""
    if re.search("[\u0900-\u097f]", text):
        return "mixed"
    hints = {
        "nahi",
        "haan",
        "kya",
        "kaise",
        "kyun",
        "mujhe",
        "tum",
        "tumhara",
        "mera",
        "aaj",
        "kal",
        "hai",
        "hoon",
        "karna",
        "chalo",
        "yaar",
        "acha",
        "achha",
    }
    return (
        "mixed" if hints.intersection(re.findall(r"\w+", text.lower())) else "english"
    )


def normalize(rows, cfg, labels=None, replacements=None):
    result = []
    for raw in rows:
        row = dict(raw)
        text = re.sub(r"\s*<This message was edited>\s*$", "", row["text"])
        kind = media_kind(text)
        row["kind"] = kind
        row["target_ok"] = kind == "text"
        row["explicit"] = bool(EXPLICIT.search(text))
        if kind == "sticker":
            label = (labels or {}).get(text.splitlines()[0])
            text = f"[sticker: {label}]" if label else "[unseen sticker]"
            row["target_ok"] = bool(label)
        elif kind != "text":
            # A caption remains context. No media bytes or vision descriptions are loaded.
            caption = " ".join(text.splitlines()[1:]).strip()
            text = f"[unseen {kind}]" + (" " + caption if caption else "")
            row["caption"] = caption
        row["text"] = redact(" ".join(text.splitlines()), replacements)
        row["lang"] = language(row["text"])
        if row["explicit"] and not cfg.get("explicit_local_training", False):
            row["text"] = "[intimate message]"
            row["target_ok"] = False
        result.append(row)
    return result


def bursts(rows, cfg):
    grouped = []
    for row in rows:
        if (
            grouped
            and grouped[-1]["sender"] == row["sender"]
            and not any(
                grouped[-1]["start"] < boundary <= row["ts"]
                for boundary in (cfg["valid_start"], cfg["test_start"])
            )
            and (
                datetime.fromisoformat(row["ts"])
                - datetime.fromisoformat(grouped[-1]["end"])
            ).total_seconds()
            <= cfg.get("burst_seconds", 180)
        ):
            grouped[-1]["messages"].append(row)
            grouped[-1]["end"] = row["ts"]
        else:
            grouped.append(
                {
                    "id": f"b{len(grouped):07}",
                    "sender": row["sender"],
                    "start": row["ts"],
                    "end": row["ts"],
                    "messages": [row],
                }
            )
    return grouped
