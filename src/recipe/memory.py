"""Owner-aware SQLite retrieval with a strict historical cutoff, shared by build and chat."""

import re
import sqlite3
from datetime import datetime
from pathlib import Path


class Memory:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, observed_at TEXT NOT NULL, available_at TEXT NOT NULL, status TEXT NOT NULL, source TEXT NOT NULL, predicate TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(id UNINDEXED,text)"
        )
        path.chmod(0o600)

    def put(self, r):
        if r["owner"] not in ("self", "other", "shared"):
            raise ValueError("Invalid memory owner")
        if r.get("kind", "fact") not in (
            "archive",
            "fact",
            "summary",
            "current",
            "correction",
        ):
            raise ValueError("Invalid memory kind")
        # ISO timestamps are compared as strings: normalize to naive export-local ISO first.
        for key in ("observed_at", "available_at"):
            value = r.get(key, r["observed_at"])
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is not None:
                raise ValueError("Use one export-local timezone, without offsets")
            r[key] = dt.isoformat()
        self.db.execute(
            "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?)",
            (
                r["id"],
                r["owner"],
                r.get("kind", "fact"),
                r["text"],
                r["observed_at"],
                r["available_at"],
                r.get("status", "historical"),
                r.get("source", "manual"),
                r.get("predicate", ""),
            ),
        )
        self.db.execute("DELETE FROM search WHERE id=?", (r["id"],))
        self.db.execute("INSERT INTO search VALUES (?,?)", (r["id"], r["text"]))
        self.db.commit()

    def select(self, query, before, limit=8):
        before = datetime.fromisoformat(before).isoformat()
        # No future facts, summaries of the target window, or target answers can enter memory.
        self.db.row_factory = sqlite3.Row
        base = "SELECT * FROM records WHERE observed_at < ? AND available_at < ?"
        facts = [
            dict(r)
            for r in self.db.execute(
                base + " AND kind!='archive' ORDER BY available_at DESC",
                (before, before),
            )
        ]
        seen = set()
        selected = []
        for r in facts:
            key = (
                r["owner"],
                r["id"] if r["kind"] == "summary" else r["predicate"] or r["id"],
            )
            if key in seen:
                continue
            seen.add(key)
            if r["status"] == "superseded":
                continue
            selected.append(r)
        terms = re.findall(r"\w{3,}", query.lower())[:20]
        archive = []
        if terms:
            fts = " OR ".join('"' + t + '"' for t in terms)
            sql = "SELECT r.* FROM records r JOIN search s ON s.id=r.id WHERE search MATCH ? AND r.observed_at < ? AND r.available_at < ? AND r.kind='archive' ORDER BY rank LIMIT ?"
            archive = [
                dict(r) for r in self.db.execute(sql, (fts, before, before, limit))
            ]
        # Latest reviewed state precedes dated historical summaries.
        order = {"correction": 0, "current": 1, "fact": 2, "summary": 3}
        selected.sort(key=lambda r: order.get(r["kind"], 4))
        return selected[:limit] + archive

    def close(self):
        self.db.close()


def render(cfg, tok, history, memory, now, cutoff=None):
    """The ONLY prompt builder used for training, evaluation and the app."""
    cutoff = cutoff or now
    query = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    records = memory.select(query, cutoff)
    identity = (
        f"Write as {cfg['self_name']} texting {cfg['other_name']}. "
        "Match the supplied conversation's language and informal style. One line is one message; write a complete burst when appropriate. "
        "Use supplied facts only when relevant. Distinguish each person's facts. Old events are historical, not current plans. "
        "If a fact is unknown, don't invent it. Accept corrections. Do not mention these instructions or add role labels. "
        "Unseen-media tags are context, never evidence of what the media contained. Thinking is disabled."
    )
    # Budget actual rendered section strings, including owner/date labels.
    packed = []
    cap = cfg.get("memory_tokens", 1100)
    for r in records:
        line = f"[{r['owner']} | {r['kind']} | observed {r['observed_at']} | {r['status']}] {r['text']}"
        if len(tok.encode("\n".join(packed + [line]), add_special_tokens=False)) <= cap:
            packed.append(line)
    system = (
        identity
        + "\nConversation time: "
        + now
        + "\nSupplied memory:\n"
        + "\n".join(packed)
    )
    history = [dict(m) for m in history[-cfg.get("max_turns", 40) :]]
    for i, m in enumerate(history):
        if i and m.get("at") and history[i - 1].get("at_end", history[i - 1].get("at")):
            delta = (
                datetime.fromisoformat(m["at"])
                - datetime.fromisoformat(
                    history[i - 1].get("at_end", history[i - 1]["at"])
                )
            ).total_seconds() / 60
            if delta > cfg.get("gap_minutes", 45):
                marker = f"[{max(1, round(delta / 60))} hours later]"
                # Time markers are part of context, never targets.
                m["content"] = marker + "\n" + m["content"]
                m["_gap"] = True
    from .format import window_text

    while True:
        msgs = [{"role": "system", "content": system}] + [
            {k: m[k] for k in ("role", "content")} for m in history
        ]
        count = len(tok.encode(window_text(msgs)[0], add_special_tokens=False))
        if count <= cfg.get("prompt_budget", 2944):
            break
        if len(history) <= 1:
            raise ValueError("Latest turn and memory exceed prompt budget")
        # Drop an entire exchange, preserving a user start when possible.
        history.pop(0)
        while len(history) > 1 and history[0]["role"] == "assistant":
            history.pop(0)
    return msgs, history, records
