"""Teacher-context replays and blinded human comparisons; no fabricated authenticity score."""

import html
import random
from collections import Counter
from pathlib import Path

from .app import backend, inspect_reply
from .common import read_jsonl, write_json, write_jsonl
from .ingest import language


def evaluate(cfg, adapter, kind="hf", limit=40, label="candidate"):
    work = Path(cfg["work_dir"])
    _, generate = backend(cfg, adapter, kind)
    rows = read_jsonl(work / "datasets/windows_render.jsonl")
    results = []
    for row in rows:
        if row["split"] != "test":
            continue
        weights = row["weights"].get("everything", {})
        for index in sorted(map(int, weights)):
            prompt = row["messages"][:index]
            reference = row["messages"][index]["content"]
            reply = generate(prompt)
            results.append(
                {
                    "id": f"{row['id']}-{index}",
                    "label": label,
                    "prompt": prompt,
                    "reply": reply,
                    "reference": reference,
                    "issues": inspect_reply(reply),
                    "language": language(reply),
                    "bubbles": len([x for x in reply.splitlines() if x.strip()]),
                }
            )
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    out = work / "evaluation" / label
    write_jsonl(out / "replays.jsonl", results)
    write_json(
        out / "metrics.json",
        {
            "n": len(results),
            "bubble_histogram": dict(Counter(min(r["bubbles"], 4) for r in results)),
            "reference_bubble_histogram": dict(
                Counter(min(len(r["reference"].splitlines()), 4) for r in results)
            ),
            "issues": dict(Counter(i for r in results for i in r["issues"])),
            "mixed_language": sum(r["language"] == "mixed" for r in results),
            "mode": "teacher-context, not multi-turn free-running",
        },
    )
    return {"evaluated": len(results), "output_label": label}


def blind(cfg, paths):
    work = Path(cfg["work_dir"])
    runs = [read_jsonl(p) for p in paths]
    by = [{r["id"]: r for r in run} for run in runs]
    ids = sorted(set.intersection(*(set(x) for x in by)))
    rng = random.Random(42)
    cards = []
    key = {}
    for i, id_ in enumerate(ids):
        order = list(range(len(by)))
        rng.shuffle(order)
        key[str(i)] = {str(n): by[j][id_]["label"] for n, j in enumerate(order)}
        context = by[0][id_]["prompt"][-1]["content"]
        choices = "".join(
            f'<label><input type="radio" name="q{i}" value="{n}">Option {n + 1}<pre>{html.escape(by[j][id_]["reply"])}</pre></label>'
            for n, j in enumerate(order)
        )
        cards.append(
            f"<fieldset><legend>Case {i + 1}</legend><pre>{html.escape(context)}</pre>{choices}</fieldset>"
        )
    page = (
        """<!doctype html><meta charset="utf-8"><title>Private blind comparison</title><style>body{font:17px/1.6 system-ui;max-width:900px;margin:auto;padding:30px}pre{white-space:pre-wrap}label{display:block;padding:15px}fieldset{margin:20px 0}</style><h1>Blind comparison</h1><p>Choose the reply that sounds more like you. Also inspect factual accuracy, context, Hindi quality and burst shape. No automatic judge can decide this for you.</p><form id="picks">"""
        + "".join(cards)
        + """</form><button id="save">Save picks</button><script>document.getElementById('save').onclick=()=>{const b=new Blob([JSON.stringify(Object.fromEntries(new FormData(document.getElementById('picks'))),null,2)],{type:'application/json'});const u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='blind-picks.json';a.click();URL.revokeObjectURL(u)}</script>"""
    )
    p = work / "blind.html"
    p.write_text(page)
    p.chmod(0o600)
    write_json(work / "blind-key.json", key)
    return {"matched_cases": len(ids)}
