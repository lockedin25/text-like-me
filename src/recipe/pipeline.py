"""Data build: bursts first, temporal splits, shared prompts, loss only on intended replies."""

import html
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from .common import cfg_hash, dataset_hash, read_jsonl, sha, write_json, write_jsonl
from .format import GEN_PREFIX, encode_window, window_text
from .ingest import bursts, normalize, parse_export
from .memory import Memory, render


def ingest(cfg):
    work = Path(cfg["work_dir"])
    raw, stats = parse_export(Path(cfg["export"]).expanduser(), cfg)

    def optional(name):
        p = work / name
        return json.loads(p.read_text()) if p.exists() else {}

    stream = normalize(
        raw, cfg, optional("sticker-labels.json"), optional("redactions.json")
    )
    grouped = bursts(stream, cfg)
    write_jsonl(work / "stream.jsonl", stream)
    write_jsonl(work / "bursts.jsonl", grouped)
    mem = Memory(work / "memory.sqlite3")
    for row in stream:
        if row["kind"] == "text":
            mem.put(
                {
                    "id": row["id"],
                    "owner": row["sender"],
                    "kind": "archive",
                    "text": row["text"],
                    "observed_at": row["ts"],
                    "source": "export",
                }
            )
    mem.close()
    own = [b for b in grouped if b["sender"] == "self"]
    hist = Counter(min(len(b["messages"]), 4) for b in own)
    report = {
        "messages": len(stream),
        "bursts": len(grouped),
        "self_bursts": len(own),
        "bubble_histogram": dict(hist),
        "parse": stats,
        "config_sha256": cfg_hash(cfg),
    }
    write_json(work / "ingest-report.json", report)
    return report


def split(cfg, ts):
    return (
        "train"
        if ts < cfg["valid_start"]
        else "valid"
        if ts < cfg["test_start"]
        else "test"
    )


def make_turns(cfg, grouped, variant, size_weights):
    turns = []
    for burst_index, b in enumerate(grouped):
        role = "assistant" if b["sender"] == "self" else "user"
        rows = b["messages"]
        weights = []
        previous = grouped[burst_index - 1] if burst_index > 0 else None
        unseen_previous = bool(
            previous
            and previous["sender"] == "other"
            and any(m["kind"] not in ("text", "sticker") for m in previous["messages"])
        )
        weight = 1.0 if b["start"] >= cfg["recent_start"] else 0.6
        if variant == "english":
            weight *= size_weights.get(min(len(rows), 4), 1.0)
        for m in rows:
            ok = role == "assistant" and m["target_ok"] and not unseen_previous
            if variant == "english" and m["lang"] != "english":
                ok = False
            if m.get("bridge"):
                ok = False
            if m.get("reviewed_target_ok") is not None:
                ok = ok and m["reviewed_target_ok"]
            weights.append(weight if ok else 0.0)
        turn = {
            "role": role,
            "content": "\n".join(m["text"] for m in rows),
            "at": b["start"],
            "at_end": b["end"],
            "id": b["id"],
            "weights": weights,
        }
        # Consecutive same-speaker bursts after a long silence need one model turn,
        # with a masked time-gap marker, not an illegal assistant/assistant boundary.
        if (
            turns
            and turns[-1]["role"] == role
            and split(cfg, turns[-1]["at"]) == split(cfg, turn["at"])
        ):
            old = turns[-1]
            minutes = (
                datetime.fromisoformat(turn["at"])
                - datetime.fromisoformat(old["at_end"])
            ).total_seconds() / 60
            marker = (
                f"[{max(1, round(minutes / 60))} hours later]\n"
                if minutes > cfg.get("gap_minutes", 45)
                else ""
            )
            old["content"] += "\n" + marker + turn["content"]
            old["weights"] += ([0.0] if marker else []) + weights
            old["at_end"] = turn["at_end"]
            old["id"] += "+" + turn["id"]
        else:
            turns.append(turn)
    return turns


def build(cfg, tok):
    work = Path(cfg["work_dir"])
    grouped = read_jsonl(work / "bursts.jsonl")
    mem = Memory(work / "memory.sqlite3")
    own = [
        b
        for b in grouped
        if b["sender"] == "self" and split(cfg, b["start"]) == "train"
    ]
    hist = Counter(min(len(b["messages"]), 4) for b in own)
    eng = [b for b in own if all(m["lang"] == "english" for m in b["messages"])]
    ehist = Counter(min(len(b["messages"]), 4) for b in eng)
    size_weights = {
        k: (hist[k] / max(1, len(own))) / (ehist[k] / max(1, len(eng)))
        for k in hist
        if ehist[k]
    }
    datasets = work / "datasets"
    renders = {}
    report = {}
    eligible_ids = {}
    supervised_ids = {}
    # Detect tokenizer/template drift before processing the private corpus.
    probe = [
        {"role": "system", "content": "Write short chat replies."},
        {"role": "user", "content": "hello"},
    ]
    actual = tok.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    expected = window_text(probe)[0] + GEN_PREFIX
    if actual != expected:
        raise ValueError(
            "Gemma chat template differs from the checked encoder; stop and adapt it"
        )
    for variant in ("everything", "english"):
        turns = make_turns(cfg, grouped, variant, size_weights)
        counts = Counter()
        eligible_ids[variant] = []
        supervised_ids[variant] = []
        for part in ("train", "valid", "test"):
            tt = [t for t in turns if split(cfg, t["at"]) == part]
            encoded = []
            claimed = set()
            # Overlapping 40-turn windows, stride 20: once-only claims, with smaller fallbacks.
            for hi in list(range(20, len(tt), 20)) + [len(tt)]:
                lo = max(0, hi - cfg.get("max_turns", 40))
                candidates = [
                    i
                    for i in range(lo, hi)
                    if i not in claimed and any(tt[i]["weights"])
                ]
                if not candidates:
                    continue

                def one_window(start, end, indices, tt=tt):
                    query_history = [
                        dict(t, _pos=j) for j, t in enumerate(tt[start:end], start)
                    ]
                    messages, kept, records = render(
                        cfg,
                        tok,
                        query_history,
                        mem,
                        tt[min(indices)]["at"],
                        cutoff=tt[start]["at"],
                    )
                    weights = {}
                    for n, m in enumerate(kept, 1):
                        if m["_pos"] in indices:
                            weights[n] = ([0.0] if m.get("_gap") else []) + m["weights"]
                    if not weights:
                        raise ValueError("Budget dropped every supervised reply")
                    enc = encode_window(
                        tok, messages, weights, max_len=cfg.get("max_len", 3072)
                    )
                    selected = [m["_pos"] for m in kept if m["_pos"] in indices]
                    return messages, weights, enc, selected, records

                attempts = [(lo, hi, candidates)]
                while attempts:
                    start, end, indices = attempts.pop(0)
                    indices = [i for i in indices if i not in claimed]
                    if not indices:
                        continue
                    try:
                        messages, weights, enc, selected, records = one_window(
                            start, end, indices
                        )
                    except ValueError as exc:
                        if not str(exc).startswith(
                            ("window has ", "Latest turn and memory", "Budget dropped")
                        ):
                            raise
                        if len(indices) > 1:
                            attempts = [
                                (max(0, i - 20), i + 1, [i]) for i in indices
                            ] + attempts
                            continue
                        i = indices[0]
                        try:
                            messages, weights, enc, selected, records = one_window(
                                max(0, i - 1), i + 1, [i]
                            )
                        except ValueError as exc:
                            if not str(exc).startswith(
                                (
                                    "window has ",
                                    "Latest turn and memory",
                                    "Budget dropped",
                                )
                            ):
                                raise
                            counts["oversize"] += 1
                            claimed.add(i)
                            continue
                    key = f"{part}-{tt[selected[0]]['id']}-{tt[selected[-1]]['id']}"
                    enc["id"] = key
                    encoded.append(enc)
                    claimed.update(selected)
                    for i in selected:
                        supervised_ids[variant].append(tt[i]["id"])
                    entry = renders.setdefault(
                        key,
                        {
                            "id": key,
                            "messages": messages,
                            "weights": {},
                            "split": part,
                            "source": "real",
                            "memory_ids": [r["id"] for r in records],
                        },
                    )
                    # Different variants must see exactly the same context for identical window IDs.
                    if entry["messages"] != messages:
                        raise ValueError("Variant prompt mismatch")
                    entry["weights"][variant] = {str(k): v for k, v in weights.items()}
                    remaining = [i for i in indices if i not in selected]
                    attempts.extend((max(0, i - 20), i + 1, [i]) for i in remaining)
            for t in tt:
                if any(t["weights"]):
                    eligible_ids[variant].append(t["id"])
            write_jsonl(datasets / variant / f"{part}.jsonl", encoded)
            counts[part] = len(encoded)
        report[variant] = dict(counts)
    # Optional targeted/synthetic additions are manually approved and train-only.
    extra = work / "approved-examples.jsonl"
    if extra.exists():
        for n, row in enumerate(read_jsonl(extra)):
            if row.get("approved") is not True:
                raise ValueError("An unapproved synthetic example was supplied")
            if row.get("kind") not in (
                "identity",
                "repair",
                "unknown_fact",
                "negation",
                "capability",
                "synthetic",
            ):
                raise ValueError("Unknown targeted kind")
            context = row["messages"]
            now = row.get("at", cfg["valid_start"] + "T00:00:00")
            if len(context) < 2 or context[-1]["role"] != "assistant":
                raise ValueError("Targeted example needs a final assistant reply")
            msgs, _, _ = render(cfg, tok, context[:-1], mem, now, cutoff=now)
            msgs.append(context[-1])
            wt = {
                len(msgs) - 1: [
                    row.get("weight", 0.05 if row["kind"] == "capability" else 1.0)
                ]
                * len(msgs[-1]["content"].split("\n"))
            }
            key = f"targeted-{n}"
            enc = encode_window(tok, msgs, wt, max_len=cfg.get("max_len", 3072))
            enc["id"] = key
            variants = (
                ("everything", "english")
                if row.get("english", False)
                else ("everything",)
            )
            renders[key] = {
                "id": key,
                "messages": msgs,
                "weights": {},
                "split": "train",
                "source": row["kind"],
            }
            for v in variants:
                p = datasets / v / "train.jsonl"
                write_jsonl(p, read_jsonl(p) + [enc])
                renders[key]["weights"][v] = {str(k): x for k, x in wt.items()}
    write_jsonl(datasets / "windows_render.jsonl", list(renders.values()))
    tok.save_pretrained(work / "tokenizer")
    write_json(
        datasets / "build.json",
        {
            "model": cfg["model"],
            "revision": cfg.get("model_revision", "main"),
            "config_sha256": cfg_hash(cfg),
            "tokenizer_class": type(tok).__name__,
            "report": report,
            "eligible": {k: len(v) for k, v in eligible_ids.items()},
            "supervised": {k: len(v) for k, v in supervised_ids.items()},
        },
    )
    for variant in supervised_ids:
        if len(supervised_ids[variant]) != len(set(supervised_ids[variant])):
            raise ValueError("Duplicate real supervision")
    mem.close()
    return report


def review_page(cfg):
    work = Path(cfg["work_dir"])
    rows = read_jsonl(work / "datasets/windows_render.jsonl")
    cards = []
    for r in rows:
        cards.append(
            "<details><summary>"
            + html.escape(r["id"])
            + " · "
            + r["split"]
            + "</summary><pre>"
            + html.escape(json.dumps(r, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Private dataset review</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:20px}pre{white-space:pre-wrap}details{border-top:1px solid #aaa;padding:12px}</style><h1>Private dataset review</h1><p>Review source windows, memory, weights and API review results before signing. This file contains private data. Do not publish it.</p>'
        + "".join(cards)
    )
    p = work / "review.html"
    p.write_text(page)
    p.chmod(0o600)
    return {"windows": len(rows), "review_page_created": True}


def preflight(cfg):
    work = Path(cfg["work_dir"])
    datasets = work / "datasets"
    rendered = {r["id"]: r for r in read_jsonl(datasets / "windows_render.jsonl")}
    counts = {}
    for variant in ("everything", "english"):
        seen = set()
        for part in ("train", "valid", "test"):
            rows = read_jsonl(datasets / variant / f"{part}.jsonl")
            counts[variant + "_" + part] = len(rows)
            if not rows:
                raise ValueError("Empty split: adjust dates or provide more data")
            for r in rows:
                if r["id"] in seen:
                    raise ValueError("Duplicate row across splits")
                seen.add(r["id"])
                rr = rendered[r["id"]]
                if rr["split"] != part or variant not in rr["weights"]:
                    raise ValueError("Missing render provenance")
                n = len(r["input_ids"])
                if not (
                    n == len(r["labels"]) == len(r["loss_weights"])
                    and n <= cfg.get("max_len", 3072)
                ):
                    raise ValueError("Bad token lengths")
                if not any(w > 0 for w in r["loss_weights"]):
                    raise ValueError("No supervised tokens")
                for token, label, weight in zip(
                    r["input_ids"], r["labels"], r["loss_weights"], strict=True
                ):
                    if (
                        weight < 0
                        or not math.isfinite(weight)
                        or (label != -100 and (label != token or weight <= 0))
                        or (label == -100 and weight != 0)
                    ):
                        raise ValueError("Invalid loss mask")
    report = {
        "dataset_sha256": dataset_hash(datasets),
        "build_sha256": sha(datasets / "build.json"),
        "counts": counts,
    }
    write_json(work / "preflight.json", report)
    return report


def signoff(cfg):
    work = Path(cfg["work_dir"])
    report = preflight(cfg)
    if not (work / "review.html").exists():
        raise ValueError("Generate and inspect the review page first")
    write_json(
        work / "signoff.json",
        dict(report, approved_at=datetime.now().isoformat(), human_reviewed=True),
    )
    return {"signed": True, "dataset_sha256": report["dataset_sha256"]}
