"""Hosted preparation jobs and explicit application of human-reviewed suggestions."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from .api import jobs
from .common import read_jsonl, sha, write_json, write_jsonl
from .memory import Memory


def make_jobs(cfg, task):
    work = Path(cfg["work_dir"])
    stream = read_jsonl(work / "stream.jsonl")
    if task == "bridge":
        return [
            {
                "id": m["id"],
                "media": m,
                "preceding": stream[max(0, i - 5) : i],
                "following": stream[i + 1 : i + 5],
            }
            for i, m in enumerate(stream)
            if m["kind"] == "photo"
        ]
    if task == "memory":
        # Small blocks avoid whole-month truncation. Each block has a real availability bound.
        return [
            {
                "id": f"memory-{i}",
                "available_after": stream[min(i + 79, len(stream) - 1)]["ts"],
                "messages": stream[i : i + 80],
            }
            for i in range(0, len(stream), 80)
        ]
    if task == "targeted":
        return [
            {
                "id": "targeted-pilot",
                "style_examples": [
                    m
                    for m in stream
                    if m["sender"] == "self"
                    and m["kind"] == "text"
                    and m["ts"] < cfg["valid_start"]
                ][:30],
                "instructions": "Draft identity, unknown fact, repair, negation and capability examples. Use only these participant aliases: self and other. Do not invent biographical facts.",
            }
        ]
    if task == "review":
        return read_jsonl(work / "datasets/windows_render.jsonl")
    if task == "judge":
        return [
            dict(r, id=p.parent.name + "-" + r["id"])
            for p in sorted((work / "evaluation").glob("*/replays.jsonl"))
            for r in read_jsonl(p)
        ]
    raise ValueError("Unknown task")


def run_task(cfg, task, pilot=False):
    prompt = Path(__file__).resolve().parents[2] / "prompts" / f"{task}.md"
    return jobs(cfg, task, make_jobs(cfg, task), prompt, pilot)


def freeze(cfg, task):
    work = Path(cfg["work_dir"])
    prompt = Path(__file__).resolve().parents[2] / "prompts" / f"{task}.md"
    digest = sha(prompt)
    if not (work / "api" / f"{task}-{digest[:12]}.jsonl").exists():
        raise ValueError("No pilot outputs to approve")
    p = work / "prompt-approvals.json"
    obj = json.loads(p.read_text()) if p.exists() else {}
    obj[task] = digest
    write_json(p, obj)
    return {"frozen": task, "prompt_sha256": digest}


def apply_suggestions(cfg, task, path):
    """Only a local approval file can apply a bridge or memory proposal; models cannot approve themselves."""
    work = Path(cfg["work_dir"])
    approved = json.loads(Path(path).read_text())
    if approved.get("human_reviewed") is not True:
        raise ValueError("Approval must record human_reviewed=true")
    prompt = Path(__file__).resolve().parents[2] / "prompts" / f"{task}.md"
    digest = sha(prompt)
    output = work / "api" / f"{task}-{digest[:12]}.jsonl"
    if approved.get("output_sha256") != sha(output):
        raise ValueError("Approval is not bound to these exact suggestions")
    ids = set(approved["approved_ids"])
    rows = [r for r in read_jsonl(output) if r["item_id"] in ids]
    if task == "memory":
        items = {i["id"]: i for i in make_jobs(cfg, task)}
        memory = Memory(work / "memory.sqlite3")
        for row in rows:
            at = (
                datetime.fromisoformat(items[row["item_id"]]["available_after"])
                + timedelta(seconds=1)
            ).isoformat()
            for n, rec in enumerate(row["result"]["records"]):
                memory.put(
                    dict(
                        rec,
                        id=f"{row['item_id']}-{n}",
                        observed_at=at,
                        available_at=at,
                        source="reviewed-hosted-proposal",
                    )
                )
        memory.close()
    elif task == "bridge":
        proposals = {r["item_id"]: r["result"] for r in rows}
        stream = read_jsonl(work / "stream.jsonl")
        for m in stream:
            if m["id"] in proposals:
                bridge = proposals[m["id"]]["bridge"].strip()
                if bridge:
                    m["text"] = bridge
                    m["bridge"] = True
                    m["target_ok"] = False
                # Future context was used to propose a bridge: never train the immediate reply
                # to that media as though it had independently inferred unseen pixels.
        write_jsonl(work / "stream.jsonl", stream)
        grouped = read_jsonl(work / "bursts.jsonl")
        by_id = {m["id"]: m for m in stream}
        for b in grouped:
            b["messages"] = [by_id[m["id"]] for m in b["messages"]]
        write_jsonl(work / "bursts.jsonl", grouped)
    elif task == "targeted":
        examples = []
        for row in rows:
            for e in row["result"]["examples"]:
                examples.append(
                    dict(
                        e,
                        approved=True,
                        weight=0.05 if e["kind"] == "capability" else 1.0,
                    )
                )
        write_jsonl(work / "approved-examples.jsonl", examples)
    else:
        raise ValueError(
            "Reviews/judgments remain evidence; edit data locally and rebuild"
        )
    return {"applied_items": len(rows)}
