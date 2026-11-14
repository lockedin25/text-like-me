"""Explicit stages keep paid requests and GPU work out of the default local path."""

import argparse
import json
from pathlib import Path

from .common import load_config, read_jsonl, tokenizer, write_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    commands = ap.add_subparsers(dest="command", required=True)
    for name in ("ingest", "build", "review-page", "preflight"):
        commands.add_parser(name)
    p = commands.add_parser("signoff")
    p.add_argument("--i-reviewed", action="store_true", required=True)
    p = commands.add_parser("api")
    p.add_argument("task", choices=["bridge", "memory", "targeted", "review", "judge"])
    p.add_argument("--pilot", action="store_true")
    p = commands.add_parser("freeze")
    p.add_argument("task")
    p.add_argument("--i-reviewed", action="store_true", required=True)
    p = commands.add_parser("apply")
    p.add_argument("task", choices=["bridge", "memory", "targeted"])
    p.add_argument("--approval", required=True)
    p = commands.add_parser("chat")
    p.add_argument("--adapter", required=True)
    p.add_argument("--backend", choices=["mlx", "hf"], default="mlx")
    p = commands.add_parser("evaluate")
    p.add_argument("--adapter", required=True)
    p.add_argument("--backend", choices=["mlx", "hf"], default="hf")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--label", default="candidate")
    p = commands.add_parser("blind")
    p.add_argument("replays", nargs="+")
    p = commands.add_parser("export-preferences")
    p.add_argument("--out", required=True)
    p = commands.add_parser("add-memory")
    p.add_argument("--file", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from . import pipeline

    if args.command == "ingest":
        result = pipeline.ingest(cfg)
    elif args.command == "build":
        import re

        if not re.fullmatch(r"[0-9a-f]{40}", cfg.get("model_revision", "")):
            raise ValueError(
                "Set model_revision to an immutable 40-character Hugging Face commit before a real build"
            )
        result = pipeline.build(cfg, tokenizer(cfg))
    elif args.command == "review-page":
        result = pipeline.review_page(cfg)
    elif args.command == "preflight":
        result = pipeline.preflight(cfg)
    elif args.command == "signoff":
        result = pipeline.signoff(cfg)
    elif args.command == "api":
        from .tasks import run_task

        _, result = run_task(cfg, args.task, args.pilot)
    elif args.command == "freeze":
        from .tasks import freeze

        result = freeze(cfg, args.task)
    elif args.command == "apply":
        from .tasks import apply_suggestions

        result = apply_suggestions(cfg, args.task, args.approval)
    elif args.command == "chat":
        from .app import chat

        chat(cfg, args.adapter, args.backend)
        return
    elif args.command == "evaluate":
        from .evaluate import evaluate

        result = evaluate(cfg, args.adapter, args.backend, args.limit, args.label)
    elif args.command == "blind":
        from .evaluate import blind

        result = blind(cfg, args.replays)
    elif args.command == "export-preferences":
        rows = read_jsonl(Path(cfg["work_dir"]) / "corrections.jsonl")
        write_jsonl(
            args.out,
            [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows],
        )
        result = {"exported_pairs": len(rows), "trained": False}
    elif args.command == "add-memory":
        from .memory import Memory

        mem = Memory(Path(cfg["work_dir"]) / "memory.sqlite3")
        rows = read_jsonl(args.file)
        for row in rows:
            mem.put(row)
        mem.close()
        result = {"stored_records": len(rows)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
