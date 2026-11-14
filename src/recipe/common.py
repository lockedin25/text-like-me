"""Configuration, private files and content-addressed provenance."""

import hashlib
import json
import os
import tomllib
from pathlib import Path


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["_config_path"] = str(path)
    cfg["_repo"] = str(Path(__file__).resolve().parents[2])
    cfg["work_dir"] = str(Path(cfg["work_dir"]).expanduser().resolve())
    repo = Path(__file__).resolve().parents[2]
    work = Path(cfg["work_dir"])
    if work == repo or repo in work.parents:
        raise ValueError("Keep private work_dir outside the repository")
    if cfg["date_order"] not in ("DMY", "MDY"):
        raise ValueError("date_order must be DMY or MDY")
    if cfg["self_sender"] == cfg["other_sender"]:
        raise ValueError("The two sender names must differ")
    if cfg.get("max_len", 3072) <= cfg.get("prompt_budget", 2944):
        raise ValueError("max_len must leave reply space beyond prompt_budget")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    return cfg


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)
    path.chmod(0o600)


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    path.chmod(0o600)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_hash(root):
    h = hashlib.sha256()
    root = Path(root)
    for p in sorted(root.rglob("*.jsonl")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def cfg_hash(cfg):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True
        ).encode()
    ).hexdigest()


def tokenizer(cfg):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg.get("model_revision", "main")
    )
    if not tok.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required")
    return tok
