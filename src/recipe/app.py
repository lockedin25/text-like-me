"""Local-only text app: same memory/rendering, checked burst output, correction collection."""

import re
import time
from datetime import datetime
from pathlib import Path

from .common import read_jsonl, write_jsonl
from .memory import Memory, render


def inspect_reply(text, previous=()):
    issues = []
    low = text.lower()
    if any(
        s in low
        for s in (
            "<|channel>",
            "<|turn>",
            "<turn|>",
            "as an ai",
            "language model",
            "system prompt",
        )
    ):
        issues.append("identity_or_control_leak")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        issues.append("empty")
    if len(lines) != len(set(x.lower() for x in lines)):
        issues.append("repeated_lines")
    if any(text.strip() == p.strip() for p in previous):
        issues.append("exact_repeat")
    if re.search(r"\b(here to help|how can i assist)\b", low):
        issues.append("assistant_style")
    return issues


def backend(cfg, adapter, kind):
    if kind == "mlx":
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        model, tok = load(cfg["mlx_model"], adapter_path=adapter)

        def generate_reply(messages):
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return generate(
                model,
                tok,
                prompt=prompt,
                max_tokens=256,
                sampler=make_sampler(temp=0.6, top_p=0.9, min_p=0.05),
                verbose=False,
            )
    else:
        import torch
        import transformers
        from peft import PeftModel

        tok = transformers.AutoTokenizer.from_pretrained(
            cfg["model"], revision=cfg["model_revision"]
        )
        config = transformers.AutoConfig.from_pretrained(
            cfg["model"], revision=cfg["model_revision"]
        )
        model = getattr(transformers, config.architectures[0]).from_pretrained(
            cfg["model"],
            revision=cfg["model_revision"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, adapter).eval()

        def generate_reply(messages):
            ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9
                )
            return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    return tok, generate_reply


def chat(cfg, adapter, kind="mlx"):
    work = Path(cfg["work_dir"])
    memory = Memory(work / "memory.sqlite3")
    tok, generate = backend(cfg, adapter, kind)
    history = []
    last_prompt = None
    print(
        "Local chat. /reset, /correct preferred text, /remember owner|predicate|fact, /quit. Corrections do not train weights."
    )
    while True:
        text = input("you> ").strip()
        if not text:
            continue
        now = datetime.now().isoformat(timespec="seconds")
        if text == "/quit":
            break
        if text == "/reset":
            history = []
            last_prompt = None
            continue
        if text.startswith("/correct "):
            if not history or history[-1]["role"] != "assistant" or last_prompt is None:
                print("No response to correct.")
                continue
            chosen = text[len("/correct ") :]
            p = work / "corrections.jsonl"
            rows = read_jsonl(p) if p.exists() else []
            rows.append(
                {
                    "at": now,
                    "prompt": last_prompt,
                    "chosen": chosen,
                    "rejected": history[-1]["content"],
                }
            )
            write_jsonl(p, rows)
            history[-1]["content"] = chosen
            continue
        if text.startswith("/remember "):
            parts = text[len("/remember ") :].split("|", 2)
            if len(parts) != 3:
                print("Use owner|predicate|fact; owner is self, other, or shared.")
                continue
            owner, predicate, fact = parts
            memory.put(
                {
                    "id": "correction-" + now,
                    "owner": owner,
                    "kind": "correction",
                    "predicate": predicate,
                    "text": fact,
                    "observed_at": now,
                    "status": "reviewed",
                    "source": "manual",
                }
            )
            continue
        history.append({"role": "user", "content": text, "at": now})
        prompt, _, _ = render(cfg, tok, history, memory, now)
        last_prompt = prompt
        previous = [m["content"] for m in history if m["role"] == "assistant"]
        reply = ""
        for _ in range(3):
            candidate = generate(prompt)
            if not inspect_reply(candidate, previous):
                reply = candidate
                break
        if not reply:
            print("[Reply failed output checks; use /reset or try again.]")
            continue
        for line in reply.splitlines():
            if line.strip():
                print("model> " + line)
                time.sleep(0.35)
        history.append(
            {
                "role": "assistant",
                "content": reply,
                "at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_jsonl(work / "session.jsonl", history)
    memory.close()
