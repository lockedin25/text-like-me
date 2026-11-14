"""Persona v3 trainer (Phase 9). Runs ON THE GPU VM. BF16 LoRA on google/gemma-4-12B-it.

  python -m recipe.train --revision MODEL_COMMIT --data DATA/everything --datasets DATA --signoff SIGNOFF --out runs/p1 \
      --lr 5e-5 --accum 16 --epochs 1 --schedule-epochs 2
  python -m recipe.train ... --memory-check            # 2 steps on the longest rows: peak memory, speed
  python -m recipe.train ... --out runs/p1 --epochs 1 --resume     # continue the same run into epoch 2

Differences from train_persona_v2.py (critique points 8 and 9):
  * a real multi-epoch loop: groups reshuffled every epoch with a fixed seed; the cosine schedule spans
    --schedule-epochs, so a 1-epoch pilot can be resumed into epoch 2 without restarting the schedule;
  * per-token loss weights: loss = sum(weight x CE) / sum(weight) over the whole optimizer step;
  * rows are pre-encoded (input_ids, labels, loss_weights); a start-up check re-tokenizes a sample of
    windows with THIS tokenizer and refuses to train if any id differs from the local build;
  * aborts without saving on a non-finite loss or gradient; best/last adapters plus optimizer/RNG state.
Never prints chat text.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

try:  # torch only exists on the VM; the data checks below also run on the Mac
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = F = None

MODEL_ID = "google/gemma-4-12B-it"


def load_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def dataset_hash(root):
    """Same hash as recipe.common.dataset_hash (every *.jsonl under datasets/, sorted, path + bytes)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.jsonl")):
        h.update(str(p.relative_to(root)).encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def verify_rows(tok, datasets_dir, variant, rows_by_id, max_len):
    """Re-encode EVERY row (windows and targeted) with this tokenizer via encode_window and require identical
    input_ids, labels and loss_weights (Gate-1 review, 14 Nov: the variant's own weights, all rows, all fields)."""
    from recipe.format import encode_window

    checked = 0
    for r in load_rows(datasets_dir / "windows_render.jsonl"):
        row = rows_by_id.get(r["id"])
        if row is None:
            continue
        weights = r["weights"].get(variant) or {}
        enc = encode_window(
            tok, r["messages"], {int(k): v for k, v in weights.items()}, max_len=max_len
        )
        if any(enc[k] != row[k] for k in ("input_ids", "labels", "loss_weights")):
            raise SystemExit(
                "A row failed tokenizer/mask verification; refusing to train"
            )
        checked += 1
    if checked != len(rows_by_id):
        raise SystemExit(
            f"verified {checked} rows but the dataset has {len(rows_by_id)}: refusing to train"
        )
    return checked


def batches_for_epoch(n_rows, rows_per_step, epoch, seed=42):
    order = list(range(n_rows))
    random.Random(seed + epoch).shuffle(order)
    return [order[i : i + rows_per_step] for i in range(0, n_rows, rows_per_step)]


def collate(rows, device, pad_id):
    L = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    wt = torch.zeros((len(rows), L), dtype=torch.float32)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        att[i, :n] = 1
        lab[i, :n] = torch.tensor(r["labels"])
        wt[i, :n] = torch.tensor(r["loss_weights"])
    return ids.to(device), att.to(device), lab.to(device), wt.to(device)


def weighted_loss_sum(model, ids, att, lab, wt):
    """Sum of weight x token CE (predicting token t+1 from position t), and the weight sum."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1, :]
    target = lab[:, 1:]
    w = wt[:, 1:]
    mask = target != -100
    if not mask.any():
        return logits.sum() * 0.0, torch.tensor(0.0, device=ids.device)
    ce = F.cross_entropy(logits[mask].float(), target[mask], reduction="none")
    return (ce * w[mask]).sum(), w[mask].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument(
        "--revision", required=True, help="same model revision as dataset build"
    )
    ap.add_argument(
        "--data",
        required=True,
        help="dir with train.jsonl, valid.jsonl, test.jsonl",
    )
    ap.add_argument(
        "--datasets",
        required=True,
        help="the datasets/ folder (render files + both variants), for checks",
    )
    ap.add_argument(
        "--signoff", required=True, help="signoff.json: the dataset hash must match"
    )
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument(
        "--memory-check",
        action="store_true",
        help="2 steps on the longest rows, then exit",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument(
        "--epochs", type=int, default=1, help="epochs to run in this invocation"
    )
    ap.add_argument(
        "--schedule-epochs", type=int, default=2, help="horizon of the cosine schedule"
    )
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument(
        "--eval-every", type=int, default=0, help="0 = about 4 evaluations per epoch"
    )
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--no-grad-checkpointing",
        action="store_true",
        help="checkpointing is ON by default (3,072 tokens)",
    )
    a = ap.parse_args()

    import transformers
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from transformers import AutoConfig, AutoTokenizer

    torch.manual_seed(42)
    random.seed(42)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    datasets = Path(a.datasets)
    variant = data.name
    if variant not in ("everything", "english"):
        raise SystemExit("--data must be datasets/everything or datasets/english")
    signed = json.loads(Path(a.signoff).read_text())
    want = signed["dataset_sha256"]
    if (
        signed.get("build_sha256")
        != hashlib.sha256((datasets / "build.json").read_bytes()).hexdigest()
    ):
        raise SystemExit("Build configuration changed after sign-off")
    have = dataset_hash(datasets)
    if have != want:
        raise SystemExit(
            "datasets/ does not match the signed-off hash: refusing to train"
        )
    build = json.loads((datasets / "build.json").read_text())
    if build["model"] != a.model or build["revision"] != a.revision:
        raise SystemExit("Model/revision differs from dataset build")
    tok = AutoTokenizer.from_pretrained(
        a.model, revision=a.revision, local_files_only=True
    )
    train = load_rows(data / "train.jsonl")
    valid = load_rows(data / "valid.jsonl")
    for r in train + valid:
        if len(r["input_ids"]) > a.max_len:
            raise SystemExit("row longer than --max-len")
    test_rows = load_rows(data / "test.jsonl")
    checked = verify_rows(
        tok,
        datasets,
        variant,
        {r["id"]: r for r in train + valid + test_rows},
        a.max_len,
    )
    rows_per_step = a.batch * a.accum
    steps_per_epoch = math.ceil(len(train) / rows_per_step)
    total_steps = steps_per_epoch * a.schedule_epochs
    eval_every = a.eval_every or max(1, math.ceil(steps_per_epoch / 4))
    run_config = {
        "model": a.model,
        "revision": a.revision,
        "data": str(data.resolve()),
        "variant": variant,
        "dataset_sha256": have,
        "attn": a.attn,
        "gradient_checkpointing": not a.no_grad_checkpointing,
        "batch": a.batch,
        "accum": a.accum,
        "lr": a.lr,
        "schedule_epochs": a.schedule_epochs,
        "max_len": a.max_len,
    }
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(data.glob("*.jsonl"))
    }
    manifest = {
        "model": a.model,
        "args": vars(a),
        "run_config": run_config,
        "rows": {"train": len(train), "valid": len(valid)},
        "steps_per_epoch": steps_per_epoch,
        "total_schedule_steps": total_steps,
        "eval_every": eval_every,
        "rows_verified": checked,
        "data_hashes": hashes,
        "torch": torch.__version__,
    }
    try:
        import peft
        import transformers

        manifest["transformers"], manifest["peft"] = (
            transformers.__version__,
            peft.__version__,
        )
    except Exception:
        pass
    (out / ("manifest.resume.json" if a.resume else "manifest.json")).write_text(
        json.dumps(manifest, indent=1)
    )
    print(
        json.dumps(
            {
                "rows": manifest["rows"],
                "steps_per_epoch": steps_per_epoch,
                "rows_verified": checked,
                "eval_every": eval_every,
            }
        ),
        flush=True,
    )

    config = AutoConfig.from_pretrained(
        a.model, revision=a.revision, local_files_only=True
    )
    architecture = config.architectures[0]
    if architecture not in (
        "Gemma4UnifiedForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
    ):
        raise SystemExit(
            "This verified trainer supports Gemma 4; adapt and re-test other model families"
        )
    model = getattr(transformers, architecture).from_pretrained(
        a.model,
        revision=a.revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=a.attn,
    )
    model.config.use_cache = False
    if not a.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        ),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    step, best, start_epoch = 0, float("inf"), 0
    if a.resume:
        st = torch.load(out / "state.pt", weights_only=False, map_location="cpu")
        if st.get("run_config") != run_config:
            raise SystemExit(
                "resume refused: data, batch, accum, lr, schedule or max-len differ from the saved run"
            )
        set_peft_model_state_dict(model, st["adapter"])
        opt.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        step, best, start_epoch = st["step"], st["best"], st["epoch"]
        mpath = out / "metrics.jsonl"
        if (
            mpath.exists()
        ):  # steps after the last checkpoint were lost: drop their log lines
            kept = [
                line
                for line in mpath.read_text().splitlines()
                if json.loads(line).get("step", 0) <= step
            ]
            mpath.write_text("".join(x + "\n" for x in kept))

    def lr_at(s):
        warm = min(10, max(1, total_steps // 10))
        return (
            a.lr
            * min(1.0, (s + 1) / warm)
            * (
                0.1
                + 0.9 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)) / 2
            )
        )

    def evaluate():
        model.eval()
        tot = wsum = 0.0
        with torch.no_grad():
            for i in range(0, len(valid), a.batch):
                ids, att, lab, wt = collate(valid[i : i + a.batch], device, pad)
                ls, ws = weighted_loss_sum(model, ids, att, lab, wt)
                if not torch.isfinite(ls):
                    raise RuntimeError("non-finite validation loss")
                tot += float(ls)
                wsum += float(ws)
        model.train()
        return tot / max(wsum, 1e-9)

    def save_adapter(name):
        """Write to a temporary folder, then swap it in, so a crash never leaves a half-written adapter."""
        import shutil

        tmp, final, old = out / (name + ".tmp"), out / name, out / (name + ".old")
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        if final.exists():
            final.rename(old)
        tmp.rename(final)
        shutil.rmtree(old, ignore_errors=True)

    def save(epoch):
        save_adapter("last")
        st = {
            "adapter": get_peft_model_state_dict(model),
            "optimizer": opt.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "step": step,
            "best": best,
            "epoch": epoch,
            "run_config": run_config,
        }
        torch.save(st, out / "state.tmp")
        (out / "state.tmp").replace(out / "state.pt")

    model.train()
    t0 = time.time()
    if a.memory_check:
        longest = sorted(range(len(train)), key=lambda j: -len(train[j]["input_ids"]))[
            : rows_per_step * 2
        ]
        for k in range(0, len(longest), rows_per_step):
            g = longest[k : k + rows_per_step]
            group_w = sum(sum(train[j]["loss_weights"]) for j in g) or 1.0
            opt.zero_grad(set_to_none=True)
            for m in range(0, len(g), a.batch):
                ids, att, lab, wt = collate(
                    [train[j] for j in g[m : m + a.batch]], device, pad
                )
                ls, _ = weighted_loss_sum(model, ids, att, lab, wt)
                (ls / group_w).backward()
        print(
            json.dumps(
                {
                    "memory_check": "ok",
                    "longest_tokens": len(train[longest[0]]["input_ids"]),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                    "seconds_per_step": round((time.time() - t0) / 2, 1),
                }
            ),
            flush=True,
        )
        return
    for epoch in range(start_epoch, start_epoch + a.epochs):
        groups = batches_for_epoch(len(train), rows_per_step, epoch)
        done_in_epoch = step - epoch * steps_per_epoch
        for g in groups[max(0, done_in_epoch) :]:
            group_w = sum(sum(train[j]["loss_weights"]) for j in g) or 1.0
            group_tokens = sum(
                sum(1 for x in train[j]["labels"] if x != -100) for j in g
            )
            opt.zero_grad(set_to_none=True)
            loss_report = 0.0
            for k in range(0, len(g), a.batch):
                ids, att, lab, wt = collate(
                    [train[j] for j in g[k : k + a.batch]], device, pad
                )
                ls, _ = weighted_loss_sum(model, ids, att, lab, wt)
                if not torch.isfinite(ls):
                    raise RuntimeError(
                        "non-finite training loss: nothing saved for this step"
                    )
                (ls / group_w).backward()
                loss_report += float(ls) / group_w
            gn = torch.nn.utils.clip_grad_norm_(params, 0.3)
            if not torch.isfinite(gn):
                raise RuntimeError("non-finite gradient: nothing saved for this step")
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step)
            opt.step()
            step += 1
            rec = {
                "step": step,
                "epoch": epoch,
                "loss": round(loss_report, 4),
                "grad_norm": round(float(gn), 4),
                "group_weight": round(group_w, 2),
                "supervised_tokens": group_tokens,
                "rows": len(g),
                "lr": lr_at(step - 1),
                "seconds": round(time.time() - t0, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            }
            epoch_done = step == (epoch + 1) * steps_per_epoch
            if step % eval_every == 0 or epoch_done:
                val = evaluate()
                rec["validation"] = round(val, 4)
                if val < best:
                    best = val
                    save_adapter("best")
                save(epoch + 1 if epoch_done else epoch)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if a.max_steps and step >= a.max_steps:
                save(
                    epoch + 1 if epoch_done else epoch
                )  # never regress the epoch at a boundary
                print("stopped at --max-steps", flush=True)
                return
    print(
        json.dumps(
            {
                "finished_step": step,
                "best_validation": best,
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
