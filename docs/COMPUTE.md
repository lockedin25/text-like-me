# GPU and inference setup

The original tested CUDA environment was Python 3.12, torch 2.8.0+cu128, transformers 5.22.2, tokenizers 0.23.3 and peft 0.21.3. The recipe pins the relevant versions. CUDA wheels/driver compatibility depend on your GPU image; install the correct torch wheel from the official PyTorch instructions before using the train extra. Do not casually upgrade a model serializer mid-experiment.

The historical plan selected H100; actual training used H200 when H100 was unavailable. Eager attention without gradient checkpointing used approximately 118 GB in that measured configuration and was faster than the lower-memory checkpointed alternative. Those figures are observations, not promised capacity for another dataset. Default training here enables gradient checkpointing. Measure your own longest windows first.

## Provider-independent path

1. Create a CUDA GPU VM through Jarvis or another provider. Keep provider credentials in its CLI/environment, outside Git. This repository does not launch or bill resources.
2. Review the VM's authorized SSH keys. Use a dedicated training key. Do not delete unknown keys on a shared machine; use an isolated VM you control. Recheck access after every resume.
3. Clone this public repository. Install the train extra. Download the base model at the exact revision used by preprocessing; verify model access/license requirements with Hugging Face.
4. Upload only the signed `datasets/` tree, `signoff.json` and any files needed for evaluation. Both contain sensitive data. Use SSH/SFTP/rsync, then verify hashes. Do not upload source media.
5. In a persistent terminal (`tmux`), run the smoke test below. Read its counts and memory measurements before starting pilots.
6. Pause when waiting on review; download results and destroy VM/disks at the end. Check the provider's dashboard to confirm billing has stopped.

```bash
uv sync --extra train
# Authenticate to HF only if the chosen model requires it.
# Download on the VM using the exact revision recorded in datasets/build.json:
uv run hf download google/gemma-4-12B-it --revision "$MODEL_REVISION"
uv run python -m recipe.train \
  --data "$WORK_DIR/datasets/everything" --datasets "$WORK_DIR/datasets" \
  --signoff "$WORK_DIR/signoff.json" --revision "$MODEL_REVISION" \
  --out "$WORK_DIR/runs/smoke" --memory-check
# First paid pilot, still one epoch on a two-epoch schedule:
uv run python -m recipe.train \
  --data "$WORK_DIR/datasets/everything" --datasets "$WORK_DIR/datasets" \
  --signoff "$WORK_DIR/signoff.json" --revision "$MODEL_REVISION" \
  --out "$WORK_DIR/runs/p1" --lr 5e-5 --accum 16 --epochs 1 --schedule-epochs 2
# Continue that same pilot into epoch two only after comparison:
uv run python -m recipe.train \
  --data "$WORK_DIR/datasets/everything" --datasets "$WORK_DIR/datasets" \
  --signoff "$WORK_DIR/signoff.json" --revision "$MODEL_REVISION" \
  --out "$WORK_DIR/runs/p1" --lr 5e-5 --accum 16 --epochs 1 --schedule-epochs 2 --resume
```

`scripts/pilots.sh` expects an activated environment with the package installed (`uv run bash scripts/pilots.sh` is also suitable). Run from the repository root. It stops on a failed pilot rather than continuing with a broken environment.

## Training guarantees and limits

- Rank 16, alpha 32, dropout 0.05 LoRA on Gemma language-model attention and MLP projections; BF16 base, AdamW, clipped gradients.
- Token-weighted loss normalization spans the whole accumulation group, not the average of unrelated microbatch means.
- Training/validation abort on non-finite values. Padding receives no loss.
- Best/last adapters and optimizer/RNG state are saved. Resume checks dataset hash, model/revision, batch, accumulation, schedule and LR.
- Only load your own `state.pt`: optimizer checkpoints are Python serialization, not an untrusted interchange format.
- `--memory-check` performs backward passes on longest rows and reports peak allocation. It does not certify final conversation quality.

Official references: [Transformers Gemma 4](https://huggingface.co/docs/transformers/model_doc/gemma4), [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora), [PyTorch installation](https://pytorch.org/get-started/locally/), [MLX LM](https://github.com/ml-explore/mlx-lm), [OpenRouter API](https://openrouter.ai/docs/api-reference/overview).
