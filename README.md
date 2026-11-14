# V4 · A recipe for a model that texts like you

Export a conversation, preserve how you actually text, build dated memory, review the training data, run small LoRA pilots, compare them, and collect corrections from local use.

This repository packages the workflow from my V3 experiment. **V4 is the public recipe's name, not a claim that a fourth personal model has been trained.** It contains code, configuration templates, fictional fixtures, prompts and the reasoning behind the process. It contains no real chat exports, personal memories, images, adapters, credentials or private experiment logs.

[The essay](https://sskorde.com/writing/fine-tuning-again/) · [Project](https://sskorde.com/projects/personal-chat-v4/) · [16-step runbook](docs/RUNBOOK.md) · [Failure modes](docs/FAILURE_MODES.md)

## Start without a GPU or an API key

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/) are enough for the fictional offline demo:

```bash
git clone https://github.com/lockedin25/V4.git
cd V4
uv sync --extra dev
uv run python scripts/demo.py
uv run pytest -q
```

The demo exercises parsing, real message boundaries, burst grouping, chronological splits, memory retrieval, token masks, review-page generation and hash-bound sign-off. Its character tokenizer tests the plumbing; it does **not** test Gemma, measure model quality or train a model.

## Use your own export

Export a **two-person WhatsApp conversation as text**. Android and iOS formats with numeric DMY or MDY dates are supported; inspect the parse report before proceeding. Get the other person's agreement before using a private conversation or sending its text to a hosted model.

```bash
cp examples/config.toml ~/chat-recipe-config.toml
# Edit export, exact sender names, split dates and work_dir in that private file.
# Use an immutable model commit instead of "main" for a reproducible real run.
uv sync --extra prepare --extra dev
uv run chat-recipe --config ~/chat-recipe-config.toml ingest
uv run chat-recipe --config ~/chat-recipe-config.toml build
uv run chat-recipe --config ~/chat-recipe-config.toml review-page
uv run chat-recipe --config ~/chat-recipe-config.toml preflight
# Open ~/chat-recipe-work/review.html locally, inspect it, then:
uv run chat-recipe --config ~/chat-recipe-config.toml signoff --i-reviewed
```

Preparation downloads the tokenizer, not the full weights. Actual training requires the model weights and a suitable GPU. Follow the runbook for media, memory, hosted-model review, pilots, evaluation, conversion and local chat before treating this short path as a complete experiment.

## What is in the recipe

| Stage | What it does | Why |
|---|---|---|
| Parse and normalize | Explicit sender mapping, multiline messages, edited markers, PII substitutions | Speaker mistakes and fake message boundaries corrupt training |
| Bursts | Group same-sender messages within three minutes before any language selection | Learn double texting from real boundaries, not forced verbosity |
| Media | Human sticker labels, text-only bridge proposals, conservative masks for unseen media | A reply should not be trained against information the model cannot see |
| Memory | SQLite FTS, owner labels, observed/available dates, reviewed facts and corrections | Fine-tuning does not make weights a reliable personal database |
| Windows | Up to 40 turns, visible time gaps, once-only real reply supervision | Keep context without repeatedly training the same answer |
| Mixed-language/English pilots | Preserve all languages as context; mask selected targets for the fallback | Test your actual language mix rather than guessing |
| Hosted preparation | GLM writer and DeepSeek reviewer through a configurable compatible API | Draft bridges/memory/targeted data, then inspect them |
| Review gates | Frozen prompt hash, bounded pilot, local review, dataset sign-off | Catch data failures before paying for training |
| Training | BF16 rank-16 LoRA, weighted token loss, exact re-encoding, resumable schedule | Preserve inference alignment and make pilot continuation meaningful |
| Evaluation | Held-out replays, burst/language checks, blinded human picks | Validation loss cannot decide whether a reply sounds like you |
| Local application | Shared prompt/memory rendering, output checks, burst delivery | The wrapper changes the conversation too |
| Feedback | `/correct`, `/remember`, preference-pair export | Collect evidence for later learning; no automatic weight updates |

## Hosted models and GPUs are optional, explicit steps

Set `api_enabled = true` in your private config only after reviewing what will leave your machine. Set `RECIPE_API_KEY` in your shell. The configured endpoint can be Jarvis Token Factory or OpenRouter; model identifiers are provider-specific. The defaults record the original GLM/DeepSeek choices, not a guarantee of ongoing provider availability. No key is bundled or requested by GitHub Actions.

The repository does not provision or spend on a GPU automatically. You choose a provider, create a VM, restrict its SSH access, upload only signed artifacts, smoke-test memory/speed, and stop/delete it when finished. The same trainer works on a suitable rented CUDA VM regardless of provider. See [compute instructions](docs/COMPUTE.md).

## Important boundaries

- This is a **generalized extraction**, not a byte-for-byte copy of the private application. [Parity and differences](docs/PARITY.md) names what is preserved, simplified and still future work.
- The original tested family is **Gemma 4 12B**. The serializer and adapter mapping are family-specific. Do not change the model name and assume another family is compatible.
- Regex redaction is a starting point. It cannot identify every person, address, secret or intimate message. Private data stays outside Git; review hosted payloads and any intended publication manually.
- No API/VM/model calls happen in the offline tests. A new end-to-end CUDA training run is **not** claimed for this public extraction. Validate on your own small pilot before a full run.
- DPO is a later training stage. This release collects/exports pairs; it does not contain a validated preference trainer or autonomous production RL loop.

Code is MIT licensed. Model weights remain subject to their own license and access terms. Do not redistribute someone else's private conversations or a personal adapter without their permission.
