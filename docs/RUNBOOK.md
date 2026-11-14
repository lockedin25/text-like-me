# The 16-step recipe

Run commands from the repository root. Keep config, exports and work outside it. The public examples are entirely fictional.

## 1. Define the task and permission boundary

Decide whose replies the model should learn. This recipe assigns the target to `self` and the correspondent to `other`. Record whether the conversation may be used locally, uploaded for training, or processed by hosted models. Those are separate decisions. An API setting is not proof that another participant agreed.

Define what good means with examples: warmth, directness, language, burst shape, topic continuity, factual grounding and appropriate corrections. There is no single correct answer to most personal messages.

## 2. Export and configure

Export the conversation as a text file. Copy `examples/config.toml` to a private location. Set exact export sender strings, display aliases, path, DMY/MDY order and chronological boundaries. Pick dates that leave substantial train, validation and test periods. Do not split randomly across adjacent conversations.

Put optional `redactions.json` in `work_dir` to replace names, employers, places and other identifiers using exact substitutions. See the fictional example. Review it yourself: a phone-number regex does not anonymize a person's story.

## 3. Parse and inspect accounting

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml ingest
```

This writes private `stream.jsonl`, `bursts.jsonl`, `memory.sqlite3` and `ingest-report.json`. System notices are counted separately. Unknown senders, unsupported lines and out-of-order dates fail the build. Multiline messages stay one message; the edited marker is removed after continuations are joined. Reserved model-control tokens in source chats are escaped.

Compare message counts with the export. Inspect representative messages locally before using any generated output.

## 4. Label stickers and handle unseen media

`work_dir/sticker-labels.json` maps the **exact first line of an exported sticker message** to its human meaning. Label what the sticker means in that relationship, not merely what it looks like. Re-run ingest after adding labels. The parser never opens media bytes.

Unseen photos, voice notes, documents, video and location records remain visible as neutral context. Their own targets and immediate media-dependent replies are conservatively masked. This can discard more targets than the original manually reviewed pipeline. The tradeoff is explicit: preserve conversation continuity without rewarding an answer to unseen content.

Optionally ask GLM for text-only photo bridges using step 6. Bridges remain context-only. The public implementation does not use a bridge to certify that an immediate response is grounded. Captions remain context. No visual descriptions are sent to a hosted model.

## 5. Build dated memory

The export archive is indexed in SQLite. Add locally reviewed facts with owner, predicate, source, observed date and availability date:

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml add-memory --file ~/my-reviewed-memory.jsonl
```

Use the schema in `examples/memory.jsonl`. `self`, `other` and `shared` are distinct. For each owner/predicate, the latest available fact supersedes the earlier one at retrieval time. A summary becomes available only after its source period ended. Historical facts must never be assigned a date earlier than their actual source. Current state is reviewed separately from historical summaries.

## 6. Pilot hosted prompts before scaling

Hosted work is disabled by default. Set the endpoint, exact provider model IDs and `api_enabled=true`. Export the API key without committing it. Set request/output/call caps in your private config. A run may stop at the call cap; rerun resumes completed jobs using payload and prompt hashes.

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml api bridge --pilot
uv run chat-recipe --config ~/chat-recipe-config.toml api memory --pilot
uv run chat-recipe --config ~/chat-recipe-config.toml api targeted --pilot
```

Open the private outputs under `work_dir/api/`. Check at least 25 items when available. Validate bridge grounding, speaker ownership, dates, tone and language. The prompt/schema live in `prompts/` and `recipe/api.py`. Change the prompt if needed, then repeat its pilot. No private response bodies are printed to the terminal.

Optional V2-style synthetic conversations can be added through the same approved-example schema (`kind: synthetic`). They need coherent context, representative burst lengths and separate validation. Generating more polished text is not evidence that it preserves your voice.

## 7. Freeze and apply reviewed suggestions

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml freeze bridge --i-reviewed
uv run chat-recipe --config ~/chat-recipe-config.toml api bridge
```

Repeat for memory/targeted. Freezing binds approval to the prompt hash. It does not approve new output automatically. Copy `examples/approval.json` outside the repository, fill its `output_sha256` from the actual local API output and select approved item IDs. Record `human_reviewed=true` only after inspection.

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml apply bridge --approval ~/bridge-approval.json
```

`apply memory` installs reviewed proposals with availability after their source block. `apply targeted` writes approved examples. Review a draft before approving an item: approving a targeted batch accepts all examples in that item. API reviews remain evidence; repair source data locally and rebuild.

## 8. Build windows and masks

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml build
```

The builder creates `everything/` and `english/` variants, chronological splits, overlapping windows and per-line weights. It supervises each eligible real turn once; trimmed windows get smaller fallback windows. Oversize targets are reported, not silently truncated. Time-gap and media stand-ins have zero loss. Mixed-language targets remain intact in the main variant. The English fallback keeps other languages as context and corrects the burst-size weighting of its remaining targets.

Recent training turns receive weight 1; older ones receive 0.6. Approved targeted/capability examples are train-only. Keep their weight and share small enough that they do not overwhelm natural replies. Capability weight defaults to 0.05 per token; this is **not** a guarantee of exactly 5% of the total signal.

## 9. Automated dataset review

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml api review --pilot
uv run chat-recipe --config ~/chat-recipe-config.toml freeze review --i-reviewed
uv run chat-recipe --config ~/chat-recipe-config.toml api review
```

DeepSeek flags broken context, unsupported facts, ownership, language, masks and unnatural style. It does not silently fix the dataset. Review results locally. For a second independent pass, switch `reviewer_model` to your other configured model and repeat; caching includes model identity.

## 10. Human review and hash-bound sign-off

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml review-page
uv run chat-recipe --config ~/chat-recipe-config.toml preflight
# Read review.html, API results, counts and the burst histogram first.
uv run chat-recipe --config ~/chat-recipe-config.toml signoff --i-reviewed
```

Inspect masks, language, cold replies, corrections, media, split dates and retrieved facts. `signoff.json` binds approval to dataset bytes and build configuration. Any rebuild requires fresh approval. The trainer re-encodes **every** row and compares token IDs, labels and weights before loading the model.

## 11. Prepare compute and a two-step smoke test

Follow [COMPUTE.md](COMPUTE.md). Install the tested stack, download the chosen immutable model revision, copy signed data, and run `--memory-check`. Test eager and SDPA only if their forward/backward results agree on a tiny probe. Check numerical finiteness and speed. A locally fitting four-bit checkpoint does not imply that BF16 training fits the same memory.

## 12. Run four one-epoch pilots

`scripts/pilots.sh` runs the original grid: mixed data at 5e-5/batch16, mixed at 1e-4/batch16, mixed at selected LR/batch8, English at selected LR/batch16. Schedules span two epochs so a selected pilot can continue without resetting the optimizer or schedule. Inspect p1/p2 before choosing the later LR; the helper's default records the historical selected LR, not a recommendation for every corpus.

Evaluate burst shape, coherence, language following and factual errors in addition to loss. Pause the VM while doing long human reviews.

## 13. Compare blind, then continue the winner

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml evaluate --backend hf --adapter ~/chat-recipe-work/runs/p2/best --label p2
uv run chat-recipe --config ~/chat-recipe-config.toml blind ~/chat-recipe-work/evaluation/p1/replays.jsonl ~/chat-recipe-work/evaluation/p2/replays.jsonl
```

Generate all compared replay files first. The HTML randomizes option labels and saves choices; the private key is separate. Current evaluator is teacher-context replay, not a free-running multi-turn benchmark. Also test fresh sessions and extended conversations manually, including corrections and factual uncertainty.

Continue the chosen pilot with the exact original data/config using `--resume --epochs 1`. If you change weights, data, LR or effective batch, start a fresh run. Do not report a blind win rate until the human actually completes the comparison.

## 14. Convert and run locally

```bash
uv sync --extra mac
uv run python scripts/convert_mlx.py ~/downloaded-peft-best ~/converted-mlx-adapter
uv run chat-recipe --config ~/chat-recipe-config.toml chat --adapter ~/converted-mlx-adapter --backend mlx
```

The checked converter is Gemma-4-specific and transposes PEFT LoRA matrices into the MLX layout. Compare held-out behaviour/loss after conversion. Quantized inference and BF16 evaluation can differ. HF inference is also available on appropriate hardware via `--backend hf`.

## 15. Record corrections and update facts

`/correct your preferred reply` records the prompt, chosen reply and rejected reply and replaces the rejected reply in active history. `/remember self|job|new fact` stores a dated correction. `/reset` starts a new conversation. These operations do **not** change weights.

```bash
uv run chat-recipe --config ~/chat-recipe-config.toml export-preferences --out ~/private-preferences.jsonl
```

Audit preference pairs and held-out regressions before a separate DPO experiment. A validated DPO trainer and online RL are future work, not features of this release.

## 16. Archive, stop compute, repeat

Download the adapter, metrics, model/config revisions and review choices. Verify checksums. Stop/destroy the training VM and remove its disks according to your provider's controls. Keep a private decision log. Revisit memory, context layout, data preparation and decoding before assuming every failure requires another training run.
