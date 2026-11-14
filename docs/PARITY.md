# What this release preserves

The public package is an extraction of the workflow used for the third personal model, published as Text Like Me. It is not a claim of a newly trained checkpoint.

| Part | Public implementation |
|---|---|
| Parsing and language | Configurable Android/iOS parser, multiline repair, no mixed-language deletion. Hindi heuristic is intentionally inspectable and imperfect. |
| Bursts | Three-minute grouping; one source message stays one line; no forced minimum reply length. |
| Memory | SQLite FTS, ownership, dated availability, reviewed current facts, strict as-of retrieval shared by build/runtime. |
| Monthly cards | Generalized into reviewed dated summary records. Hosted extraction uses small chronological blocks; users can import monthly summaries with explicit availability dates. No private historical cards shipped. |
| Media | Human sticker meanings; optional text-only GLM bridges; media-dependent replies masked conservatively. It does not recreate the private segmentation review UI or automatically trust a bridge as supervision. |
| Windows | Overlapping windows, once-only real supervision, fallback for trimmed replies, per-line token masks and recency weights. |
| Distractor memory | Not randomly injected by default in this public extraction. Irrelevant retrieval can still occur. Add deliberate reviewed distractors only as a separately tested experiment. |
| Targeted data | Reviewed identity, unknown-fact, repair, negation and capability proposals; optional approved synthetic conversations via JSONL. Automatic large-scale repair mining is not included. |
| Prompt gates | Bounded hosted pilot, prompt hash approval, per-item application approval, schema checks, resumable cache and dataset sign-off. |
| Trainer | Adapted executed BF16 LoRA trainer, inference-aligned token masks, all-row verification, weighted accumulation, numerical checks, best/last and resumable two-epoch schedule. |
| Pilot grid | Four one-epoch comparisons with a two-epoch horizon; human selection required. |
| Evaluation | Teacher-context held-out replay, structural/language metrics and blind picks. Extended fresh-session/free-running tests remain human work. No universal personal-authenticity score. |
| Local chat | HF or MLX backend, shared rendering/memory, output checks, multi-line burst display, corrections/facts. Terminal input is one submitted message; the original app's four-second input batching and richer web UI are not reproduced. |
| Runtime guardrails | Generic leakage/repeat/style checks and retries. No private vocabulary, biography rules, nicknames or hand-written personal fallback lines are bundled. |
| RL/DPO | Preference collection/export only. No DPO run or autonomous online trainer is claimed. |
| Compute lifecycle | Provider-independent manual runbook, not an automated VM orchestrator. |

## How to interpret evidence

The original workflow was trained and used. This extraction is verified with fictional offline tests; it has not been put through another paid 12B CUDA run for publication. Test actual tokenizer alignment and a two-step GPU smoke run before using private data at scale. A pass of plumbing tests is not a quality benchmark.

The original observation was that the larger model was more coherent and that Hindi/Hinglish made conversation warmer while remaining weaker than English. Data, context and model size changed together, so the experiment does not isolate a scaling law. Synthetic data improved coherence but could wash out the author's voice. These are qualitative lessons, not promised outcomes for a new reader.
