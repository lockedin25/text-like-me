# Verification for the initial public release

Verified locally before publication:

- 19 offline tests: export formats and accounting, preserved multilingual records, real burst boundaries, redaction/API boundary, strict dated memory and owner separation, masked targets and stop markers, sign-off tamper detection, template drift, gap rendering, output checks, split boundaries, train-only synthetic style examples, overlapping-window coverage and unseen-media masking, error propagation and distinct historical summaries.
- A complete fictional-data offline demo from parsing to dataset sign-off, with no API/GPU/model inference.
- Actual cached Gemma 4 tokenizer: the fictional build passes the template-prefix assertion and all six variant/split preflights. This check uses the real tokenizer, not the demo's character tokenizer.
- CPU PyTorch 2.8.0: shifted token loss, padding masks, weighted microbatch/batch equivalence, finite backward gradients and deterministic epoch grouping. `tests/test_training.py` reproduces this optional check.
- Ruff checks and Python compilation; shell syntax for the pilot runner.
- Public-file scan plus manual review for private identities, source paths, chat exports, credentials and model artifacts.

Not claimed: a fresh 12B CUDA training run, hosted API calls with real private data, real model-quality improvements from this generalized extraction, or a completed DPO/online-RL experiment. The trainer is adapted from the executed private workflow; the public pipeline still needs a two-step GPU smoke test and human validation on each new user's data.

GitHub Actions runs the offline tests, demo, source audit and lint on Linux/Python 3.12 without secrets. Optional torch numerical tests skip in the minimal environment. On a machine with torch installed, run `uv run --extra dev --with torch==2.8.0 pytest tests/test_training.py -q`.
