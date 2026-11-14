# Privacy boundaries

The public repository is an allowlisted code/documentation package. It is not an anonymized dump of the author's project. No historical chat, real person's profile, private screenshot, adapter, database, source export, original private plan or raw log is included. Examples were written from scratch for this release.

## Your local artifacts

`work_dir` must be outside the clone. It holds exports after normalization, hosted results, SQLite memory, rendered datasets, review pages, evaluation replies, correction pairs and checkpoints. Files are written with restrictive permissions where supported. Do not assume `.gitignore` prevents an explicit force-add or upload.

A model can memorize training text. An adapter is private data too. Do not publish one merely because the base weights are public.

## Hosted boundary

Hosted requests require explicit config enablement and an environment key. They contain text only. The code never reads source images, audio or video. Explicit-content flags and a word-list check mask known intimate messages. Automatic email/phone/link/token patterns and optional local substitutions reduce PII; they do not guarantee anonymization. Context can identify someone even without a name. Review a pilot's payload/source locally and use a provider/data-retention arrangement you and the correspondent accept.

API call caps bound jobs, not the provider's bill: each failed response can cause retries, and input/output token counts depend on the provider. Check usage/billing. No published price is hard-coded as a guarantee.

The dataset approval hash detects changed files. It is not encryption or proof that every private item was removed. Local review HTML must never be copied to a public site.

## Publication audit

Run `uv run python scripts/audit_public.py` before publishing. It examines tracked files (or source files before Git initialization), blocks data/model/database/key formats, detects common secrets, absolute home paths and known local project identifiers. Inspect filenames and documentation manually too. The release's audit reports counts and file names, never matched values.

The essay and public project page are separate, explicitly selected publication artifacts. Only the author's approved article prose, selected non-sexual model-test cards and aggregate graphics should be published. Private source screenshots, local provenance paths, review files and original Doc exports stay local.
