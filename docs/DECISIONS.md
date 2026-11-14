# Decisions and checks

This is the generalized decision record, separate from the private project log.

1. **Preserve the conversation before selecting targets.** Removing a language or a media placeholder from the middle of a conversation teaches replies to invisible context. Mask unsuitable targets instead.
2. **Learn actual bursts.** More content was the goal, not a minimum word count. Real single replies remain valid. Group first and audit the natural histogram.
3. **Keep multilingual context.** Mixed-language conversation can carry warmth and intent. Compare an English fallback on the same history rather than erasing other languages.
4. **Keep source boundaries.** An internal newline is not a second text. Edited markers are parser artifacts, not something to imitate.
5. **Expose elapsed time.** A long pause can change intent, but need not delete all earlier context. Time markers are visible and masked from the target.
6. **Separate style from personal facts.** Fine-tuning can shape response habits. Explicit, dated retrieval keeps a changing biography available without assuming the weights remember it correctly.
7. **Attribute every memory.** Owner, source, observed date, availability date and predicate matter. An old trip is not a current plan; a correspondent's job is not the model author's job.
8. **Train with the runtime layout.** Same renderer, token budget, memory order and thinking setting. No silent switch from one prompt format to another.
9. **Check the first target token.** Template drift once meant the beginning of a reply received no training signal. Every row must re-encode identically before GPU training.
10. **Use per-token weights.** Zero loss for stand-ins/gaps, consistent weighting across accumulation groups, and no padding loss. Averages of microbatch losses are not interchangeable with token-weighted loss.
11. **Prevent future-memory leakage.** A training window only retrieves facts available before that window. Current state is not inserted into historical examples. The public version uses strict date bounds even for apparently stable facts.
12. **Ask hosted models for suggestions.** GLM drafts; DeepSeek reviews. Version and pilot prompts, validate JSON and inspect outputs. Hosted reviewers do not know the owner's voice well enough to approve all examples.
13. **Bind approvals to bytes.** A human-approved dataset must be the one uploaded. Changed prompt/data/config hashes require fresh inspection.
14. **Measure hardware rather than inferring capacity.** Local four-bit inference, BF16 training and checkpointed training have different memory needs. Test actual code with longest rows.
15. **Compare small pilots before extending one.** Two learning rates, a smaller batch and a language fallback expose different failures. Keep a two-epoch schedule so continuation is a continuation.
16. **Choose with behaviour and personal judgment.** Loss is useful, but cannot tell whether a reply is cold, funny, believable or recognizably yours. Blind comparisons reduce some expectation bias without making the task objectively verifiable.
17. **Distinguish implementation from proposals.** A sampling probe is not a deployed change. A correction collector is not DPO. A recipe extraction's plumbing tests are not a new 12B training result.
18. **Expect ongoing human work.** Memory, context, data, sampling and guardrails all influence the experience. A larger model improved coherence in the original experiment, but multiple things changed together. Continuous feedback is a sensible next experiment, not evidence of autonomous self-improvement.
