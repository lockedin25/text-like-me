# Failure modes that shaped the recipe

These descriptions are generalized. No real person's messages, names or biographical details are reproduced.

| Failure | Why it matters | Change and check |
|---|---|---|
| Single cold replies dominate | Preprocessing can erase the second/third message before the model ever sees it | Build bursts before selection; compare raw/target/generated burst histograms |
| A third of a stream disappears | Dropping languages or media creates invisible context holes | Preserve records and mark context-only content; inspect accounting |
| Reply invents a pet or a meal | Coherence and factual grounding are different | Supply dated memory, include unknown-fact training and test corrections; do not claim one failure proves one cause |
| Doubles down after correction | Earlier generated text becomes its own evidence | Train reviewed repair examples; replace rejected active history with `/correct` |
| Warm words disappear | A broad language or profanity filter changes interpersonal tone | Keep natural language mix; separate sexual-content policy from ordinary swearing |
| Short isolated examples lose the topic | Model is rewarded without enough conversational history | Longer windows, gap markers and shared context budgets |
| Memory only appears at inference | The model has never learned the input format it must use | Same memory layout in training/runtime, evaluated with dated retrieval |
| Prompt changes after training | Instruction/style mismatch can overwhelm the learned behaviour | One renderer and exact template assertions |
| Thinking text leaks | Library defaults can differ | Explicit thinking-off rendering plus output checks |
| Repeated supervision encourages loops | Overlap can train a reply many times | Claim each eligible real turn once and assert uniqueness |
| First target tokens are masked | Prompt-only and full-sequence tokenization may not share a prefix | Assert prefix equality for every supervised boundary |
| Padding receives loss | A superficially low loss can reward a non-reply token | Explicit `-100` labels, zero padding weights and shifted CE tests |
| Edited markers appear in output | Cleaning each physical line misses continuation suffixes | Strip edited markers after reconstructing the message |
| Summaries copy an earlier month | A summary model can repeat plausible stale context | Human review, source/availability bounds and current-state separation |
| Adapter seems absent | Undertraining and loading errors look similar | Check adapter loading and held-out loss before increasing adapter strength |
| Validation loss selects a bland model | Average likelihood does not measure personal humour or warmth | Behaviour checks and blind human comparisons |
| A test carries yesterday's session | Old state changes the condition being measured | Fresh-session tests and explicit reset |
| Identity falls back to assistant | Instruct habits are strong | Reviewed in-character data; no private nickname list in public code |
| A multiline message becomes fake double texting | Line breaks and message boundaries are different | Collapse internal newlines, preserve real source boundaries |
| Problems discovered only after training | GPU time does not repair a bad dataset | Hosted review plus human sign-off before upload |
| Hindi/Hinglish becomes gibberish | Smaller capacity, tokenizer cost and context can interact | Mixed-language pilot, English fallback, human language ratings and sampling probes |
| Current state is clipped or old travel becomes a plan | The wrapper's ordering and budgets change which facts are visible | Pack reviewed current facts before historical summaries; measure rendered token counts |
| More messages become filler | Forcing a minimum length can create repeated acknowledgements | Preserve natural single replies; check repetition and burst distribution |
| Unseen image response looks fluent | A bridge inferred from later text can leak an answer | Treat bridges as context-only; conservatively mask media-dependent targets |
| GPU setup assumptions fail | Quantized local inference is not BF16 training | Measure longest-window memory and actual forward/backward throughput |
| Resume silently resets optimization | A second epoch becomes a different experiment | Persist optimizer/RNG/schedule and reject config drift |
| Reviewer sounds authoritative | Another LLM does not know what sounds like you | Reviewers flag; the human decides |
| Correction collection is called RL | Saving preference pairs does not optimize weights | Document the implemented collector and the unimplemented DPO stage separately |

Most failures are partly verifiable: masks, counts, dates, ownership and repeated tokens can be tested. Whether a reply feels warm or recognizably yours still requires personal judgment. The recipe keeps both kinds of evaluation visible.
