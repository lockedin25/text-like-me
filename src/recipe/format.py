"""One inference-aligned encoder for Gemma personal chat training and evaluation."""

from copy import deepcopy

TURN_END = "<turn|>"


def encode_reply(tok, messages, max_len=2048):
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("A complete assistant target is required")
    target = messages[-1]["content"].strip()
    if not target or TURN_END in target or "<|channel>" in target:
        raise ValueError("Empty target or reserved control token")
    context = deepcopy(messages[:-1])
    while True:
        prompt = tok.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prefix = tok.encode(prompt, add_special_tokens=False)
        full = tok.encode(prompt + target + TURN_END + "\n", add_special_tokens=False)
        if full[: len(prefix)] != prefix:
            raise ValueError("Tokenizer changed the prompt boundary")
        if len(full) <= max_len:
            if len(full) <= len(prefix):
                raise ValueError("No target tokens")
            return {
                "input_ids": full,
                "attention_mask": [1] * len(full),
                "labels": [-100] * len(prefix) + full[len(prefix) :],
            }
        # Drop oldest complete exchanges, never the latest user message or target.
        start = 1 if context[0]["role"] == "system" else 0
        if len(context) - start <= 1:
            raise ValueError(
                "System, latest user and complete target exceed token budget"
            )
        end = start + 1
        while end < len(context) and context[end]["role"] != "user":
            end += 1
        context = context[:start] + context[end:]


# ---------------------------------------------------------------- v3: several supervised replies per window
BOS = "<bos>"
GEN_PREFIX = "<|turn>model\n<|channel>thought\n<channel|>"  # what the app's generation prompt ends with
ROLE_TAG = {"system": "system", "user": "user", "assistant": "model"}


def window_text(messages, supervised=()):
    """Render messages exactly like the Gemma 4 template (thinking off), except that every
    supervised assistant turn starts with the generation prefix the app uses at inference.

    Returns (text, spans) where spans[i] = (content_start, content_end) character offsets.
    Consecutive assistant turns must already be merged (the template would merge them).
    """
    text, spans = BOS, {}
    for i, m in enumerate(messages):
        if (
            m["role"] == "assistant"
            and i > 0
            and messages[i - 1]["role"] == "assistant"
        ):
            raise ValueError(
                "consecutive assistant turns must be merged before encoding"
            )
        head = (
            GEN_PREFIX
            if (m["role"] == "assistant" and i in supervised)
            else f"<|turn>{ROLE_TAG[m['role']]}\n"
        )
        text += head
        start = len(text)
        text += m["content"]
        spans[i] = (start, len(text))
        text += TURN_END + "\n"
    return text, spans


def encode_window(tok, messages, line_weights, max_len=3072):
    """Encode one v3 training window with per-token loss weights.

    line_weights: {message index: [weight per content line]} for assistant turns with any
    supervision. Lines with weight 0 (photo stand-ins, gap markers, untrainable bursts) stay
    context. The end-of-turn token gets the weight of the turn's last weighted line.
    Asserts, for every supervised turn, that the tokens before its content are exactly the
    tokenization of the prefix text (no token crosses the prompt/reply boundary).
    """
    supervised = {i for i, ws in line_weights.items() if any(w > 0 for w in ws)}
    text, spans = window_text(messages, supervised)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_len:
        raise ValueError(f"window has {len(ids)} tokens > {max_len}")
    labels = [-100] * len(ids)
    weights = [0.0] * len(ids)
    char_w = {}  # (start, end) char span -> weight
    for i in supervised:
        cs, ce = spans[i]
        lines = messages[i]["content"].split("\n")
        ws = line_weights[i]
        if len(ws) != len(lines):
            raise ValueError("one weight per content line is required")
        pos = cs
        last_w = 0.0
        last_k = max(k for k, w in enumerate(ws) if w > 0)
        tail_untrained = last_k < len(lines) - 1
        for k, (line, w) in enumerate(zip(lines, ws, strict=True)):
            end = (
                pos + len(line) + (1 if k < len(lines) - 1 else 0)
            )  # a line owns its trailing newline
            if w > 0:
                # If only untrained lines follow (a merged follow-up or a photo stand-in), the last trained
                # line does not own its newline: the model must not learn "continue after this" (Gate 1, 14 Nov).
                char_w[(pos, end - 1 if (k == last_k and tail_untrained) else end)] = w
                last_w = w
            pos = end
        # "<turn|>\n": learn to stop, unless the turn really ends with lines the target participant did not write.
        char_w[(ce, ce + len(TURN_END) + 1)] = 0.0 if tail_untrained else last_w
        prefix_ids = tok(text[:cs], add_special_tokens=False)["input_ids"]
        if ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("tokenizer merged across the prompt/reply boundary")
    spans_sorted = sorted(char_w.items())
    j = 0
    for t, (a, _b) in enumerate(offsets):
        while j < len(spans_sorted) and spans_sorted[j][0][1] <= a:
            j += 1
        if j < len(spans_sorted):
            (s0, s1), w = spans_sorted[j]
            if s0 <= a < s1 and w > 0:
                labels[t] = ids[t]
                weights[t] = float(w)
    if not any(label != -100 for label in labels):
        raise ValueError("no supervised tokens")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "loss_weights": weights,
    }
