"""Offline integration demo. Uses a character tokenizer; this is not model training."""

import json
import tempfile
from pathlib import Path

from recipe.common import load_config
from recipe.format import GEN_PREFIX, window_text
from recipe.pipeline import build, ingest, preflight, review_page, signoff


class DemoTokenizer:
    is_fast = True

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, text, **kwargs):
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def apply_chat_template(self, messages, **kwargs):
        return window_text(messages)[0] + GEN_PREFIX

    def save_pretrained(self, path):
        Path(path).mkdir(exist_ok=True, parents=True)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "examples/config.toml")
    with tempfile.TemporaryDirectory(prefix="chat-recipe-demo-") as work:
        cfg.update(
            work_dir=work,
            export=str(root / "examples/chat.txt"),
            max_len=10000,
            prompt_budget=9000,
        )
        result = {
            "ingest": ingest(cfg),
            "build": build(cfg, DemoTokenizer()),
            "review": review_page(cfg),
            "preflight": preflight(cfg),
            "signoff": signoff(cfg),
        }
        print(json.dumps(result, indent=2))
    print(
        "Offline fictional-data pipeline passed. No API, GPU or model inference was used."
    )


if __name__ == "__main__":
    main()
