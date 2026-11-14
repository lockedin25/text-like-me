"""Convert the supported Gemma 4 PEFT LoRA into MLX; refuse an existing destination."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("destination")
a = p.parse_args()
src = Path(a.source)
dst = Path(a.destination)
if dst.exists():
    raise SystemExit("Destination already exists")
c = json.loads((src / "adapter_config.json").read_text())
sd = mx.load(str(src / "adapter_model.safetensors"))
out = {}
layers = set()
for k, v in sd.items():
    for suffix, ending in [
        (".lora_A.weight", ".lora_a"),
        (".lora_B.weight", ".lora_b"),
    ]:
        if k.endswith(suffix):
            name = k.replace(
                "base_model.model.model.language_model.", "language_model.model."
            ).replace(suffix, "")
            out[name + ending] = v.T
            layers.add(int(k.split("layers.")[1].split(".")[0]))
assert len(out) == len(sd) and all(bool(mx.all(mx.isfinite(v))) for v in out.values())
dst.mkdir(parents=True)
mx.save_safetensors(str(dst / "adapters.safetensors"), out)
(dst / "adapter_config.json").write_text(
    json.dumps(
        {
            "model": "mlx-community/gemma-4-12B-it-4bit",
            "fine_tune_type": "lora",
            "num_layers": max(layers) + 1,
            "lora_parameters": {
                "rank": c["r"],
                "scale": c["lora_alpha"] / c["r"],
                "dropout": 0.0,
            },
        },
        indent=2,
    )
)
print("converted", len(out), "finite tensors")
