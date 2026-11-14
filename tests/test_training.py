"""Optional numerical tests: run with a CPU torch installation, no model download."""

import pytest

torch = pytest.importorskip("torch")
from recipe.train import collate, weighted_loss_sum, batches_for_epoch  # noqa: E402


def test_accumulation_matches_padded_batch():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 6)
            self.head = torch.nn.Linear(6, 8)

        def forward(self, input_ids, attention_mask):
            from types import SimpleNamespace

            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    rows = [
        {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "loss_weights": [0, 1, 0.5]},
        {"input_ids": [1, 4], "labels": [-100, 4], "loss_weights": [0, 2]},
    ]
    torch.manual_seed(7)
    model = Toy()
    loss, weight = weighted_loss_sum(model, *collate(rows, "cpu", 0))
    assert weight.item() == 3.5
    micro = [weighted_loss_sum(model, *collate([r], "cpu", 0)) for r in rows]
    assert torch.allclose(loss, sum(x[0] for x in micro), atol=1e-6)
    (loss / weight).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert batches_for_epoch(5, 2, 0) == batches_for_epoch(5, 2, 0)
