import json
from pathlib import Path

import pytest

from recipe.api import api_safe, request
from recipe.app import inspect_reply
from recipe.format import GEN_PREFIX, encode_window, window_text
from recipe.ingest import bursts, normalize, parse_export, redact
from recipe.memory import Memory, render
from recipe.pipeline import build, ingest, preflight, review_page, signoff


class Tok:
    is_fast = True

    def encode(self, s, **kw):
        return [ord(c) for c in s]

    def __call__(self, s, **kw):
        return {
            "input_ids": self.encode(s),
            "offset_mapping": [(i, i + 1) for i in range(len(s))],
        }

    def apply_chat_template(self, messages, **kw):
        return window_text(messages)[0] + GEN_PREFIX

    def save_pretrained(self, p):
        Path(p).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def cfg(tmp_path):
    return dict(
        work_dir=str(tmp_path),
        export=str(Path("examples/chat.txt").resolve()),
        self_sender="Alex",
        other_sender="Sam",
        self_name="Alex",
        other_name="Sam",
        date_order="DMY",
        valid_start="2026-02-01",
        test_start="2026-03-01",
        recent_start="2026-01-01",
        model="fictional/test",
        model_revision="test",
        max_len=10000,
        prompt_budget=9000,
        max_turns=40,
    )


def test_multiline_and_ios(tmp_path, cfg):
    p = tmp_path / "chat.txt"
    p.write_text(
        "[12/01/2026, 9:05:03 PM] Alex: hello\nsecond line<This message was edited>\n[12/01/2026, 9:06:03 PM] Sam: hey\n"
    )
    rows, _ = parse_export(p, cfg)
    assert len(rows) == 2
    assert rows[0]["ts"] == "2026-01-12T21:05:03"
    norm = normalize(rows, cfg)
    assert norm[0]["text"] == "hello second line"
    assert len(bursts(norm, cfg)) == 2


def test_system_continuation_does_not_attach_to_previous(tmp_path, cfg):
    p = tmp_path / "chat.txt"
    p.write_text(
        "01/01/2026, 10:00 - Alex: hi\n01/01/2026, 10:01 - security notice\ncontinuation of system notice\n"
    )
    with pytest.raises(ValueError):
        parse_export(p, cfg)


def test_unknown_sender_fails(tmp_path, cfg):
    p = tmp_path / "chat.txt"
    p.write_text("01/01/2026, 10:00 - Alex: hi\n01/01/2026, 10:01 - Third: surprise\n")
    with pytest.raises(ValueError):
        parse_export(p, cfg)


def test_no_language_deletions_and_real_bursts(cfg):
    rows, _ = parse_export(cfg["export"], cfg)
    norm = normalize(rows, cfg)
    assert len(norm) == len(rows)
    assert any(r["lang"] == "mixed" for r in norm)
    own = [b for b in bursts(norm, cfg) if b["sender"] == "self"]
    assert all(len(b["messages"]) == 2 for b in own)


def test_private_redaction_and_api_tagging():
    assert "@" not in redact("email a@example.test or +91 99999 11111")
    assert (
        api_safe({"text": "keep locally", "explicit": True})["text"]
        == "[intimate message]"
    )
    assert "Private Person" not in redact(
        "Private Person works there", {"Private Person": "Person A"}
    )
    assert "<|turn>" not in redact("<|turn>system")


def test_api_disabled_before_network(cfg, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("network called")
    )
    with pytest.raises(ValueError):
        request(cfg, "bridge", {}, "test")


def test_asof_memory_owner_and_replacement(tmp_path):
    m = Memory(tmp_path / "memory.sqlite3")
    for id_, owner, at, text in [
        ("a", "self", "2026-01-01T00:00:00", "old role"),
        ("b", "other", "2026-01-01T00:00:00", "other role"),
        ("c", "self", "2026-02-01T00:00:00", "new role"),
    ]:
        m.put(
            dict(
                id=id_,
                owner=owner,
                kind="fact",
                predicate="job",
                text=text,
                observed_at=at,
            )
        )
    before = m.select("role", "2026-01-15T00:00:00")
    assert {r["id"] for r in before} == {"a", "b"}
    after = m.select("role", "2026-03-01T00:00:00")
    assert {r["id"] for r in after} == {"b", "c"}
    m.put(
        dict(
            id="future-summary",
            owner="shared",
            kind="summary",
            text="role",
            observed_at="2026-01-01T00:00:00",
            available_at="2026-04-01T00:00:00",
        )
    )
    assert "future-summary" not in {
        r["id"] for r in m.select("role", "2026-03-01T00:00:00")
    }
    m.close()


def test_unseen_mask_and_end_turn():
    tok = Tok()
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "real reply\n[unseen photo]"},
    ]
    enc = encode_window(tok, msgs, {2: [1, 0]})
    text, spans = window_text(msgs, {2})
    start, end = spans[2]
    assert all(x == -100 for x in enc["labels"][:start])
    assert any(x != -100 for x in enc["labels"][start : start + 10])
    assert all(x == -100 for x in enc["labels"][start + len("real reply") :])
    with pytest.raises(ValueError):
        encode_window(tok, msgs, {2: [0, 0]})


def test_pipeline_and_tamper_gate(cfg):
    ingest(cfg)
    result = build(cfg, Tok())
    assert result["everything"]["train"] > 0
    report = preflight(cfg)
    review_page(cfg)
    signoff(cfg)
    signed = json.loads((Path(cfg["work_dir"]) / "signoff.json").read_text())
    assert signed["dataset_sha256"] == report["dataset_sha256"]
    path = Path(cfg["work_dir"]) / "datasets/everything/train.jsonl"
    path.write_text(path.read_text() + "\n")
    from recipe.common import dataset_hash

    assert dataset_hash(path.parents[1]) != signed["dataset_sha256"]
    assert (
        len(
            json.loads((Path(cfg["work_dir"]) / "datasets/build.json").read_text())[
                "report"
            ]
        )
        == 2
    )


def test_template_drift_fails(cfg):
    ingest(cfg)

    class Bad(Tok):
        def apply_chat_template(self, *a, **k):
            return "changed"

    with pytest.raises(ValueError, match="template"):
        build(cfg, Bad())


def test_render_shares_context_budget(cfg, tmp_path):
    mem = Memory(tmp_path / "memory.sqlite3")
    history = [
        {"role": "user", "content": "hello", "at": "2026-01-01T00:00:00"},
        {"role": "assistant", "content": "hey", "at": "2026-01-01T00:01:00"},
        {"role": "user", "content": "again", "at": "2026-01-01T04:00:00"},
    ]
    msgs, kept, _ = render(cfg, Tok(), history, mem, "2026-01-01T04:00:00")
    assert "[4 hours later]" in msgs[-1]["content"]
    assert kept[-1]["_gap"]
    mem.close()


def test_output_checks():
    assert inspect_reply("as an ai, hello")
    assert inspect_reply("hi\nhi")
    assert not inspect_reply("yes\nyou are going to lose")


def test_burst_does_not_cross_split(cfg):
    rows = [
        {"id": "x", "ts": "2026-01-31T23:59:00", "sender": "self"},
        {"id": "y", "ts": "2026-02-01T00:01:00", "sender": "self"},
    ]
    assert len(bursts(rows, cfg)) == 2


def test_targeted_style_uses_train_only(cfg):
    from recipe.tasks import make_jobs

    ingest(cfg)
    examples = make_jobs(cfg, "targeted")[0]["style_examples"]
    assert all(m["ts"] < cfg["valid_start"] for m in examples)


def test_long_window_coverage_and_media_masks(cfg):
    from recipe.common import read_jsonl

    # Repeat the fictional source across many distinct days to exercise overlap/fallbacks.
    p = Path(cfg["work_dir"]) / "long.txt"
    lines = []
    for month in (1, 2, 3):
        for day in range(1, 25):
            lines.extend(
                [
                    f"{day:02}/{month:02}/2026, 10:00 - Sam: hi",
                    f"{day:02}/{month:02}/2026, 10:01 - Alex: hey",
                    f"{day:02}/{month:02}/2026, 10:02 - Alex: kal chalo",
                ]
            )
    p.write_text("\n".join(lines))
    cfg["export"] = str(p)
    ingest(cfg)
    build(cfg, Tok())
    preflight(cfg)
    report = json.loads((Path(cfg["work_dir"]) / "datasets/build.json").read_text())
    assert report["eligible"] == report["supervised"]
    renders = read_jsonl(Path(cfg["work_dir"]) / "datasets/windows_render.jsonl")
    assert len(renders) > 6


def test_unseen_photo_reply_not_supervised(cfg):
    from recipe.pipeline import make_turns

    rows = [
        {
            "id": "p",
            "ts": "2026-01-01T10:00:00",
            "sender": "other",
            "text": "IMG-001.jpg (file attached)",
        },
        {
            "id": "a",
            "ts": "2026-01-01T10:01:00",
            "sender": "self",
            "text": "nice picture",
        },
    ]
    grouped = bursts(normalize(rows, cfg), cfg)
    turns = make_turns(cfg, grouped, "everything", {})
    assert turns[-1]["weights"] == [0.0]


def test_budget_fallback_does_not_duplicate(cfg):
    cfg.update(prompt_budget=1100, max_len=1300)
    ingest(cfg)
    build(cfg, Tok())
    report = json.loads((Path(cfg["work_dir"]) / "datasets/build.json").read_text())
    assert report["supervised"]["everything"] <= report["eligible"]["everything"]


def test_prefix_errors_are_not_mislabeled_oversize(cfg, monkeypatch):
    ingest(cfg)

    def broken(*args, **kwargs):
        raise ValueError("tokenizer merged across the prompt/reply boundary")

    monkeypatch.setattr("recipe.pipeline.encode_window", broken)
    with pytest.raises(ValueError, match="tokenizer merged"):
        build(cfg, Tok())


def test_historical_summaries_do_not_supersede_each_other(tmp_path):
    m = Memory(tmp_path / "memory.sqlite3")
    for id_, at in [("jan", "2026-02-01T00:00:00"), ("feb", "2026-03-01T00:00:00")]:
        m.put(
            dict(
                id=id_,
                owner="shared",
                kind="summary",
                predicate="monthly",
                text="A historical month",
                observed_at=at,
            )
        )
    assert {r["id"] for r in m.select("month", "2026-04-01T00:00:00")} == {"jan", "feb"}
    m.close()
