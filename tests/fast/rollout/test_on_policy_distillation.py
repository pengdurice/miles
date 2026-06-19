"""Unit tests for cross-tokenizer on-policy distillation helpers.

Covers the character-span alignment algorithm, sglang response normalization,
and the full post_process_rewards round-trip with heterogeneous tokenizers.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

from types import SimpleNamespace

import pytest
import torch

from examples.on_policy_distillation import on_policy_distillation as opd
from miles.utils.types import Sample


class FakeTokenizer:
    """Minimal tokenizer for testing — maps token IDs to fixed strings."""

    def __init__(self, vocab: dict[int, str], encoded: dict[str, list[int]] | None = None):
        self.vocab = vocab
        self._encoded = encoded or {}

    def decode(self, token_ids, **kwargs):
        return "".join(self.vocab[tid] for tid in token_ids)

    def encode(self, text, add_special_tokens=False):
        if text in self._encoded:
            return self._encoded[text]
        reverse = {v: k for k, v in self.vocab.items()}
        return [reverse[ch] for ch in text]


# ---------------------------------------------------------------------------
# _align_teacher_log_probs_to_student_response
# ---------------------------------------------------------------------------


def test_align_splits_coarse_teacher_token_evenly():
    # One coarse teacher token covering chars 0-2, split across two fine student tokens (0-1), (1-2)
    aligned, covered = opd._align_teacher_log_probs_to_student_response(
        student_response_spans=[(0, 1), (1, 2), (2, 3)],
        teacher_spans=[(0, 2), (2, 3)],
        teacher_log_probs=[-4.0, -2.0],
    )
    assert aligned == pytest.approx([-2.0, -2.0, -2.0])
    assert covered == [1, 1, 1]


def test_align_none_logprob_skipped():
    aligned, covered = opd._align_teacher_log_probs_to_student_response(
        student_response_spans=[(0, 1), (1, 2)],
        teacher_spans=[(0, 1), (1, 2)],
        teacher_log_probs=[None, -3.0],
    )
    assert aligned == pytest.approx([0.0, -3.0])
    assert covered == [0, 1]


def test_align_zero_width_teacher_span_skipped():
    aligned, covered = opd._align_teacher_log_probs_to_student_response(
        student_response_spans=[(0, 2)],
        teacher_spans=[(0, 0), (0, 2)],
        teacher_log_probs=[-1.0, -5.0],
    )
    assert aligned == pytest.approx([-5.0])
    assert covered == [2]


# ---------------------------------------------------------------------------
# _normalize_input_token_logprobs
# ---------------------------------------------------------------------------


def test_normalize_exact_length():
    items = [{"logprob": -1.0}, {"logprob": -2.0}]
    result = opd._normalize_input_token_logprobs(items, 2)
    assert result == pytest.approx([-1.0, -2.0])


def test_normalize_one_short_prepends_none():
    items = [{"logprob": -2.0}]
    result = opd._normalize_input_token_logprobs(items, 2)
    assert result[0] is None
    assert result[1] == pytest.approx(-2.0)


def test_normalize_raises_on_bad_length():
    with pytest.raises(ValueError, match="does not match teacher token count"):
        opd._normalize_input_token_logprobs([{"logprob": -1.0}], 5)


# ---------------------------------------------------------------------------
# Full round-trip: post_process_rewards (cross-tokenizer path)
# ---------------------------------------------------------------------------


def test_post_process_cross_tokenizer_aligns_to_student_response(monkeypatch):
    # Student: tokens [1="X", 2="a", 3="b", 4="c"], response_length=3 (tokens 2,3,4)
    # Teacher: vocab {10:"X", 11:"ab", 12:"c"}, so "Xabc" -> [10, 11, 12]
    # Teacher logprobs: [None (first token), -4.0 (covers "ab" = chars 1-3), -2.0 (covers "c" = char 3)]
    # Student response spans (for "abc"): (1,2), (2,3), (3,4)
    # Teacher spans: (0,1), (1,3), (3,4)
    # Alignment:
    #   teacher[0]=None  -> skipped
    #   teacher[1]=-4.0 covers (1,3): student[0]=(1,2) gets -4.0*1/2=-2.0, student[1]=(2,3) gets -4.0*1/2=-2.0
    #   teacher[2]=-2.0 covers (3,4): student[2]=(3,4) gets -2.0
    student_tokenizer = FakeTokenizer({1: "X", 2: "a", 3: "b", 4: "c"})
    teacher_tokenizer = FakeTokenizer(
        {10: "X", 11: "ab", 12: "c"},
        encoded={"Xabc": [10, 11, 12]},
    )

    def fake_load(path):
        return {"student": student_tokenizer, "teacher": teacher_tokenizer}[path]

    monkeypatch.setattr(opd, "_load_opd_tokenizer", fake_load)

    args = SimpleNamespace(
        opd_cross_tokenizer=True,
        opd_cross_student_tokenizer="student",
        opd_cross_teacher_tokenizer="teacher",
        hf_checkpoint=None,
        reward_key=None,
    )
    sample = Sample(
        tokens=[1, 2, 3, 4],
        response_length=3,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [None, 10],
                    [-4.0, 11],
                    [-2.0, 12],
                ]
            }
        },
    )

    scalar_rewards, teacher_log_probs = opd.post_process_rewards(args, [sample])

    expected = torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float32)
    assert torch.allclose(sample.teacher_log_probs, expected)
    assert torch.allclose(teacher_log_probs[0], expected)
    assert scalar_rewards == [0.0]
    assert sample.metadata["opd_cross_tokenizer"]["teacher_token_count"] == 3
    assert sample.metadata["opd_cross_tokenizer"]["uncovered_student_response_tokens"] == 0


def test_post_process_same_tokenizer_returns_scalar_zero(monkeypatch):
    # The same-tokenizer path should also return scalar_rewards=[0.0], not teacher_log_probs
    args = SimpleNamespace(opd_cross_tokenizer=False, reward_key=None)
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [None, 1],
                    [-1.0, 2],
                    [-2.0, 3],
                ]
            }
        },
    )

    scalar_rewards, teacher_log_probs = opd.post_process_rewards(args, [sample])

    assert scalar_rewards == [0.0]
    assert isinstance(teacher_log_probs[0], torch.Tensor)
    assert len(teacher_log_probs[0]) == sample.response_length


# ---------------------------------------------------------------------------
# _build_cross_tokenizer_payload
# ---------------------------------------------------------------------------


def test_build_cross_tokenizer_payload_uses_teacher_ids(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "h", 2: "i"})
    teacher_tokenizer = FakeTokenizer({10: "hi"}, encoded={"hi": [10]})

    def fake_load(path):
        return {"student": student_tokenizer, "teacher": teacher_tokenizer}[path]

    monkeypatch.setattr(opd, "_load_opd_tokenizer", fake_load)
    args = SimpleNamespace(
        opd_cross_student_tokenizer="student",
        opd_cross_teacher_tokenizer="teacher",
        hf_checkpoint=None,
    )

    payload = opd._build_cross_tokenizer_payload(args, Sample(tokens=[1, 2], response_length=2))

    assert payload["input_ids"] == [10]
    assert payload["return_logprob"] is True
    assert payload["sampling_params"]["max_new_tokens"] == 0
