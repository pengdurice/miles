from types import SimpleNamespace

import pytest
import torch

from examples.on_policy_distillation import on_policy_distillation as opd
from miles.utils.types import Sample
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-fast")


class FakeTokenizer:
    def __init__(self, vocab: dict[int, str], encoded: dict[str, list[int]] | None = None):
        self.vocab = vocab
        self.encoded = encoded or {}

    def decode(self, token_ids, **kwargs):
        return "".join(self.vocab[token_id] for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        if text in self.encoded:
            return self.encoded[text]
        reverse_vocab = {value: key for key, value in self.vocab.items()}
        return [reverse_vocab[char] for char in text]


def test_align_teacher_log_probs_splits_coarse_teacher_token():
    aligned, covered = opd._align_teacher_log_probs_to_student_response(
        student_response_spans=[(0, 1), (1, 2), (2, 3)],
        teacher_spans=[(0, 2), (2, 3)],
        teacher_log_probs=[-4.0, -2.0],
    )

    assert aligned == pytest.approx([-2.0, -2.0, -2.0])
    assert covered == [1, 1, 1]


def test_post_process_cross_tokenizer_rewards_aligns_to_student_response(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "X", 2: "a", 3: "b", 4: "c"})
    teacher_tokenizer = FakeTokenizer(
        {10: "X", 11: "ab", 12: "c"},
        encoded={"Xabc": [10, 11, 12]},
    )

    def fake_load_tokenizer(path):
        return {"student": student_tokenizer, "teacher": teacher_tokenizer}[path]

    monkeypatch.setattr(opd, "_load_opd_tokenizer", fake_load_tokenizer)
    args = SimpleNamespace(
        opd_cross_tokenizer=True,
        opd_student_tokenizer="student",
        opd_teacher_tokenizer="teacher",
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

    teacher_log_probs, returned_rewards = opd.post_process_rewards(args, [sample])

    expected = torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float32)
    assert torch.allclose(sample.teacher_log_probs, expected)
    assert torch.allclose(teacher_log_probs[0], expected)
    assert torch.allclose(returned_rewards[0], expected)
    assert sample.metadata["opd_cross_tokenizer"]["teacher_token_count"] == 3
    assert sample.metadata["opd_cross_tokenizer"]["uncovered_student_response_tokens"] == 0


def test_build_cross_tokenizer_payload_uses_teacher_token_ids(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "h", 2: "i"})
    teacher_tokenizer = FakeTokenizer({10: "hi"}, encoded={"hi": [10]})

    def fake_load_tokenizer(path):
        return {"student": student_tokenizer, "teacher": teacher_tokenizer}[path]

    monkeypatch.setattr(opd, "_load_opd_tokenizer", fake_load_tokenizer)
    args = SimpleNamespace(
        opd_student_tokenizer="student",
        opd_teacher_tokenizer="teacher",
        hf_checkpoint=None,
    )

    payload = opd._build_cross_tokenizer_payload(args, Sample(tokens=[1, 2], response_length=2))

    assert payload["input_ids"] == [10]
    assert payload["return_logprob"] is True
    assert payload["sampling_params"]["max_new_tokens"] == 0
