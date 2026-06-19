from functools import lru_cache
from typing import Any

import aiohttp
import torch

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample


# ---------------------------------------------------------------------------
# Tokenizer helpers (cross-tokenizer path)
# ---------------------------------------------------------------------------


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids)


def _encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@lru_cache(maxsize=8)
def _load_opd_tokenizer(name_or_path: str):
    return load_tokenizer(name_or_path, trust_remote_code=True)


def _student_tokenizer_path(args) -> str:
    path = getattr(args, "opd_cross_student_tokenizer", None) or getattr(args, "hf_checkpoint", None)
    if not path:
        raise ValueError("--opd-cross-tokenizer requires --opd-cross-student-tokenizer or --hf-checkpoint")
    return path


def _teacher_tokenizer_path(args) -> str:
    path = getattr(args, "opd_cross_teacher_tokenizer", None)
    if not path:
        raise ValueError("--opd-cross-tokenizer requires --opd-cross-teacher-tokenizer")
    return path


def _token_spans_from_prefix_decodes(tokenizer, token_ids: list[int]) -> list[tuple[int, int]]:
    """Return [(char_start, char_end), ...] for each token.

    Uses return_offsets_mapping=True (O(n)) when the tokenizer supports it;
    falls back to O(n) individual-token decode otherwise.  The old O(n²)
    prefix-decode loop is never used — it was the main rollout bottleneck.
    """
    if not token_ids:
        return []
    full_text = _decode_tokens(tokenizer, token_ids)
    # Fast path: fast tokenizers expose offset_mapping in O(n).
    try:
        enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc.get("offset_mapping")
        if offsets is not None and len(offsets) == len(token_ids):
            return [(int(s), int(e)) for s, e in offsets]
    except Exception:
        pass
    # Fallback: decode each token individually (O(n), minor boundary inaccuracies
    # possible for some tokenizers but acceptable for span alignment).
    spans: list[tuple[int, int]] = []
    pos = 0
    for tid in token_ids:
        piece = _decode_tokens(tokenizer, [tid])
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    return spans


def _extract_logprob(item: Any) -> float | None:
    if item is None:
        return None
    if isinstance(item, dict):
        value = item.get("logprob")
    else:
        value = item[0]
    return None if value is None else float(value)


def _normalize_input_token_logprobs(raw_logprobs: list[Any], token_count: int) -> list[float | None]:
    """Normalise sglang's input_token_logprobs to exactly token_count entries.

    sglang may return either token_count or token_count-1 entries depending on
    whether it includes a logprob for the first token (which has no predecessor).
    """
    if len(raw_logprobs) == token_count:
        return [_extract_logprob(item) for item in raw_logprobs]
    if len(raw_logprobs) == token_count - 1:
        return [None] + [_extract_logprob(item) for item in raw_logprobs]
    raise ValueError(
        f"teacher input_token_logprobs length ({len(raw_logprobs)}) does not match teacher token count "
        f"({token_count})"
    )


def _align_teacher_log_probs_to_student_response(
    student_response_spans: list[tuple[int, int]],
    teacher_spans: list[tuple[int, int]],
    teacher_log_probs: list[float | None],
) -> tuple[list[float], list[int]]:
    """Align teacher token logprobs onto student response tokens via character-span overlap.

    When teacher and student tokenizers differ, a single teacher token may cover
    multiple student tokens (or vice-versa). This function distributes the teacher
    logprob proportionally by the character overlap fraction, so each student token
    receives a weighted sum of the teacher logprobs that cover it.

    Returns:
        aligned: one float per student response token (weighted teacher logprob).
        covered_chars: character coverage count per student response token (for diagnostics).
    """
    aligned = [0.0 for _ in student_response_spans]
    covered_chars = [0 for _ in student_response_spans]

    for (teacher_start, teacher_end), teacher_log_prob in zip(teacher_spans, teacher_log_probs, strict=True):
        if teacher_log_prob is None or teacher_end <= teacher_start:
            continue
        teacher_width = teacher_end - teacher_start
        for i, (student_start, student_end) in enumerate(student_response_spans):
            overlap = max(0, min(teacher_end, student_end) - max(teacher_start, student_start))
            if overlap == 0:
                continue
            aligned[i] += teacher_log_prob * (overlap / teacher_width)
            covered_chars[i] += overlap

    return aligned, covered_chars


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _build_cross_tokenizer_payload(args, sample: Sample) -> dict:
    student_tokenizer = _load_opd_tokenizer(_student_tokenizer_path(args))
    teacher_tokenizer = _load_opd_tokenizer(_teacher_tokenizer_path(args))
    full_text = _decode_tokens(student_tokenizer, sample.tokens)
    teacher_input_ids = _encode_text(teacher_tokenizer, full_text)
    if not teacher_input_ids:
        raise ValueError("cross-tokenizer OPD: teacher input is empty after retokenization")

    return {
        "input_ids": teacher_input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }


def _build_same_tokenizer_payload(sample: Sample) -> dict:
    return {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }


# ---------------------------------------------------------------------------
# Reward function (called once per sample during rollout)
# ---------------------------------------------------------------------------


async def reward_func(args, sample, **kwargs):
    if getattr(args, "opd_cross_tokenizer", False):
        payload = _build_cross_tokenizer_payload(args, sample)
    else:
        payload = _build_same_tokenizer_payload(sample)

    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


# ---------------------------------------------------------------------------
# Post-processing (called once per batch after rollout)
# ---------------------------------------------------------------------------


def _post_process_cross_tokenizer_rewards(args, samples: list[Sample]) -> list[torch.Tensor]:
    student_tokenizer = _load_opd_tokenizer(_student_tokenizer_path(args))
    teacher_tokenizer = _load_opd_tokenizer(_teacher_tokenizer_path(args))
    rewards = [sample.get_reward_value(args) for sample in samples]
    teacher_log_probs: list[torch.Tensor] = []

    for sample, reward in zip(samples, rewards, strict=False):
        full_text = _decode_tokens(student_tokenizer, sample.tokens)
        teacher_input_ids = _encode_text(teacher_tokenizer, full_text)
        raw_teacher_log_probs = reward["meta_info"]["input_token_logprobs"]
        token_log_probs = _normalize_input_token_logprobs(raw_teacher_log_probs, len(teacher_input_ids))

        student_spans = _token_spans_from_prefix_decodes(student_tokenizer, sample.tokens)
        teacher_spans = _token_spans_from_prefix_decodes(teacher_tokenizer, teacher_input_ids)
        student_response_spans = student_spans[-sample.response_length:] if sample.response_length > 0 else []
        aligned, covered_chars = _align_teacher_log_probs_to_student_response(
            student_response_spans,
            teacher_spans,
            token_log_probs,
        )

        t_log_probs = torch.tensor(aligned, dtype=torch.float32)
        sample.teacher_log_probs = t_log_probs
        sample.metadata.setdefault("opd_cross_tokenizer", {})
        sample.metadata["opd_cross_tokenizer"].update(
            {
                "teacher_token_count": len(teacher_input_ids),
                "uncovered_student_response_tokens": sum(
                    1
                    for span, covered in zip(student_response_spans, covered_chars, strict=False)
                    if span[1] > span[0] and covered == 0
                ),
            }
        )
        teacher_log_probs.append(t_log_probs)

    return teacher_log_probs


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process teacher server response and populate sample.teacher_log_probs.

    Returns (raw_rewards, scalar_rewards) matching the contract of
    _post_process_rewards in train_data_conversion.py:
      raw_rewards  — original reward values (0.0 for pure distillation)
      scalar_rewards — float rewards for the advantage estimator (also 0.0)

    teacher_log_probs are stored on sample.teacher_log_probs and picked up
    by convert_samples_to_train_data via train_data["teacher_log_probs"].
    """
    if getattr(args, "opd_cross_tokenizer", False):
        _post_process_cross_tokenizer_rewards(args, samples)
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
