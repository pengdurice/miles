from functools import lru_cache
from typing import Any

import aiohttp
import torch

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample


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
    path = getattr(args, "opd_student_tokenizer", None) or getattr(args, "hf_checkpoint", None)
    if not path:
        raise ValueError("--opd-cross-tokenizer requires --opd-student-tokenizer or --hf-checkpoint")
    return path


def _teacher_tokenizer_path(args) -> str:
    path = getattr(args, "opd_teacher_tokenizer", None)
    if not path:
        raise ValueError("--opd-cross-tokenizer requires --opd-teacher-tokenizer")
    return path


def _token_spans_from_prefix_decodes(tokenizer, token_ids: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev_len = 0
    for i in range(1, len(token_ids) + 1):
        cur_len = len(_decode_tokens(tokenizer, token_ids[:i]))
        spans.append((prev_len, cur_len))
        prev_len = cur_len
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


def _build_cross_tokenizer_payload(args, sample: Sample) -> dict:
    student_tokenizer = _load_opd_tokenizer(_student_tokenizer_path(args))
    teacher_tokenizer = _load_opd_tokenizer(_teacher_tokenizer_path(args))
    full_text = _decode_tokens(student_tokenizer, sample.tokens)
    teacher_input_ids = _encode_text(teacher_tokenizer, full_text)
    if not teacher_input_ids:
        raise ValueError("cross-tokenizer OPD teacher input is empty after retokenization")

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


async def reward_func(args, sample, **kwargs):
    if getattr(args, "opd_cross_tokenizer", False):
        payload = _build_cross_tokenizer_payload(args, sample)
    else:
        payload = {
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
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


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
        student_response_spans = student_spans[-sample.response_length :] if sample.response_length > 0 else []
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
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if getattr(args, "opd_cross_tokenizer", False):
        teacher_log_probs = _post_process_cross_tokenizer_rewards(args, samples)
        return teacher_log_probs, teacher_log_probs

    rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
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

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
