# On-Policy Distillation Example

This example shows how to run **on-policy distillation** using miles. A small student (Qwen3-8B) is aligned to imitate a larger teacher (Qwen3-32B) by training only on the student's own rollouts and matching the teacher's token-level log-probabilities.

In this example, the teacher model acts as a reward model (RM) by providing teacher log probabilities as the supervision signal.

## Components

- `on_policy_distillation.py` implements::
  - `reward_func` calls the teacher server (via `args.rm_url`) with every sample to obtain token-level logprobs.
  - `post_process_rewards` trims the teacher logprobs to the generated response span and writes the tensors back to each `Sample` to compute advantages.
- `run-qwen3-8B-opd.sh` launches an SGLang teacher server, then submits a Ray job that runs `train.py`.

## Cross-tokenizer OPD

The default path assumes teacher and student share a tokenizer: the student token IDs are sent directly to the teacher. For heterogeneous teacher/student pairs, enable the cross-tokenizer path:

```bash
--opd-cross-tokenizer \
--opd-teacher-tokenizer /path/to/teacher-hf-checkpoint \
--opd-student-tokenizer /path/to/student-hf-checkpoint
```

`--opd-student-tokenizer` defaults to `--hf-checkpoint` when omitted. In this mode, miles decodes the student rollout, re-tokenizes the same text with the teacher tokenizer, asks the teacher to score those teacher token IDs, then aligns teacher token logprobs back to student response tokens by decoded character spans. This keeps the training-side OPD contract unchanged (`teacher_log_probs` has one value per student response token) while avoiding the invalid “student token IDs interpreted by teacher vocab” failure mode.

This first implementation is a text/span likelihood bridge. It does not yet implement full SimCT-style multi-token candidate continuation distributions or ULD/GOLD-style merged top-k logit losses.

## Running the example

1. Download or prepare the required checkpoints and data.
```bash
hf download Qwen/Qwen3-32B --local-dir /root/Qwen3-32B
hf download Qwen/Qwen3-8B --local-dir /root/Qwen3-8B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
```

2. Run the hf to mcore for student model conversion:
```bash
cd /root/miles
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-8B \
    --save /root/Qwen3-8B_torch_dist
```
3. run on-policy distillation:
```bash
bash examples/on_policy_distillation/run-qwen3-8B-opd.sh
```


# Preliminary Results
Using Qwen3-8B-Base model sfted on part of the [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) dataset, we performed on-policy distillation with a Qwen3-32B teacher on the remaining data. Evaluation on Math500 shows:

|                                  | Pass@1 |
|-----------------------------------------------|--------|
| Qwen3-8B-Base + SFT                           | 76%    |
| Qwen3-8B-Base + SFT + On-Policy Distillation  | 94%    |





# FAQ
1. **Why are teacher logits computed via a sglang server instead of inside the training backend?**
The teacher runs on an independent SGLang server that miles treats as a reward model. Hosting it inside Megatron/FSDP would require maintaining a second, fully configured training stack for the teacher.


# References
1. https://thinkingmachines.ai/blog/on-policy-distillation/
2. https://arxiv.org/abs/2306.13649
3. https://arxiv.org/abs/2306.08543