#!/usr/bin/env python3
"""Run TRIAL's teacher-gap, allocation, loss, and backward pass on one GPU.

This validates the model-facing core without benchmark assets. It does not
replace an ALFWorld or WebShop training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.trainer.ppo.trial_utils import build_relative_turn_profile, compute_dense_hindsight_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model id")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("A CUDA device is required for this smoke test.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).train()
    responses = [" the agent clicks search", " the agent clicks item"]
    plain_prompts = ["Task: buy a book. Action:"] * 2
    hindsight_prompts = [
        "Task: buy a book. Hindsight outcome: search revealed a matching item. Action:",
        "Task: buy a book. Hindsight outcome: the item page did not match. Action:",
    ]

    def response_log_probs(prompts: list[str]) -> torch.Tensor:
        values = []
        for prompt, response in zip(prompts, responses):
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            response_ids = tokenizer(response, add_special_tokens=False).input_ids
            ids = torch.tensor([prompt_ids + response_ids], device=args.device)
            logits = model(input_ids=ids).logits[:, :-1]
            labels = ids[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            values.append(log_probs[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(response_ids)])
        width = min(value.numel() for value in values)
        return torch.stack([value[:width] for value in values])

    student = response_log_probs(plain_prompts)
    with torch.no_grad():
        teacher = response_log_probs(hindsight_prompts)
    response_mask = torch.ones_like(student)
    batch = SimpleNamespace(
        batch={
            "old_log_probs": student.detach(),
            "hindsight_log_probs": teacher,
            "response_mask": response_mask,
        },
        non_tensor_batch={"traj_uid": np.array(["trajectory-0", "trajectory-0"], dtype=object)},
    )
    metrics = build_relative_turn_profile(batch, gap_clip=2.0)
    loss, loss_metrics = compute_dense_hindsight_loss(
        student,
        teacher,
        response_mask,
        turn_weight=batch.batch["turn_weight"],
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).item()
    if not torch.isfinite(loss) or grad_norm <= 0:
        raise RuntimeError("TRIAL GPU smoke failed: non-finite loss or zero gradient.")
    if metrics["trial/audit_max_relative_residual"] >= 1e-5:
        raise RuntimeError("TRIAL GPU smoke failed: turn-weight budget was not conserved.")
    print(
        {
            "loss": round(float(loss.detach()), 8),
            "grad_norm": round(grad_norm, 8),
            "weight_mean": round(metrics["trial/w_mean"], 8),
            "budget_residual": metrics["trial/audit_max_relative_residual"],
            "gap_abs_mean": round(loss_metrics["hindsight/gap_abs_mean"], 8),
        }
    )


if __name__ == "__main__":
    main()
