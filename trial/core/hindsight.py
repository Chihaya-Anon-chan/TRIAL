"""Construction of TRIAL's training-only OutcomeView.

The same representation is used for every supported environment: append a
compact record of the action and its realized outcome to the original prompt,
then score the unchanged sampled response under that context.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import torch

from verl.utils.model import compute_position_id_with_mask


_THINKING = re.compile(r"<think\\b[^>]*>.*?</think>\\s*", re.IGNORECASE | re.DOTALL)
_ACTION = re.compile(r"<action\\b[^>]*>(.*?)</action>", re.IGNORECASE | re.DOTALL)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist() if value.ndim > 0 else value.item()
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(item) for item in value if item is not None)
    return str(value).strip()


def _action_text(response: Any) -> str:
    """Project a model response to the action supplied to the environment."""
    text = _THINKING.sub("", _text(response))
    match = _ACTION.search(text)
    return (match.group(1) if match else text).strip()


def _truncate_tokens(text: str, tokenizer, limit: int) -> tuple[str, bool]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text, False
    return tokenizer.decode(ids[:limit], skip_special_tokens=True), True


def _outcome_text(next_observation: Any, feedback: Any) -> str:
    return _text(next_observation) or _text(feedback) or "(no outcome was recorded)"


def _render_outcome_view(action: str, outcome: str) -> str:
    return (
        "\n\n[Training-only outcome context]\n"
        f"Action: {action or '(empty action)'}\n"
        f"Observed outcome: {outcome}\n"
    )


def build_outcome_view_batch(
    batch,
    tokenizer,
    response_length: int,
    max_prompt_length: int,
    outcome_view: str = "trajectory_context",
    environment_name: str = "",
    outcome_observation_cap: Optional[int] = None,
    **_: Any,
):
    """Build an aligned scoring batch for the paper's sole OutcomeView.

    ``environment_name`` is intentionally accepted only for the common trainer
    interface; rendering is environment-agnostic.  The sampled response tokens
    are copied byte-for-byte so their log probabilities remain comparable.
    """
    if outcome_view != "trajectory_context":
        raise ValueError(
            "TRIAL supports only outcome_view='trajectory_context'; "
            f"got {outcome_view!r}."
        )
    if outcome_observation_cap is None:
        outcome_observation_cap = 1024
    outcome_observation_cap = int(outcome_observation_cap)
    if outcome_observation_cap <= 0:
        raise ValueError("outcome_observation_cap must be positive.")

    from verl import DataProto

    responses = batch.batch["responses"]
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    if response_length != responses.size(1):
        raise ValueError("response_length must match the sampled response width.")
    response_texts = batch.non_tensor_batch.get("response_text")
    next_observations = batch.non_tensor_batch.get("next_observation_text")
    feedbacks = batch.non_tensor_batch.get("immediate_feedback")
    if response_texts is None or (next_observations is None and feedbacks is None):
        raise RuntimeError(
            "TRIAL requires response_text and at least one of "
            "next_observation_text or immediate_feedback from the rollout."
        )

    view_ids, view_masks, view_positions = [], [], []
    truncated = 0
    overflow = 0
    for row in range(responses.size(0)):
        prompt_width = input_ids.size(1) - response_length
        prompt_ids = input_ids[row, :prompt_width]
        prompt_mask = attention_mask[row, :prompt_width]
        valid = prompt_mask.nonzero(as_tuple=True)[0]
        student_prompt = prompt_ids[valid[0].item() if len(valid) else 0 :].tolist()

        action = _action_text(response_texts[row] if row < len(response_texts) else "")
        next_observation = next_observations[row] if next_observations is not None and row < len(next_observations) else None
        feedback = feedbacks[row] if feedbacks is not None and row < len(feedbacks) else None
        outcome, was_truncated = _truncate_tokens(
            _outcome_text(next_observation, feedback), tokenizer, outcome_observation_cap
        )
        truncated += int(was_truncated)
        suffix = tokenizer.encode(_render_outcome_view(action, outcome), add_special_tokens=False)
        prompt = student_prompt + suffix
        if len(prompt) > max_prompt_length:
            prompt = prompt[-max_prompt_length:]
            overflow += 1
        padding = max_prompt_length - len(prompt)
        padded_prompt = [tokenizer.pad_token_id] * padding + prompt
        prompt_mask = [0] * padding + [1] * len(prompt)
        full_ids = torch.tensor(padded_prompt, dtype=torch.long)
        full_mask = torch.tensor(prompt_mask, dtype=torch.long)
        full_ids = torch.cat([full_ids, responses[row]])
        full_mask = torch.cat([full_mask, attention_mask[row, -response_length:]])
        view_ids.append(full_ids)
        view_masks.append(full_mask)
        view_positions.append(compute_position_id_with_mask(full_mask.unsqueeze(0))[0])

    input_tensor = torch.stack(view_ids)
    if not torch.equal(input_tensor[:, -response_length:], responses):
        raise AssertionError("OutcomeView response tokens must exactly match rollout response tokens.")
    result = DataProto.from_dict(
        tensors={
            "input_ids": input_tensor,
            "attention_mask": torch.stack(view_masks),
            "position_ids": torch.stack(view_positions),
            "responses": responses,
        }
    )
    rows = max(responses.size(0), 1)
    result.meta_info["trial/outcome_view_truncation_rate"] = truncated / rows
    result.meta_info["trial/outcome_view_overflow_rate"] = overflow / rows
    return result
