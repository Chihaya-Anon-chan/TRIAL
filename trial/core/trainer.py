"""TRIAL integration with the vendored GRPO trainer."""

from typing import Optional

import torch

from trial.core.hindsight import build_outcome_view_batch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.trial_utils import build_relative_turn_profile


class RayTRIALTrainer(RayPPOTrainer):
    """GRPO trainer augmented with TRIAL's training-only OutcomeView."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"

    def _get_actor_worker_role(self) -> str:
        return "actor_rollout_ref"

    def _maybe_build_auxiliary_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ):
        cfg = self.config.actor_rollout_ref.actor.get("trial")
        if cfg is None:
            raise ValueError("loss_mode=trial requires actor.trial configuration.")
        if "traj_uid" not in batch.non_tensor_batch:
            raise ValueError("TRIAL requires per-trajectory identifiers in non_tensor_batch['traj_uid'].")

        response_length = batch.batch["responses"].size(1)
        outcome_batch = build_outcome_view_batch(
            batch=batch,
            tokenizer=self.tokenizer,
            response_length=response_length,
            max_prompt_length=int(self.config.data.max_prompt_length),
            outcome_view=str(cfg.get("outcome_view", "trajectory_context")),
            environment_name=str(self.config.env.env_name),
            outcome_observation_cap=cfg.get("outcome_observation_cap"),
        )
        scored_batch = self.actor_rollout_wg.compute_log_prob(outcome_batch)
        hindsight_log_probs = scored_batch.batch["old_log_probs"]
        response_mask = batch.batch["response_mask"]
        batch.batch["hindsight_log_probs"] = hindsight_log_probs
        batch.batch["hindsight_token_mask"] = response_mask

        profile_metrics = build_relative_turn_profile(
            batch=batch,
            gap_clip=float(cfg.get("gap_clip", 2.0)),
            informativeness="abs_delta",
            use_group_success_gate=False,
            aux_mask_key="hindsight_token_mask",
        )
        metrics = dict(profile_metrics)
        metrics.update(outcome_batch.meta_info)
        metrics["trial/outcome_view"] = "trajectory_context"
        return DataProto.from_dict(
            tensors={
                "hindsight_log_probs": hindsight_log_probs,
                "hindsight_token_mask": response_mask,
                "turn_weight": batch.batch["turn_weight"],
            }
        ), metrics
