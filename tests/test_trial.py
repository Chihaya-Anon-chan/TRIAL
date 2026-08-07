from types import SimpleNamespace
import unittest

import torch

from verl.trainer.ppo.trial_utils import (
    apply_turn_profile_control,
    build_relative_turn_profile,
    compute_dense_hindsight_loss,
)


class TestTRIAL(unittest.TestCase):
    def test_dense_loss_has_the_expected_stop_gradient(self):
        teacher = torch.tensor([[-0.3, -0.8, -0.2], [-0.5, -0.4, -0.9]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        student = torch.tensor(
            [[-0.6, -0.7, -0.4], [-0.2, -0.9, -0.8]], requires_grad=True
        )

        loss, _ = compute_dense_hindsight_loss(
            student_log_probs=student,
            hindsight_log_probs=teacher,
            response_mask=mask,
            turn_weight=torch.ones_like(mask),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(student.grad.abs().sum().item(), 0.0)

    def test_relative_profile_conserves_each_trajectory_budget(self):
        response_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        batch = SimpleNamespace(
            batch={
                "response_mask": response_mask,
                "old_log_probs": torch.zeros_like(response_mask),
                "hindsight_log_probs": torch.tensor([[1.0, 1.0], [0.25, 0.25], [0.5, 0.0]]),
            },
            non_tensor_batch={"traj_uid": ["episode-a", "episode-a", "episode-b"]},
        )

        metrics = build_relative_turn_profile(batch, gap_clip=2.0)
        applied = apply_turn_profile_control(batch, metrics, "relative")

        weights = batch.batch["turn_weight"]
        torch.testing.assert_close((weights[:2, 0] * response_mask[:2].sum(-1)).sum(), response_mask[:2].sum())
        self.assertLess(applied["trial/audit_max_relative_residual"], 1e-6)

    def test_non_relative_profile_is_rejected(self):
        batch = SimpleNamespace(batch={"turn_weight": torch.ones(1, 1)})
        with self.assertRaisesRegex(ValueError, "relative"):
            apply_turn_profile_control(batch, {}, "uniform")


if __name__ == "__main__":
    unittest.main()
