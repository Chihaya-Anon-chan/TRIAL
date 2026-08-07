"""Core loss and trajectory-relative allocation used by TRIAL.

For each realized turn, the training-only OutcomeView scorer evaluates the
actor's sampled response under hindsight context. TRIAL uses the resulting
token log-probability gap as dense feedback and normalizes each turn's
absolute gap within its trajectory.

    δ_t = clip(log π_teacher(y_t) - log π_student(y_t), -c, +c)

Per-turn budget-conserving weight (within trajectory τ, indexed by traj_uid):

    |δ_k|     = mean_t |δ_{k,t}| over response_mask[k]
    r_k       = |δ_k| / token-weighted-mean_j |δ_j|_τ
    w_k       = r_k / (Σ_j r_j N_j / Σ_j N_j)

Conservation invariant (per τ, AUDITED on every call):

    Σ_k w_k · N_k ≡ Σ_k N_k

Loss:

    L = - agg_t ( w_k · δ_t · log π_student(y_t) )

    (gradient flows ONLY through student log_prob; δ_t and w_k are detached.)

The actor combines this dense objective with its ordinary GRPO loss:

    policy_loss = pg_loss + clamp( dense_coef · L , ±α · |pg_loss| )

with hard-cutoff warmup (first N steps dense_coef=0).
"""

from collections import defaultdict
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import agg_loss


def _accord_concord_metrics(delta_clip, adv, response_mask, eps):
    """ACCORD per-token concordance diagnostics (no_grad, vectorized, cheap).

    Quadrants over response tokens by (sign δ, sign A):
      Q1 δ+A+ concord-up (kept)  | Q2 δ-A+ conflict (gated off)
      Q3 δ-A- concord-down(kept) | Q4 δ+A- conflict (gated off)
    Reward-silent = |A|<=eps (also gated under the safe default). ~7 scalars,
    no per-token dump → does not bloat the console log line.
    """
    m = response_mask
    msum = m.sum().clamp(min=1)
    dpos = delta_clip > 0
    dneg = delta_clip < 0
    apos = adv > eps
    aneg = adv < -eps
    asil = adv.abs() <= eps
    q1 = ((dpos & apos).to(m.dtype) * m).sum() / msum
    q2 = ((dneg & apos).to(m.dtype) * m).sum() / msum
    q3 = ((dneg & aneg).to(m.dtype) * m).sum() / msum
    q4 = ((dpos & aneg).to(m.dtype) * m).sum() / msum
    silent = (asil.to(m.dtype) * m).sum() / msum
    gap_raw = (delta_clip * m).sum() / msum
    gap_abs_raw = (delta_clip.abs() * m).sum() / msum
    return {
        "concord/q1_up_keep": q1.item(),
        "concord/q2_conflict_gated": q2.item(),
        "concord/q3_down_keep": q3.item(),
        "concord/q4_conflict_gated": q4.item(),
        "concord/gate_on_frac": (q1 + q3).item(),
        "concord/conflict_frac": (q2 + q4).item(),
        "concord/reward_silent_frac": silent.item(),
        "concord/gap_mean_raw": gap_raw.item(),
        "concord/gap_abs_raw": gap_abs_raw.item(),
    }


def _adv_alignment_metrics(delta_clip, adv, response_mask, eps):
    """Plain TRIAL diagnostics: sign alignment between delta and GRPO advantage."""
    m = response_mask
    msum = m.sum().clamp(min=1)
    dpos = delta_clip > 0
    dneg = delta_clip < 0
    apos = adv > eps
    aneg = adv < -eps
    asil = adv.abs() <= eps
    q1 = ((dpos & apos).to(m.dtype) * m).sum() / msum
    q2 = ((dneg & apos).to(m.dtype) * m).sum() / msum
    q3 = ((dneg & aneg).to(m.dtype) * m).sum() / msum
    q4 = ((dpos & aneg).to(m.dtype) * m).sum() / msum
    silent = (asil.to(m.dtype) * m).sum() / msum
    dot = (delta_clip * adv * m).sum() / msum
    return {
        "hindsight/adv_q1_delta_pos_adv_pos": q1.item(),
        "hindsight/adv_q2_delta_neg_adv_pos": q2.item(),
        "hindsight/adv_q3_delta_neg_adv_neg": q3.item(),
        "hindsight/adv_q4_delta_pos_adv_neg": q4.item(),
        "hindsight/adv_concord_frac": (q1 + q3).item(),
        "hindsight/adv_conflict_frac": (q2 + q4).item(),
        "hindsight/adv_silent_frac": silent.item(),
        "hindsight/adv_delta_dot_mean": dot.item(),
    }


def compute_dense_hindsight_loss(
    student_log_probs: torch.Tensor,
    hindsight_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gap_clip: float = 2.0,
    turn_weight: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    advantages: Optional[torch.Tensor] = None,
    concord_gate: bool = False,
    concord_eps: float = 0.0,
):
    """TRIAL auxiliary loss.

    Args:
        student_log_probs: (bs, R) log π_student over chosen response tokens.
            Gradient flows through this.
        hindsight_log_probs: (bs, R) log π_teacher over the SAME chosen tokens,
            evaluated under the hindsight-augmented teacher prompt. Detached.
        response_mask:    (bs, R) valid response token mask (1 inside response,
            0 elsewhere — matches GRPO's response_mask).
        gap_clip:           per-token clip on δ_t (default 2.0).
        turn_weight:      OPTIONAL (bs, R) per-token weights from
            build_relative_turn_profile (broadcast from per-sample). When None,
            runs as vanilla Hindsight (uniform turn weight = 1).
        loss_agg_mode:    aggregation mode (token-mean / seq-mean / etc),
            forwarded to verl's agg_loss.

    Returns:
        loss:    scalar tensor, gradient flows through student_log_probs only.
        metrics: diagnostic dict.
    """
    hindsight_log_probs = hindsight_log_probs.detach()
    with torch.no_grad():
        aux_token_count = response_mask.sum()
        active_row_fraction = (
            (response_mask.sum(dim=-1) > 0).float().mean().item()
            if response_mask.ndim == 2 and response_mask.shape[0] > 0
            else 0.0
        )
        active_token_fraction = (
            response_mask.bool().float().mean().item()
            if response_mask.numel() > 0
            else 0.0
        )
    if response_mask.sum().item() <= 0:
        zero_loss = student_log_probs.sum() * 0.0
        return zero_loss, {
            "hindsight/loss": 0.0,
            "hindsight/gap_mean": 0.0,
            "hindsight/gap_abs_mean": 0.0,
            "hindsight/gap_std": 0.0,
            "hindsight/positive_frac": 0.0,
            "hindsight/empty_aux_mask": 1.0,
            "hindsight/active_row_fraction": active_row_fraction,
            "hindsight/active_token_fraction": active_token_fraction,
            "hindsight/active_token_count": 0.0,
        }

    # δ_t = teacher_lp - student_lp. Stop-gradient: δ is a coefficient, not a
    # learning target.
    delta_t = hindsight_log_probs - student_log_probs.detach()
    if gap_clip is not None and gap_clip > 0:
        delta_t = delta_t.clamp(-gap_clip, +gap_clip)
    delta_t = delta_t.detach()

    # TRIAL per-turn budget-conserving reweight (multiplicative, gradient-free).
    delta_clip = delta_t  # clipped δ (pre-turn_weight) — sign structure for the gate/diag
    if turn_weight is not None:
        assert turn_weight.shape == delta_t.shape, (
            f"turn_weight shape {tuple(turn_weight.shape)} != "
            f"delta_t shape {tuple(delta_t.shape)}"
        )
        delta_t = (delta_t * turn_weight.detach()).detach()

    # === ACCORD concord gate (default OFF → byte-identical to plain TRIAL) ===
    # Keep the auxiliary signal ONLY on tokens where the hindsight direction
    # sign(δ_t) AGREES with the empirical advantage direction sign(A_t).
    # Conflict tokens (sign product ≤ 0) and reward-silent tokens (|A|≤eps)
    # get aux=0 and fall back to pure GRPO. On surviving tokens sign(δ)=sign(A),
    # so the aux gradient is co-directional with GRPO ⟹ structural floor = GRPO.
    accord_metrics = {}
    alignment_metrics = {}
    if advantages is not None:
        adv = advantages.detach()
        with torch.no_grad():
            alignment_metrics = _adv_alignment_metrics(delta_clip, adv, response_mask, concord_eps)
    if concord_gate and advantages is not None:
        with torch.no_grad():
            accord_metrics = _accord_concord_metrics(delta_clip, adv, response_mask, concord_eps)
            gate = (
                (torch.sign(delta_clip) * torch.sign(adv) > 0)
                & (adv.abs() > concord_eps)
            ).to(delta_t.dtype)
        delta_t = (delta_t * gate).detach()

    # Advantage-weighted policy gradient: -δ_t · log π_student.
    # Gradient flows through student_log_probs.
    weighted = delta_t * student_log_probs

    # Aggregate over response tokens (uses verl's agg_loss for consistency
    # with GRPO's loss_agg_mode handling).
    loss = -agg_loss(loss_mat=weighted, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gap_mean = (delta_t * response_mask).sum() / mask_sum
        gap_abs_mean = (delta_t.abs() * response_mask).sum() / mask_sum
        positive_frac = ((delta_t > 0).float() * response_mask).sum() / mask_sum
        gap_var = ((delta_t - gap_mean) ** 2 * response_mask).sum() / mask_sum
        gap_std = gap_var.sqrt()

    metrics = {
        "hindsight/loss": loss.detach().item(),
        "hindsight/gap_mean": gap_mean.item(),
        "hindsight/gap_abs_mean": gap_abs_mean.item(),
        "hindsight/gap_std": gap_std.item(),
        "hindsight/positive_frac": positive_frac.item(),
        "hindsight/empty_aux_mask": 0.0,
        "hindsight/active_row_fraction": active_row_fraction,
        "hindsight/active_token_fraction": active_token_fraction,
        "hindsight/active_token_count": aux_token_count.item(),
    }
    metrics.update(alignment_metrics)
    metrics.update(accord_metrics)
    return loss, metrics


def apply_turn_profile_control(
    batch,
    metrics: dict,
    mode: str,
    *,
    permutation_seed: int = 0,
) -> dict:
    """Select the applied turn weights without changing the hindsight signal."""
    mode = str(mode).strip().lower()
    if mode != "relative":
        raise ValueError(
            "This release implements only the relative TRIAL turn profile; "
            f"got {mode!r}."
        )
    if "turn_weight" not in batch.batch:
        raise KeyError("TRIAL turn-weight selection requires batch.batch['turn_weight'].")

    metrics = dict(metrics)
    metrics["trial/turn_profile_relative"] = float(mode == "relative")
    metrics["trial/turn_profile_uniform"] = float(mode == "uniform")
    metrics["trial/turn_profile_permuted"] = float(mode == "permuted")
    if mode == "relative":
        return metrics

    # Preserve diagnostics for the candidate gap allocation while applying a
    # unit weight to the exact same signed-delta auxiliary objective.
    for name in ("w_mean", "w_std", "w_std_active", "active_frac", "w_min", "w_max"):
        key = f"trial/{name}"
        if key in metrics:
            metrics[f"trial/candidate_{name}"] = metrics[key]

    candidate = batch.batch["turn_weight"]
    if mode == "permuted":
        response_mask = batch.batch["response_mask"]
        aux_mask = batch.batch.get("hindsight_token_mask", response_mask)
        aux_mask = aux_mask.to(device=response_mask.device, dtype=response_mask.dtype)
        aux_mask = aux_mask * response_mask
        token_counts = aux_mask.sum(dim=-1).float()
        traj_uid = batch.non_tensor_batch.get("traj_uid", None)
        if traj_uid is None:
            raise KeyError("Permuted turn allocation requires non_tensor_batch['traj_uid'].")

        # Candidate weights are constant within each response row. Recover the
        # scalar robustly, then permute priorities only within each trajectory.
        denom = response_mask.sum(dim=-1).clamp(min=1)
        scalar_weights = (candidate * response_mask).sum(dim=-1) / denom
        applied_scalars = scalar_weights.clone()
        groups = defaultdict(list)
        for i in range(candidate.size(0)):
            uid = str(traj_uid[i]) if not isinstance(traj_uid[i], str) else traj_uid[i]
            groups[uid].append(i)

        permuted_groups = 0
        fixed_points = 0
        audit_max_resid = 0.0
        for uid, idxs in groups.items():
            active_idxs = [i for i in idxs if token_counts[i].item() > 0]
            if len(active_idxs) <= 1:
                continue
            idx_t = torch.tensor(active_idxs, dtype=torch.long, device=candidate.device)
            n_group = token_counts[idx_t]
            w_group = scalar_weights[idx_t]

            import hashlib

            digest = hashlib.sha256(f"{int(permutation_seed)}:{uid}".encode("utf-8")).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
            identity = torch.arange(len(active_idxs), dtype=torch.long)
            permutation = None
            for _ in range(32):
                proposal = torch.randperm(len(active_idxs), generator=generator)
                if not torch.any(proposal == identity):
                    permutation = proposal
                    break
            if permutation is None:
                permutation = torch.roll(identity, shifts=1)
            fixed_points += int((permutation == identity).sum().item())

            shuffled = w_group[permutation.to(device=w_group.device)]
            target_budget = n_group.sum()
            shuffled_budget = (shuffled * n_group).sum()
            if shuffled_budget.abs().item() > 1e-8:
                shuffled = shuffled * (target_budget / shuffled_budget)
            applied_scalars[idx_t] = shuffled

            actual = (shuffled * n_group).sum().item()
            target = target_budget.item()
            resid = abs(actual - target) / max(target, 1e-9)
            audit_max_resid = max(audit_max_resid, resid)
            if resid > 1e-5:
                raise AssertionError(
                    f"Permuted TRIAL budget audit failed for traj {uid!r}: "
                    f"relative residual {resid:.3e}."
                )
            permuted_groups += 1

        applied = applied_scalars.unsqueeze(-1).expand_as(candidate).contiguous()
        batch.batch["turn_weight"] = applied
        active_mask = (applied_scalars - 1.0).abs() > 1e-8
        active_values = applied_scalars[active_mask]
        metrics.update(
            {
                "trial/w_mean": applied_scalars.mean().item(),
                "trial/w_std": applied_scalars.std(unbiased=False).item()
                if applied_scalars.numel() > 1
                else 0.0,
                "trial/w_std_active": active_values.std(unbiased=False).item()
                if active_values.numel() > 1
                else 0.0,
                "trial/active_frac": active_mask.float().mean().item(),
                "trial/w_min": applied_scalars.min().item(),
                "trial/w_max": applied_scalars.max().item(),
                "trial/permuted_groups": float(permuted_groups),
                "trial/permutation_fixed_points": float(fixed_points),
                "trial/permuted_budget_max_relative_residual": audit_max_resid,
                "trial/permutation_seed": float(permutation_seed),
            }
        )
        return metrics

    batch.batch["turn_weight"] = torch.ones_like(candidate)
    metrics.update(
        {
            "trial/w_mean": 1.0,
            "trial/w_std": 0.0,
            "trial/w_std_active": 0.0,
            "trial/active_frac": 0.0,
            "trial/w_min": 1.0,
            "trial/w_max": 1.0,
        }
    )
    return metrics


def build_relative_turn_profile(
    batch,
    *,
    gap_clip: float,
    audit_tol: float = 1e-3,
    informativeness: str = "abs_delta",
    use_group_success_gate: bool = False,
    aux_mask_key: str = "response_mask",
) -> dict:
    """Compute Δ-weighted turn-budgets and stash on batch.batch['turn_weight'].

    Pre-conditions on `batch.batch` / `batch.non_tensor_batch`:
      - response_mask, old_log_probs, hindsight_log_probs (batch-level forward)
      - non_tensor_batch["traj_uid"]
      - token_level_scores (only if use_group_success_gate=True)
    Advantages NOT required — TRIAL is advantage-free by design.

    Args:
      informativeness: per-token signal aggregated per turn.
        "abs_delta" — |teacher_lp(y_t) - student_lp(y_t)| averaged over response_mask.
        "k3_kl"    — Schulman k3: clamp(exp(δ) - δ - 1, 0, 10).
      use_group_success_gate: if True, multiply per-trajectory weights by an
        additional group-level saturation gate. Groups where all trajectories
        fail (no positive reward) OR all succeed are filtered to zero; mixed
        groups retain full TRIAL contribution. DAPO-style two-sided gating.
        GRPO step unaffected.

    Post-condition:
      batch.batch["turn_weight"] = (bs, R) tensor (per-sample, broadcast).

    Returns: TRIAL diagnostic metrics dict.
    """
    response_mask = batch.batch["response_mask"]
    aux_mask = batch.batch.get(aux_mask_key, response_mask)
    if aux_mask.shape != response_mask.shape:
        raise ValueError(
            f"TRIAL aux mask {aux_mask_key!r} shape {tuple(aux_mask.shape)} "
            f"!= response_mask shape {tuple(response_mask.shape)}"
        )
    aux_mask = aux_mask.to(device=response_mask.device, dtype=response_mask.dtype) * response_mask
    student_lp = batch.batch["old_log_probs"]
    teacher_lp = batch.batch["hindsight_log_probs"]
    traj_uid = batch.non_tensor_batch.get("traj_uid", None)
    if traj_uid is None:
        return {"trial/skipped": 1.0, "trial/skipped_reason_no_traj_uid": 1.0}

    # Step 1: per-token δ, clipped, masked.
    delta_t = (teacher_lp - student_lp).detach()
    if gap_clip is not None and gap_clip > 0:
        delta_t = delta_t.clamp(-gap_clip, +gap_clip)
    delta_t = delta_t * aux_mask

    if informativeness == "k3_kl":
        info_t = (delta_t.exp() - delta_t - 1.0).clamp(min=0.0, max=10.0)
        info_t = info_t * aux_mask
        info_per_turn = info_t.sum(dim=-1) / aux_mask.sum(dim=-1).clamp(min=1)
    elif informativeness == "abs_delta":
        info_per_turn = delta_t.abs().sum(dim=-1) / aux_mask.sum(dim=-1).clamp(min=1)
    else:
        raise ValueError(f"Unknown informativeness measure: {informativeness!r}")

    n_per_turn = aux_mask.sum(dim=-1)
    abs_delta_per_turn = info_per_turn
    bs = abs_delta_per_turn.size(0)

    # Group sample indices by trajectory uid.
    groups = defaultdict(list)
    for i in range(bs):
        uid = str(traj_uid[i]) if not isinstance(traj_uid[i], str) else traj_uid[i]
        groups[uid].append(i)

    if use_group_success_gate:
        if "token_level_scores" in batch.batch:
            traj_rewards = batch.batch["token_level_scores"].detach().sum(dim=-1)
        else:
            raise RuntimeError(
                "TRIAL use_group_success_gate=True requires "
                "batch.batch['token_level_scores'] populated by reward step."
            )
        # Task-level grouping for the gate. `uid` is per-task (env.rollout.n
        # trajectories share the same uid); `traj_uid` is per-trajectory and
        # must NOT be used here — within one trajectory all turns carry the
        # same final reward, so all_pos == has_pos by construction and the
        # gate would mask EVERY trajectory. The DAPO-style filter only makes
        # sense across the n rollouts of one task. See [[satgate-grouping-bug]].
        task_uid = batch.non_tensor_batch.get("uid", None)
        if task_uid is None:
            raise RuntimeError(
                "TRIAL use_group_success_gate=True requires "
                "non_tensor_batch['uid'] (task-level grouping uid)."
            )
        # Build per-task eta via a VARIANCE gate (plan C): a task group carries
        # NO GRPO learning signal iff all its rollouts got the same reward
        # (std == 0). This is the exact continuous generalisation of DAPO's
        # "accuracy 0 or 1" dynamic-sampling filter, and matches the
        # group-relative advantage A_i = (r_i - mean)/(std + eps) being
        # identically zero. For binary alfworld reward {0, R} this reduces to
        # the all-fail / all-success test; for webshop's [0,1] partial credit
        # it additionally KEEPS all-positive-but-varied groups (which DO carry
        # signal) instead of wrongly treating them as saturated all-success.
        std_tol = 1e-6
        task_groups = defaultdict(list)
        for i in range(bs):
            tu = str(task_uid[i]) if not isinstance(task_uid[i], str) else task_uid[i]
            task_groups[tu].append(i)
        eta_by_task = {}
        n_task_gated_allfail = 0
        n_task_gated_allpos = 0
        n_task_kept = 0
        for tu, t_idxs in task_groups.items():
            if len(t_idxs) <= 1:
                # Singleton task → no variance defined, keep aux on.
                eta_by_task[tu] = 1.0
                n_task_kept += 1
                continue
            r = traj_rewards[torch.tensor(t_idxs, dtype=torch.long, device=traj_rewards.device)]
            r_std = r.std(unbiased=False).item()
            if r_std <= std_tol:
                eta_by_task[tu] = 0.0
                if r.mean().item() <= std_tol:
                    n_task_gated_allfail += 1
                else:
                    n_task_gated_allpos += 1
            else:
                eta_by_task[tu] = 1.0
                n_task_kept += 1
    else:
        traj_rewards = None
        task_uid = None
        eta_by_task = None

    eps = 1e-8
    w_per_sample = torch.ones(bs, dtype=torch.float32, device=abs_delta_per_turn.device)
    n_singleton = 0
    n_active_groups = 0
    n_gated_groups = 0
    n_success_gate_on = 0
    audit_max_resid = 0.0

    for uid, idxs in groups.items():
        active_idxs = [i for i in idxs if n_per_turn[i].item() > 0]
        if len(active_idxs) <= 1:
            n_singleton += 1
            continue

        idx_t = torch.tensor(active_idxs, dtype=torch.long, device=abs_delta_per_turn.device)
        d_group = abs_delta_per_turn[idx_t]
        n_group = n_per_turn[idx_t].float()

        # r_k = |δ_k| / token-weighted-mean(|δ|_τ).
        d_total = (d_group * n_group).sum()
        n_total = n_group.sum()
        if d_total < eps:
            continue
        d_mean_weighted = d_total / n_total
        r = d_group / (d_mean_weighted + eps)

        # Token-weighted normalization for EXACT budget conservation
        # Σ w_k N_k ≡ Σ N_k.
        scale = (r * n_group).sum() / n_total
        w_group = r / scale

        # AUDIT.
        actual = (w_group * n_group).sum().item()
        target = n_total.item()
        resid = abs(actual - target) / max(target, 1e-9)
        audit_max_resid = max(audit_max_resid, resid)
        if resid > audit_tol:
            raise AssertionError(
                f"TRIAL AUDIT failed for traj {uid!r}: "
                f"Σ w_k N_k = {actual:.6f}, Σ N_k = {target:.6f}, "
                f"relative residual = {resid:.3e} > {audit_tol:.3e}"
            )

        # D2 task-level variance gate (post-audit, applied as a separate
        # multiplier). eta_by_task is 0 for zero-variance task groups (no GRPO
        # learning signal: binary all-fail/all-success, or continuous all-equal)
        # and 1 otherwise. Gate is task-level — NOT trajectory-level — because
        # within one trajectory the final reward is constant across turns.
        if eta_by_task is not None:
            # All trajectories grouped under this traj_uid share the same
            # task_uid (one trajectory belongs to exactly one task), so look
            # up via idxs[0].
            tu = str(task_uid[active_idxs[0]]) if not isinstance(task_uid[active_idxs[0]], str) else task_uid[active_idxs[0]]
            eta_group = eta_by_task[tu]
            if eta_group < 1.0:
                w_group = w_group * eta_group
                n_gated_groups += 1
            else:
                n_success_gate_on += 1

        n_active_groups += 1
        w_per_sample[idx_t] = w_group

    active_mask = (w_per_sample - 1.0).abs() > 1e-8
    if active_mask.sum().item() > 1:
        w_active = w_per_sample[active_mask]
        w_std_active = w_active.std(unbiased=False).item()
        active_frac = active_mask.float().mean().item()
    else:
        w_std_active = 0.0
        active_frac = active_mask.float().mean().item() if bs > 0 else 0.0

    turn_weight = w_per_sample.unsqueeze(-1).expand_as(response_mask).contiguous()
    batch.batch["turn_weight"] = turn_weight

    with torch.no_grad():
        valid_tokens = response_mask.sum().clamp(min=1)
        aux_tokens = aux_mask.sum()
        aux_rows = (aux_mask.sum(dim=-1) > 0).float()

    metrics = {
        "trial/w_mean": w_per_sample.mean().item(),
        "trial/w_std": w_per_sample.std(unbiased=False).item() if bs > 1 else 0.0,
        "trial/w_std_active": w_std_active,
        "trial/active_frac": active_frac,
        "trial/w_min": w_per_sample.min().item(),
        "trial/w_max": w_per_sample.max().item(),
        "trial/active_groups": int(n_active_groups),
        "trial/total_groups": int(len(groups)),
        "trial/singleton_groups": int(n_singleton),
        "trial/audit_max_relative_residual": audit_max_resid,
        "trial/info_signal_mean": abs_delta_per_turn.mean().item(),
        "trial/info_signal_max": abs_delta_per_turn.max().item(),
        "trial/aux_mask_token_fraction": (aux_tokens / valid_tokens).item(),
        "trial/aux_mask_row_fraction": aux_rows.mean().item() if bs > 0 else 0.0,
        "trial/aux_mask_empty_row_fraction": (1.0 - aux_rows).mean().item() if bs > 0 else 0.0,
    }
    if traj_rewards is not None:
        eligible_groups = max(n_active_groups, 1)
        metrics["trial/d2_gate_on_groups"] = int(n_success_gate_on)
        metrics["trial/d2_gate_off_groups"] = int(n_gated_groups)
        metrics["trial/d2_eta_active_frac"] = float(n_success_gate_on) / float(eligible_groups)
        metrics["trial/d2_traj_reward_mean"] = traj_rewards.mean().item()
        metrics["trial/d2_traj_reward_max"] = traj_rewards.max().item()
        metrics["trial/d2_pos_reward_frac"] = (traj_rewards > 0).float().mean().item()
        # Task-level (env.rollout.n grouping) variance-gate breakdown — answers
        # "which side (all-fail vs all-success) is being gated off". Under the
        # variance gate, n_task_kept = tasks with reward spread (GRPO signal);
        # the two gated_* counters split the zero-variance tasks by whether the
        # shared reward was ~0 (all-fail) or >0 (all-success).
        metrics["trial/d2_task_total"] = int(len(task_groups))
        metrics["trial/d2_task_kept"] = int(n_task_kept)
        metrics["trial/d2_task_gated_allfail"] = int(n_task_gated_allfail)
        metrics["trial/d2_task_gated_allpos"] = int(n_task_gated_allpos)
    return metrics
