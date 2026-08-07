"""Hydra entry point for TRIAL training."""

import hydra
import ray
from omegaconf import OmegaConf

from trial.core.trainer import RayTRIALTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="trial_trainer", version_base=None)
def main(config) -> None:
    run_trial(config)


def run_trial(config) -> None:
    if not ray.is_initialized():
        runtime_env = OmegaConf.merge(
            get_ppo_ray_runtime_env(), config.get("ray_init", {}).get("runtime_env", {})
        )
        ray_init_kwargs = OmegaConf.create(
            {**config.get("ray_init", {}), "runtime_env": runtime_env}
        )
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TRIALTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TRIALTaskRunner:
    def run(self, config) -> None:
        from pprint import pprint

        from omegaconf import OmegaConf

        from agent_system.environments import make_envs
        from agent_system.multi_turn_rollout import TrajectoryCollector
        from agent_system.reward_manager import EpisodeRewardManager
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        loss_mode = str(config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "")).lower()
        if loss_mode != "trial":
            raise ValueError("TRIAL requires actor_rollout_ref.actor.policy_loss.loss_mode=trial.")
        if str(config.algorithm.adv_estimator).lower() != "grpo":
            raise ValueError("TRIAL uses GRPO as its only outcome objective.")
        if config.actor_rollout_ref.rollout.mode != "sync":
            raise ValueError("TRIAL supports only synchronous rollout.")
        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("TRIAL supports only FSDP/FSDP2 actor training.")

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        envs, val_envs = make_envs(config)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }
        pool_id = "global_pool"
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
            mapping={Role.ActorRollout: pool_id, Role.Critic: pool_id},
        )
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            resource_pool_manager.mapping[Role.RefPolicy] = pool_id

        reward_manager = config.reward_model.get("reward_manager", "episode")
        if reward_manager != "episode":
            raise NotImplementedError(f"Unsupported reward manager: {reward_manager}")
        reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)

        trainer = RayTRIALTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
            device_name=config.trainer.device,
            traj_collector=TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor),
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
