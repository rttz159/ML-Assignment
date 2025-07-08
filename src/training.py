from ray.tune.registry import register_env
from model import GlucoseDynamicsSimulator
from glucose_env import CustomGlucoseDynamicsEnv
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.impala import ImpalaConfig


def env_creator(env_config):
    """
    Returns a new instance of CustomGlucoseDynamicsEnv.
    """
    return CustomGlucoseDynamicsEnv()


env_name = "CustomGlucoseDynamicsEnv-v0"
register_env(env_name, env_creator)

ray.init()

num_of_iter = 8

print("--- Starting Combined RL Training ---")

# PPO Training
print("\n--- Training PPO (Default Hyperparameters) ---")
ppo_config = (
    PPOConfig()
    .environment(env_name)
    # .resources(num_gpus=1)
    # .framework("torch")
    .env_runners(num_env_runners=1)
    .training(
        lr=0.00005,
        train_batch_size_per_learner=4000,
        num_epochs=num_of_iter,
        entropy_coeff=0.0,
        gamma=0.9,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=num_of_iter,
    )
)
ppo_algo = ppo_config.build()
for i in range(num_of_iter):
    result = ppo_algo.train()
    print(f"PPO Iteration {i} done")
ppo_algo.stop()

# SAC Training
print("\n--- Training SAC (Default Hyperparameters) ---")
sac_config = (
    SACConfig()
    .environment(env_name)
    # .resources(num_gpus=1)
    # .framework("torch")
    .env_runners(num_env_runners=1)
    .environment(env_name)
    .training(
        gamma=0.9,
        actor_lr=0.00005,
        critic_lr=0.00005,
        train_batch_size_per_learner=4000,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=num_of_iter,
    )
)
sac_algo = sac_config.build()
for i in range(num_of_iter):
    result = sac_algo.train()
    print(f"SAC Iteration {i} done")
sac_algo.stop()

# IMPALA Training
print("\n--- Training IMPALA (Default Hyperparameters) ---")
impala_config = (
    ImpalaConfig()
    .environment(env_name)
    # .resources(num_gpus=1)
    # .framework("torch")
    .env_runners(num_env_runners=1)
    .training(
        lr=0.00005,
        train_batch_size_per_learner=4000,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=num_of_iter,
    )
)
impala_algo = impala_config.build()
for i in range(num_of_iter):
    result = impala_algo.train()
    print(f"IMPALA Iteration {i} done")
impala_algo.stop()

# Recurrent PPO Training
print("\n--- Training Recurrent PPO (Default Hyperparameters) ---")
recurrent_ppo_config = (
    PPOConfig()
    .environment(env_name)
    # .resources(num_gpus=1)
    # .framework("torch")
    .env_runners(num_env_runners=1)
    .training(
        lr=0.00005,
        train_batch_size_per_learner=4000,
        num_epochs=num_of_iter,
        entropy_coeff=0.0,
        gamma=0.9,
        model={
            "use_lstm": True,
            "lstm_cell_size": 256,
            "max_seq_len": 20,
        },
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=num_of_iter,
    )
)
recurrent_ppo_algo = recurrent_ppo_config.build()
for i in range(num_of_iter):
    result = recurrent_ppo_algo.train()
    print(f"Recurrent PPO Iteration {i} done")
recurrent_ppo_algo.stop()

ray.shutdown()
print("\n--- All training complete ---")
