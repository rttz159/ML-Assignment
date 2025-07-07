from ray.tune.registry import register_env
from glucose_env import CustomGlucoseDynamicsEnv
import ray
from ray.rllib.algorithms.ppo import PPOConfig

def env_creator(env_config):
    """
    Returns a new instance of  CustomGlucoseDynamicsEnv.
    """
    return CustomGlucoseDynamicsEnv()

env_name = "CustomGlucoseDynamicsEnv-v0"
register_env(env_name, env_creator)

ray.init()

num_of_iter = 8

ppo_config = (
    PPOConfig()
    .environment(env_name)
    .resources(num_gpus=1)
    .framework("torch")
    .env_runners(num_env_runners=4)
    .training(
        lr=0.0001,
        train_batch_size_per_learner=8000,
        num_epochs=num_of_iter,
        entropy_coeff=0.05,
    )
    .evaluation(
        evaluation_interval=1, 
        evaluation_duration=num_of_iter,
    )
)

ppo_algo = ppo_config.build()

print("Starting training...")
for i in range(num_of_iter):
    result = ppo_algo.train()
    print(f"Iteration {i} done")

ppo_algo.stop()
ray.shutdown()