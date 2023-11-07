"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
import argparse
import numpy as np

import ray
from ray import tune
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomLoggerCallback(DefaultCallbacks):
    def on_episode_start(
            self,
            *,
            worker,
            base_env,
            policies,
            episode,
            env_index = None,
            **kwargs,
        ):
        episode.user_data["conflict_rate"] = []
        episode.user_data["avg_wait"] = []

    def on_episode_step(
            self,
            *,
            worker,
            base_env,
            policies = None,
            episode,
            env_index= None,
            **kwargs,
        ):
        conflict_rate = worker.env.monitor.conflict_rate[-1]
        episode.user_data["conflict_rate"].extend([conflict_rate])
        total_wait = 0
        for id in worker.env.previous_global_waiting.keys():
            total_wait += worker.env.previous_global_waiting[id]['sum']
        episode.user_data["avg_wait"].extend([total_wait])

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index = None,
        **kwargs,
    ):
        episode.custom_metrics["conflict_rate"] = np.mean(episode.user_data["conflict_rate"])
        episode.custom_metrics["avg_wait"] = np.mean(episode.user_data["avg_wait"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=2000)
    args = parser.parse_args()

    ray.init()
    trials = tune.run(
        "PG",
        stop={
            "training_iteration": args.num_iters,
        },
        config={
            "env": "CartPole-v0",
            "callbacks": MyCallbacks,
        },
        return_trials=True)

    # verify custom metrics for integration tests
    custom_metrics = trials[0].last_result["custom_metrics"]
    print(custom_metrics)
    assert "pole_angle_mean" in custom_metrics
    assert "pole_angle_min" in custom_metrics
    assert "pole_angle_max" in custom_metrics
    assert "num_batches_mean" in custom_metrics
    assert "callback_ok" in trials[0].last_result