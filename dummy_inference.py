from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
import argparse
import os
import random

import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig, DQNTorchPolicy
from Env import Env
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
    TorchSharedWeightsModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved


class dqn_inference(object):
    def __init__(self, model_path):
        self.algo = Algorithm.from_checkpoint(model_path)
    
    def get_decision(self, obs):
        return self.algo.compute_single_action(obs, explore=False, policy_id="shared_policy")