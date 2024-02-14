from ray.rllib.algorithms.algorithm import Algorithm
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import ray
from ray import air, tune
from ray.rllib.algorithms.dqn import DQNConfig, DQNTorchPolicy

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
from ray.rllib.policy.sample_batch import SampleBatch
# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--model-dir", type=str, required=True, help="path to the RL model for evaluation"
# )

# parser.add_argument(
#     "--save-dir", type=str, required=True, help="path to the saved onnx direction"
# )
# parser.add_argument(
#     "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
# )

class rainbow_model(nn.Module):
    def __init__(self, checkpoint_path):
        super(rainbow_model, self).__init__()
        self.algo = Algorithm.from_checkpoint(checkpoint_path)
        self.policy = self.algo.get_policy('shared_policy')
        self.model = self.policy.model
        
    def forward(self, input_dict, state, seq):
        # input_dict['obs_flat'] = input_dict["obs"]
        ## compute q values in dqn torch policy line 418 
        outputs, state_out = self.model(input_dict, state, seq)
        action_scores, z, support_logits_per_action, logits, probs = \
            self.model.get_q_value_distributions(outputs)
        state_score = self.model.get_state_value(outputs)

        support_logits_per_action_mean = torch.mean(
            support_logits_per_action, dim=1
        )
        support_logits_per_action_centered = (
            support_logits_per_action
            - torch.unsqueeze(support_logits_per_action_mean, dim=1)
        )
        support_logits_per_action = (
            torch.unsqueeze(state_score, dim=1) + support_logits_per_action_centered
        )
        support_prob_per_action = nn.functional.softmax(
            support_logits_per_action, dim=-1
        )
        value = torch.sum(z * support_prob_per_action, dim=-1)
        logits = support_logits_per_action
        probs_or_logits = support_prob_per_action
        return value, logits, probs_or_logits, state, torch.argmax(value)

class MixedTrafficControlInference(object):
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.ort_sess = ort.InferenceSession(onnx_model_path)

    def forward(self, obs):
        input = {'obs': np.expand_dims(obs, 0), 'state_ins': []}
        value, logits, probs_or_logits, state, action = self.ort_sess.run(None, input)
        return action



def export():
    from Env import Env
    # args = parser.parse_args()
    env = Env({
            "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":0.5,
            "cfg":'real_data/osm.sumocfg',
            "render":False,
            "map_xml":'real_data/CSeditClean_1.net_threelegs.xml',
            "max_episode_steps":1000,
            "conflict_mechanism":'off',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0
            }
        })
    # ray.init(num_cpus=args.num_cpus or None)
    ray.init(local_mode= True)

    checkpoint_path = "/home/david/Documents/MixedTrafficControl_models/best_models(July31)/DQN_RV0.5/checkpoint_000500"
    # algo = Algorithm.from_checkpoint(checkpoint_path)
    # algo.export_policy_model('model.onnx', policy_id="shared_policy", onnx=11)
    # algo.stop()

    # ray.shutdown()
    onnx_path = '/home/david/code/MixedTrafficControl/model.onnx'
    model = rainbow_model(checkpoint_path)
    policy_ptr = model.policy
    policy_ptr._lazy_tensor_dict(policy_ptr._dummy_batch)
    # Provide dummy state inputs if not an RNN (torch cannot jit with
    # returned empty internal states list).
    if "state_in_0" not in policy_ptr._dummy_batch:
        policy_ptr._dummy_batch["state_in_0"] = policy_ptr._dummy_batch[
            SampleBatch.SEQ_LENS
        ] = np.array([1.0])
    seq_lens = policy_ptr._dummy_batch[SampleBatch.SEQ_LENS]

    state_ins = []
    i = 0
    while "state_in_{}".format(i) in policy_ptr._dummy_batch:
        state_ins.append(policy_ptr._dummy_batch["state_in_{}".format(i)])
        i += 1
    dummy_inputs = {
        k: policy_ptr._dummy_batch[k]
        for k in policy_ptr._dummy_batch.keys()
        if k != "is_training"
    }

    torch.onnx.export(model, (dummy_inputs, state_ins, seq_lens), onnx_path, \
    export_params=True,
        opset_version=11,do_constant_folding=True,
        input_names=list(dummy_inputs.keys())+ ["state_ins", SampleBatch.SEQ_LENS],\
        output_names=["value", "logits", "probs_or_logits", "state", "action"],dynamic_axes={
                    k: {0: "batch_size"}
                    for k in list(dummy_inputs.keys())
                    + ["state_ins", SampleBatch.SEQ_LENS]
    })


def try_inference(model_path):
    env = Env({
        "junction_list":['229','499','332','334'],
        "spawn_rl_prob":{},
        "probablity_RL":0.5,
        "cfg":'real_data/osm.sumocfg',
        "render":False,
        "map_xml":'real_data/CSeditClean_1.net_threelegs.xml',
        "max_episode_steps":1000,
        "conflict_mechanism":'off',
        "traffic_light_program":{
            "disable_state":'G',
            "disable_light_start":0
        }
    })

    episode_reward = 0
    dones = truncated = {}
    dones['__all__'] = truncated['__all__'] = False
    decider = MixedTrafficControlInference(model_path)

    obs, info = env.reset()

    while not dones['__all__'] and not truncated['__all__']:
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = decider.forward(agent_obs)
        obs, reward, dones, truncated, info = env.step(actions)
        for key, done in dones.items():
            if done:
                obs.pop(key)
        if dones['__all__']:
            obs, info = env.reset()
            num_episodes += 1

if __name__ == "__main__":
    try_inference('model.onnx')