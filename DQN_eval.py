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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="DQN", help="The RLlib-registered algorithm to use."
)
parser.add_argument("--num-cpus", type=int, default=1)

parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=1000,
    help="Number of timesteps to test.",
)


parser.add_argument(
    "--model-dir", type=str, required=True, help="path to the RL model for evaluation"
)
parser.add_argument(
    "--save-dir", type=str, required=True, help="folder directory for saving evaluation results"
)
parser.add_argument(
    "--rv-rate", type=float, default=1.0, help="RV percentage. 0.0-1.0"
)
parser.add_argument(
    "--explore-during-inference",
    action="store_true",
    help="Whether the trained policy should use exploration during action "
    "inference.",
)
if __name__ == "__main__":
    args = parser.parse_args()

    # ray.init(num_cpus=args.num_cpus or None)
    ray.init(local_mode= True)

    rv_rate = args.rv_rate

    from onnx_inference import MixedTrafficControlInference
    model_path = '/home/david/code/MixedTrafficControl/onnx/1.0_model.onnx'
    decider = MixedTrafficControlInference(model_path)

    checkpoint_path = args.model_dir
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    ## TODO map xml could be parsed from sumocfg file
    env = Env({
            "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":rv_rate,
            "cfg":'real_data/osm.sumocfg',
            "render":True,
            "map_xml":'real_data/CSeditClean_1.net_threelegs.xml',
            "max_episode_steps":args.stop_timesteps,
            "conflict_mechanism":'off',
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0
            }
        })

    episode_reward = 0
    dones = truncated = {}
    dones['__all__'] = truncated['__all__'] = False


    obs, info = env.reset()
    
    while not dones['__all__'] and not truncated['__all__']:
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = algo.compute_single_action(agent_obs, explore=args.explore_during_inference ,policy_id="shared_policy")
            actions[agent_id] = decider.forward(agent_obs)
        obs, reward, dones, truncated, info = env.step(actions)
        for key, done in dones.items():
            if done:
                obs.pop(key)
        if dones['__all__']:
            obs, info = env.reset()
            num_episodes += 1
    
    env.monitor.evaluate()
    save_path = args.save_dir+'/'+str(args.rv_rate)+'log.pkl'
    env.monitor.evaluate()
    env.monitor.save_to_pickle(file_name = save_path)
    algo.stop()

    ray.shutdown()