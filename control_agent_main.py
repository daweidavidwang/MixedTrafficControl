
from copy import deepcopy
import sys, os
from pathlib import Path

sys.path.append(os.getcwd())
from Env import Env

NUM_EVALUATE_EPSIODE = 1000

def main():
    home_dir = '/home/david/code/Multiagent_intersection/'
    env = Env({
            "junction_list":['229','499','332','334'],
            "spawn_rl_prob":{},
            "probablity_RL":1.0,
            "cfg":'real_data/osm_global_routing.sumocfg',
            "render":True,
            "map_xml":'real_data/colorado_global_routing.xml',
            "max_episode_steps":1000,
            "traffic_light_program":{
                "disable_state":'G',
                "disable_light_start":0
            }
        })
    from core.control_agent import control_agent

    for epoch in range(5):
        state, _ = env.reset()
        cagent = control_agent(env, 3, 15)
        for _ in range(0,NUM_EVALUATE_EPSIODE):
            cagent.step()
            action = {}
            for virtual_id in state.keys():
                real_id = env.convert_virtual_id_to_real_id(virtual_id)
                dir, label = env.map.qurey_edge_direction(env.rl_vehicles[real_id].road_id, env.rl_vehicles[real_id].lane_index)
                action[virtual_id] = cagent.get_result(env.map.get_facing_intersection(env.rl_vehicles[real_id].road_id), label+dir)
            
            new_state, reward, is_terminal, truncated, info = env.step(action)
            for key, done in is_terminal.items():
                if done:
                    new_state.pop(key)
            state = new_state
        print("epoch "+str(epoch)+' finished')
    env.close()



if __name__ == '__main__':
    main()
