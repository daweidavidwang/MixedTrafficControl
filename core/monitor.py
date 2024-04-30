import numpy as np
import pickle
import math


class DataMonitor(object):
    def __init__(self, env) -> None:
        self.junction_list = env.junction_list
        self.keywords_order = env.keywords_order
        self.clear_data()

    def clear_data(self):
        self.conduct_traj_recorder()
        self.conduct_data_recorder()

    def conduct_traj_recorder(self):
        self.traj_record = dict()
        for JuncID in self.junction_list:
            self.traj_record[JuncID] = dict()
            for Keyword in self.keywords_order:
                self.traj_record[JuncID][Keyword] = dict()
        self.max_t = 0
        self.max_x = 0

    def conduct_data_recorder(self):
        self.data_record = dict()
        self.conflict_rate = []
        for JuncID in self.junction_list:
            self.data_record[JuncID] = dict()
            for Keyword in self.keywords_order :
                self.data_record[JuncID][Keyword] = dict()
                self.data_record[JuncID][Keyword]['t'] = [i for i in range(5000)]
                self.data_record[JuncID][Keyword]['queue_wait'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['queue_length'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['control_queue_wait'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['control_queue_length'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['throughput_av'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['throughput'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['throughput_hv'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['conflict'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['global_reward'] = np.zeros(5000)
                self.data_record[JuncID][Keyword]['failure_control_index'] = np.zeros(5000)

    def step(self, env):
        t = env.env_step
        for JuncID in self.junction_list:
            for Keyword in self.keywords_order:
                self.data_record[JuncID][Keyword]['queue_length'][t] = env.get_queue_len(JuncID, Keyword, 'all')
                self.data_record[JuncID][Keyword]['queue_wait'][t] = env.get_avg_wait_time(JuncID, Keyword, 'all')
                self.data_record[JuncID][Keyword]['control_queue_length'][t] = env.get_queue_len(JuncID, Keyword, 'rv')
                self.data_record[JuncID][Keyword]['control_queue_wait'][t] = env.get_avg_wait_time(JuncID, Keyword, 'rv')
                self.data_record[JuncID][Keyword]['throughput'][t] = len(env.inner_lane_newly_enter[JuncID][Keyword])
                self.data_record[JuncID][Keyword]['conflict'][t] = len(env.conflict_vehids)
                self.data_record[JuncID][Keyword]['global_reward'][t] = env.global_obs[JuncID]
        self.conflict_rate.extend(
            [len(env.conflict_vehids)/len(env.previous_action) if len(env.previous_action) else 0]
            )
        ## failure control record
        # total_num_record = 0
        # total_fail_record = 0
        # total_num = 0
        # total_lane = 0
        # for JuncID in env.junction_list:
        #     for EdgeID in env.map.junction_incoming_edges[JuncID]:
        #         total_num = 0
        #         total_lane = 0
        #         for lane_idx in env.map.intersection_edge[EdgeID]['straight']:
        #             dir, label = env.map.qurey_edge_direction(EdgeID, lane_idx)
        #             total_lane += 1
        #             if not env.queue_rv_detect(EdgeID, lane_idx):
        #                 total_num += 1
        #         if label:
        #             keyword = label+dir
        #             self.data_record[JuncID][keyword]['failure_control_index'][t] = total_num/total_lane
        #             total_num_record += total_lane
        #             total_fail_record += total_num
        #         total_num = 0
        #         total_lane = 0
        #         for lane_idx in env.map.intersection_edge[EdgeID]['left']:
        #             dir, label = env.map.qurey_edge_direction(EdgeID, lane_idx)
        #             total_lane += 1
        #             if not env.queue_rv_detect(EdgeID, lane_idx):
        #                 total_num += 1
        #         if label:
        #             keyword = label+dir
        #             self.data_record[JuncID][keyword]['failure_control_index'][t] = total_num/total_lane
        #             total_num_record += total_lane
        #             total_fail_record += total_num

                    
    def evaluate(self, min_step = 500, max_step = 1000):
        total_wait = []
        for JuncID in self.junction_list:
            for keyword in self.keywords_order:
                avg_wait = np.mean(self.data_record[JuncID][keyword]['queue_wait'][min_step:max_step])
                total_wait.extend([avg_wait])
                print("Avg waiting time at" + JuncID +" "+keyword+": "+str(avg_wait))
            print("Total avg wait time at junction "+JuncID+": " +str(np.mean(total_wait)))
        

    def eval_traffic_flow(self, JuncID, time_range):
        inflow_intersection = []
        for t in range(time_range[0], time_range[1]):
            inflow_intersection.extend([0])
            for Keyword in self.keywords_order:
                 inflow_intersection[-1] += self.data_record[JuncID][Keyword]['throughput'][t]
        return inflow_intersection, max(inflow_intersection), sum(inflow_intersection)/len(inflow_intersection)

    def save_to_pickle(self, file_name):
        saved_dict = {'data_record':self.data_record, 'junctions':self.junction_list, 'keyword':self.keywords_order}
        with open(file_name, "wb") as f:
            pickle.dump(saved_dict, f)
