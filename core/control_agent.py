import numpy as np
import copy
class control_agent(object):
    def __init__(self, env, yellow_step_length, control_circle_length) -> None:
        self.env = env
        self.yellow_step_length = yellow_step_length
        self.keywords = self.env.keywords_order
        self.junction_list = self.env.junction_list
        self.action_pairs = [[0,1],[2,3],[4,5],[6,7],[0,4],[2,6],[1,5],[3,7]]
        self.control_circle_length = control_circle_length
        self.reset()
        
    def reset_acts(self):
        self.yellow = dict()
        self.acts = dict()
        self.yellow_clock = dict()
        self.last_acts = dict()
        for junc in self.junction_list:
            self.yellow[junc] = False
            self.yellow_clock[junc] = 0
            self.last_acts[junc] = [0,1]
            self.acts[junc] = dict()
            for kw in self.keywords:
                self.acts[junc][kw] = False
        
    def reset(self):
        self._step = 0
        self.reset_acts()

    def set_yellow(self, junc):
        for kw in self.keywords:
            self.acts[junc][kw] = False

    def step(self):

        for junc in self.junction_list:
            if self.yellow[junc]:
                self.set_yellow(junc)
                self.yellow_clock[junc] -= 1
                if self.yellow_clock[junc] == 0:
                    self.yellow[junc] = False
                continue
            else:
                self.acts[junc][self.keywords[self.last_acts[junc][0]]] = True
                self.acts[junc][self.keywords[self.last_acts[junc][1]]] = True

        if self._step % self.control_circle_length == 0:

            for junc in self.junction_list:
                values = []
                for kw in self.keywords:
                    wait = self.env.get_avg_wait_time(junc, kw, 'all')
                    qlen = self.env.get_queue_len(junc, kw, 'all')/(self.env.compute_max_len_of_control_queue(junc)+0.000001)
                    values.extend([wait*qlen])
                after_sort = copy.deepcopy(values)
                after_sort.sort(reverse=True)
                max_idx = values.index(after_sort[0])
                action_idx = -1
                for i in range(1, len(after_sort)):
                    small_idx = values.index(after_sort[i])
                    actp = [max_idx, small_idx]
                    actp.sort()
                    if actp in self.action_pairs:
                        action_idx = self.action_pairs.index(actp)
                        break
                if action_idx == -1 and sum(values)!=0:
                    for j in range(8):
                        if max_idx in self.action_pairs[j]:
                            action_idx = j
                            break
                
                action = self.action_pairs[action_idx]
                if action != self.last_acts[junc]:
                    self.last_acts[junc] = action
                    self.yellow[junc] = True
                    self.yellow_clock[junc] = self.yellow_step_length



        self._step += 1

    def get_result(self, junc, keyword):
        return self.acts[junc][keyword]