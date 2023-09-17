from collections import OrderedDict, defaultdict, Counter
import sys
import numpy as np
from core.utils import DIR_KEYWORDS
version = sys.version_info

class Dict(dict if version.major == 3 and version.minor >= 6 else OrderedDict):
    def __add__(self, d):
        return Dict(**self).merge(d)

    def merge(self, *dicts, **kwargs):
        for d in dicts:
            self.update(d)
        self.update(kwargs)
        return self

    def filter(self, keys):
        try: # check for iterable
            keys = set(keys)
            return Dict((k, v) for k, v in self.items() if k in keys)
        except TypeError: # function key
            f = keys
            return Dict((k, v) for k, v in self.items() if f(k, v))

    def map(self, mapper):
        if callable(mapper): # function mapper
            return Dict((k, mapper(v)) for k, v in self.items())
        else: # dictionary mapper
            return Dict((k, mapper[v]) for k, v in self.items())

class Namespace(Dict):
    def __init__(self, *args, **kwargs):
        self.var(*args, **kwargs)

    def var(self, *args, **kwargs):
        kvs = Dict()
        for a in args:
            if isinstance(a, str):
                kvs[a] = True
            else: # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.update(kvs)
        return self

    def unvar(self, *args):
        for a in args:
            self.pop(a)
        return self

    def setdefaults(self, *args, **kwargs):
        args = [a for a in args if a not in self]
        kwargs = {k: v for k, v in kwargs.items() if k not in self}
        return self.var(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            self.__getattribute__(key)

    def __setattr__(self, key, value):
        self[key] = value


class Container(Namespace):
    def __iter__(self):
        return iter(self.values())

class Entity(Namespace):
    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return self.id

    def __repr__(self):
        inner_content = Namespace.format(dict(self)).replace('\n', '\n  ').rstrip(' ')
        return f"{type(self).__name__}('{self.id}',\n  {inner_content})\n\n"

class Vehicle(Entity):
    def leader(self, use_edge=False, use_route=True, max_dist=np.inf):
        try:
            return next(self.leaders(use_edge, use_route, max_dist, 1))
        except StopIteration:
            return None, 0

    def leaders(self, use_edge=False, use_route=True, max_dist=np.inf, max_count=np.inf):
        ent, i = (self.edge, self.edge_i) if use_edge else (self.lane, self.lane_i)
        route = self.route if use_route else None
        for veh, dist in ent.next_vehicles_helper(i + 1, route, max_dist + self.laneposition, max_count):
            yield veh, dist - self.laneposition

    def follower(self, use_edge=False, use_route=True, max_dist=np.inf):
        try:
            return next(self.followers(use_edge, use_route, max_dist, 1))
        except StopIteration:
            return None, 0

    def followers(self, use_edge=False, use_route=True, max_dist=np.inf, max_count=np.inf):
        ent, i = (self.edge, self.edge_i) if use_edge else (self.lane, self.lane_i)
        route = self.route if use_route else None
        for veh, dist in ent.prev_vehicles_helper(i - 1, route, max_dist - self.laneposition, max_count):
            yield veh, dist + self.laneposition

class JunctionQueue(Container):
    def __init__(self, *args, **kwargs):
        self.keywords = DIR_KEYWORDS
        super().__init__(*args, **kwargs)
        for i, key in enumerate(self.keywords):
            self[key] = Queue(id=str(self.edges[int(i/2)])+key,edge_id=self.edges[int(i/2)], key='left' if key[-4]=='Left' else 'straight', netmap=self.netmap, vehicle_len=self.vehicle_len)

    def update_all(self,vehicles):
        for key in self.keywords:
            self[key].update_queue(vehicles)

    def get_density_obs(self):
        obs = []
        for key in self.keywords:
            obs.extend([self[key].fleet_density()])
        return obs

class Queue(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lane_num = len(self.netmap.intersection_edge[self.edge_id]['left']) + len(self.netmap.intersection_edge[self.edge_id]['straight'])
        self.lane_num_in_queue = len(self.netmap.intersection_edge[self.edge_id][self.key])
        self.max_veh_cap = self.netmap.edge_length(self.edge_id)/self.vehicle_len
        self.lane_idxs = self.netmap.intersection_edge[self.edge_id][self.key]
        self.queue = Container()
    
    def update_queue(self, vehicles):
        self.queue = Container()
        self.right_queue = Container()
        for veh_id, veh in vehicles.items():
            if veh.road_id == self.edge_id and veh.lane_index in self.lane_idxs:
                self.queue[veh_id] = veh
    
    def get_queue_len(self):
        return len(self.queue)
    
    def fleet_density(self):
        return (float(self.get_queue_len())/self.lane_num_in_queue)/self.max_veh_cap