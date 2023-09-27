import sys, os
sys.path.append(os.getcwd())
from core.utils import map_parser
from copy import deepcopy
import math
    
class NetMap(object):
    def __init__(self, xml_path, junction_list = []) -> None:
        self.junction_list = junction_list

        self.xml_path = xml_path
        self.net_data, self.connection_data, self.junction_data = map_parser(xml_path)
        self.intersection_edge, self.junction_incoming_edges = self._compute_turning_map() 
        ## intersection edge: the nearest edges to the intersection
        ## intersection_edge_recursive: sorted by edge id, the recursive incoming edges 
        ## junction incoming edges: organized by junction id, the nearest incoming edges for each junction
        ## junction incoming edges recursive: organized by junction id, the recursive incoming edges for each junction
        self.intersection_edge_recursive = self._compute_recursion_incoming(recurse_step=3)
        self._keyword_check()


    def _keyword_check(self):
        for _, edge in self.intersection_edge.items():
            if 'straight' not in edge:
                edge['straight'] = []
            if 'left' not in edge:
                edge['left'] = []
            if 'right' not in edge:
                edge['right'] = []
        
        for _, edge in self.intersection_edge_recursive.items():
            if 'straight' not in edge:
                edge['straight'] = []
            if 'left' not in edge:
                edge['left'] = []
            if 'right' not in edge:
                edge['right'] = []


    def _add_recurse_intersection_edge(self, edge_id, junc_id, recurse_edge_list):
        ## TODO change recurse function to input and return based
        ## input: current edge, output:prev edge(s), this function will add prev edges in the intersection edge dict
        def query_turning(edge_id, lane_id, edge_list):
            if 'straight' in edge_list[edge_id] and lane_id in edge_list[edge_id]['straight']:
                return 'straight'
            elif 'left' in edge_list[edge_id] and lane_id in edge_list[edge_id]['left']:
                return 'left'
            else:
                return 'unidentified'
        results = []
        returned_edges = []
        for lane_id in range(self.net_data[edge_id]['numlane']):
            turning = query_turning(edge_id, lane_id, recurse_edge_list)
            label = recurse_edge_list[edge_id]['edge_label']
            prev = self.prev_edge(edge_id, lane_id)
            for idx in range(len(prev)):
                prev[idx] = prev[idx]+(turning,)+(label,)
            results.extend(prev)
        for r in results:
            # update data in self.intersection_edge
            if not self.net_data[r[0]]['lanes'][r[0]+'_'+str(r[1])]['type']:
                # check and add to returned array
                if r[0] not in returned_edges:
                    returned_edges.extend([r[0]])
                # check if edge name exists in self.intersection_edge
                if r[0] not in recurse_edge_list.keys():
                    recurse_edge_list[r[0]] = dict()
                if 'unidentified' not in recurse_edge_list[r[0]].keys():
                    recurse_edge_list[r[0]]['unidentified'] = [r[1]]
                else:
                    recurse_edge_list[r[0]]['unidentified'].extend([r[1]])
                if r[2] not in recurse_edge_list[r[0]].keys():
                    recurse_edge_list[r[0]][r[2]] = [r[1]]
                else:
                    recurse_edge_list[r[0]][r[2]].extend([r[1]])
                recurse_edge_list[r[0]]['junction'] = junc_id
                recurse_edge_list[r[0]]['edge_label'] = r[3]
        return returned_edges, recurse_edge_list

            
    def _compute_recursion_incoming(self, recurse_step):
        ## TODO
        recurse_intersection_edges = deepcopy(self.intersection_edge)
        for root_edge_id in self.intersection_edge.keys():
            if root_edge_id[-2:] == 'NE':
                continue
            current_edge = [root_edge_id]
            JuncID = self.get_facing_intersection(root_edge_id, False)
            for _ in range(recurse_step):
                new_current = []
                for edge_id in current_edge:
                    prev_edges, recurse_intersection_edges = self._add_recurse_intersection_edge(edge_id, JuncID, recurse_intersection_edges)
                    new_current.extend(prev_edges)
                current_edge = deepcopy(new_current)

        return recurse_intersection_edges

    def detect_threeleg_intersection(self, JuncID):
        # input: Intersection ID
        # return: True if the intersection is a three leg interesection, otherwise false
        for edge_nm in self.junction_incoming_edges[JuncID]:
            if edge_nm[-2:] == 'NE':
                return True
        return False

    def _allow_car(self, EdgeID, LaneID):
        ## return whether the lane allows passenger cars
        return (not self.net_data[EdgeID]['lanes'][EdgeID+'_'+str(LaneID)]['type']) \
             or 'passenger' in self.net_data[EdgeID]['lanes'][EdgeID+'_'+str(LaneID)]['type'] 

    def _compute_turning_map(self):
        def identify_lanedir(edge, edge_id):
            edge['unidentified'] = list(set(edge['unidentified']))
            edge['unidentified'].sort()
            previous_edge = 0
            fliped = 0
            tmp_list = []
            for lane_idx in edge['unidentified']:
                target_edge= self.next_edge(edge_id, lane_idx, True)
                if len(target_edge) ==0:
                    continue
                
                if not previous_edge:
                    previous_edge = target_edge[0][0]
                    tmp_list.append([lane_idx])
                    continue
                
                if previous_edge != target_edge[0][0]:
                    fliped += 1
                    tmp_list.append([lane_idx])
                else:
                    tmp_list[fliped].extend([lane_idx])
                previous_edge = target_edge[0][0]
            return tmp_list

        incoming_edges = dict()
        junction_incoming_edges = dict()
        for junc_id, juncs in self.junction_data.items():
            if len(self.junction_list)> 0 and junc_id not in self.junction_list:
                continue
            inclanes = juncs['incLanes']
            junction_incoming_edges[junc_id] = []
            for ln in inclanes:
                if len(ln)<2:
                    continue
                if ln[:-2] not in incoming_edges.keys():
                    incoming_edges[ln[:-2]] = dict()
                    junction_incoming_edges[junc_id].extend([ln[:-2]])

                if self._allow_car(ln[:-2], ln[-1:]):
                    try:
                        incoming_edges[ln[:-2]]['unidentified'].extend([int(ln[-1])])
                    except:
                        incoming_edges[ln[:-2]]['unidentified'] = [int(ln[-1])]
                incoming_edges[ln[:-2]]['junction'] = junc_id

        ## pop the invalid lanes which is not for vehicle
        pop_list = []
        for edge_id, edge in incoming_edges.items():
            if 'unidentified' not in incoming_edges[edge_id] or len(incoming_edges[edge_id]['unidentified'])<2:
                pop_list.extend([edge_id])
        for idx_to_pop in pop_list:
            incoming_edges.pop(idx_to_pop)
            for _, edges in  junction_incoming_edges.items():
                try:
                    edges.remove(idx_to_pop)
                except:
                    pass
                
        def add_NE_edge(edge_list, junc_id, dir):
            edge_list[junc_id+'NE'] = dict()
            edge_list[junc_id+'NE']['unidentified'] = []
            edge_list[junc_id+'NE']['junction'] =junc_id
            edge_list[junc_id+'NE']['right'] = []
            edge_list[junc_id+'NE']['straight'] = []
            edge_list[junc_id+'NE']['left'] = []
            edge_list[junc_id+'NE']['edge_label'] = dir

        for junc_id in self.junction_list:
            if len(junction_incoming_edges[junc_id]) == 4:
                pass
            elif len(junction_incoming_edges[junc_id]) == 3:
                ## find which one leg is missing
                if len(identify_lanedir(incoming_edges[junction_incoming_edges[junc_id][0]], junction_incoming_edges[junc_id][0]))>=2:
                    junction_incoming_edges[junc_id].extend([junc_id+'NE'])
                    add_NE_edge(incoming_edges, junc_id, 'left')
                elif len(identify_lanedir(incoming_edges[junction_incoming_edges[junc_id][1]], junction_incoming_edges[junc_id][1]))==1:
                    junction_incoming_edges[junc_id].insert(2,junc_id+'NE')
                    add_NE_edge(incoming_edges, junc_id, 'bottom')
                elif len(identify_lanedir(incoming_edges[junction_incoming_edges[junc_id][2]], junction_incoming_edges[junc_id][2]))==1:
                    junction_incoming_edges[junc_id].insert(1,junc_id+'NE')
                    add_NE_edge(incoming_edges, junc_id, 'right')
                else:
                    print("error in identifying junctions structure")

        ## compute lane direction
        for edge_id, edge in incoming_edges.items():

            edge['unidentified'] = list(set(edge['unidentified']))
            edge['unidentified'].sort()
            previous_edge = 0
            fliped = 0
            tmp_list = []
            for lane_idx in edge['unidentified']:
                target_edge= self.next_edge(edge_id, lane_idx, True)
                if len(target_edge) == 0:
                    continue
                
                if not previous_edge:
                    previous_edge = target_edge[0][0]
                    tmp_list.append([lane_idx])
                    continue
                
                if previous_edge != target_edge[0][0]:
                    fliped += 1
                    tmp_list.append([lane_idx])
                else:
                    tmp_list[fliped].extend([lane_idx])
                previous_edge = target_edge[0][0]
            
            threeleg = False
            # check whether the intersection is a three legged intersection
            for edge_nm in junction_incoming_edges[edge['junction']]:
                if edge_nm[-2:] == 'NE':
                    threeleg = True

            if len(tmp_list) == 3: 
                edge['right'] = tmp_list[0]
                edge['straight'] = tmp_list[1]
                edge['left'] = tmp_list[2]
            elif len(tmp_list) == 2:
                if threeleg:
                    current_idx  = junction_incoming_edges[edge['junction']].index(edge_id)
                    if junction_incoming_edges[edge['junction']][(current_idx+1)%4][-2:] == 'NE':
                        edge['straight'] = tmp_list[1]
                        edge['left'] = []
                        edge['right'] = tmp_list[0]
                    elif junction_incoming_edges[edge['junction']][(current_idx+2)%4][-2:] == 'NE':
                        edge['straight'] = []
                        edge['left'] = tmp_list[1]
                        edge['right'] = tmp_list[0]
                    elif junction_incoming_edges[edge['junction']][(current_idx+3)%4][-2:] == 'NE':
                        edge['straight'] = tmp_list[0]
                        edge['left'] = tmp_list[1]
                        edge['right'] = []
                    else:
                        print(edge_id+" error in indentifying")
                else:
                    edge['straight'] = tmp_list[0]
                    edge['left'] = tmp_list[1]
                    edge['right'] = []

            elif len(tmp_list) == 1:
                if threeleg:
                    current_idx  = junction_incoming_edges[edge['junction']].index(edge_id)
                    if junction_incoming_edges[edge['junction']][(current_idx+1)%4][-2:] == 'NE':
                        edge['straight'] = tmp_list[0]
                        edge['left'] = []
                        edge['right'] = []
                    elif junction_incoming_edges[edge['junction']][(current_idx+2)%4][-2:] == 'NE':
                        edge['left'] = tmp_list[0]
                        edge['straight'] = []
                        edge['right'] = []
                    else:
                        print(edge_id+" error in indentifying") 
                else:
                    print(edge_id+" error in indentifying") 
            else:
                print(edge_id+" error in indentifying")

            junc_edge_label_list = ['top', 'right', 'bottom', 'left']
            edge['edge_label'] = junc_edge_label_list[junction_incoming_edges[edge['junction']].index(edge_id)]

        return incoming_edges, junction_incoming_edges

    def get_facing_intersection(self, edge_id, recursion=True):
        try:
            if recursion:
                return self.intersection_edge_recursive[edge_id]['junction']
            else:
                return self.intersection_edge[edge_id]['junction']
        except KeyError:
            return []


    def qurey_edge_direction(self, edge_id, lane_id):
        ## get edge id and lane id 
        ## return direction, and edge label
        try:
            if 'left' in self.intersection_edge_recursive[edge_id] and lane_id in self.intersection_edge_recursive[edge_id]['left']:
                return 'left', self.intersection_edge_recursive[edge_id]['edge_label']
            elif 'straight' in self.intersection_edge_recursive[edge_id] and lane_id in self.intersection_edge_recursive[edge_id]['straight']:
                return 'straight', self.intersection_edge_recursive[edge_id]['edge_label']
            else:
                return 'wrong', None
        except:
            return 'wrong', None 


    def qurey_inner_edge_direction(self, edge_id, lane_id):
        ## get edge and lane id 
        ## return direction, and edge label
        prev_info = self.prev_edge(edge_id, lane_id)
        try:
            if 'left' in self.intersection_edge[prev_info[0][0]] and prev_info[0][1] in self.intersection_edge[prev_info[0][0]]['left']:
                return 'left', self.intersection_edge[prev_info[0][0]]['edge_label']
            elif 'straight' in self.intersection_edge[prev_info[0][0]] and prev_info[0][1] in self.intersection_edge[prev_info[0][0]]['straight']:
                return 'straight', self.intersection_edge[prev_info[0][0]]['edge_label']
            else:
                return 'wrong', None
        except:
            return 'wrong', None 


    def query_turning(self, edge, lane_id):
        if 'straight' in self.intersection_edge[edge] and lane_id in self.intersection_edge[edge]['straight']:
            return 'straight'
        elif 'left' in self.intersection_edge[edge] and lane_id in self.intersection_edge[edge]['left']:
            return 'left'
        else:
            return 'unidentified'

    def check_veh_location_to_control(self, veh):
        return True if (veh.road_id in self.intersection_edge) and \
            (('straight' in self.intersection_edge[veh.road_id] and veh.lane_index in self.intersection_edge[veh.road_id]['straight'])\
                 or ('left' in self.intersection_edge[veh.road_id] and veh.lane_index in self.intersection_edge[veh.road_id]['left'])) \
                     else False
    
    def get_veh_moving_direction(self, veh):
        if veh.road_id[0] != ':':
            dir, label = self.qurey_edge_direction(veh.road_id, veh.lane_index)
            facing_junction_id = self.get_facing_intersection(veh.road_id)
        else:
            for ind in range(len(veh.road_id)):
                if veh.road_id[len(veh.road_id)-1-ind] == '_':
                    break
            last_dash_ind = len(veh.road_id)-1-ind
            facing_junction_id = veh.road_id[1:last_dash_ind]
            dir, label = self.qurey_inner_edge_direction(veh.road_id, veh.lane_index)
        if label:
            return facing_junction_id, label+dir
        else:
            ## try to avoid empty result here
            return facing_junction_id, 'topstraight'
        
    def get_distance_to_intersection(self, veh):
        junc_id = self.get_facing_intersection(veh.road_id)
        if len(junc_id) == 0:
            return 1000000
        junc_pos = self.junction_pos(junc_id)
        return math.sqrt((veh.position[0]-float(junc_pos[0]))**2+(veh.position[1]-float(junc_pos[1]))**2)


    def junction_pos(self, junc_id):
        return [self.junction_data[junc_id]['x'], self.junction_data[junc_id]['y']]

    def edge_length(self, edge_id):
        try:
            return self.net_data[edge_id]['length']
        except:
            print("fail to load edge length of "+str(edge_id))
            return -1.0

    def get_edge_veh_lanes(self, edge_id):
        lane_ids = []
        if not edge_id in self.net_data.keys():
            print('invalid edge: '+edge_id)
            return lane_ids
        for lane_id in self.net_data[edge_id]['lanes'].keys():
            if self._allow_car(edge_id, lane_id.split('_')[-1]): 
                lane_ids.extend([lane_id.split('_')[-1]])
        return lane_ids

    def prev_edge(self, edge, lane, skip_junction=False):
        """See parent class."""
        if skip_junction:
            try:
                tlanes = self.connection_data['prev'][edge][lane]
                result = deepcopy(tlanes)
                for tlane in tlanes:
                    if tlane[0][0] == ':':
                        result.extend(self.prev_edge(tlane[0], tlane[1]))
                        result.remove(tlane)
                return result
            except KeyError:
                return []
        else:
            try:
                return self.connection_data['prev'][edge][lane]
            except KeyError:
                return []

    def next_edge(self, edge, lane, skip_junction=False):
        """See parent class."""
        if skip_junction:
            try:
                tlanes = self.connection_data['next'][edge][lane]
                result = deepcopy(tlanes)
                for tlane in tlanes:
                    if tlane[0][0] == ':':
                        result.extend(self.next_edge(tlane[0], tlane[1]))
                        result.remove(tlane)
                return result
            except KeyError:
                return []
        else:
            try:
                return self.connection_data['next'][edge][lane]
            except KeyError:
                return []




if __name__ == "__main__":
    map = NetMap('real_data/CSeditClean_1.net_threelegs.xml', ['229','499','332','334', 'cluster_2059459190_2059459387_423609690_429990307_455858124_5692002934_8446346736', '140'])
    map._add_recurse_intersection_edge("229357869#5", "229")
    map.get_facing_intersection('ddd')