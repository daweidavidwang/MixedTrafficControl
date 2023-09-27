import time 
import xml.etree.ElementTree as ElementTree
from lxml import etree
## global const
junction_list = \
    ['229','499','332','334',\
    'cluster_2059459190_2059459387_423609690_429990307_455858124_5692002934_8446346736','429989387','cluster_2021192413_428994253_55750386','565876699',\
            '140', 'cluster_565877085_565877100', 'cluster_1496058258_420887788']
start_edges = ['907576708','37569255#0','229357869#2', '877763350#3', '-gneE4', '36870636#7', '37552734#3', '229489781#10-AddedOnRampEdge', \
    '-E2', '962919866#2', '229743700#3', '192573956#21']
    # 718258618#0 's lane type is not correct
end_edges = ['41931740#0', '718258618#0','39830149#5', '877763366#3', 'gneE4', '223369741#8', '436475433#10', '229489593#3-AddedOffRampEdge', \
    'E2', '847825696#6', '36326561#3', '966127428#12']
edge_dirs_in = ['topin', 'rightin', 'bottomin', 'leftin']
edge_dirs_out =  ['topout', 'rightout', 'bottomout', 'leftout']
dir_list=['left','straight', 'right']    
DIR_KEYWORDS = ['TopStraight', 'TopLeft', 'RightStraight', 'RightLeft','BottomStraight', 'BottomLeft', 'LeftStraight','LeftLeft']
UNCONFLICT_SET =[ ['topstraight', 'topleft'], \
                    ['rightstraight', 'rightleft'], \
                        ['bottomstraight', 'bottomleft'], \
                            ['leftstraight', 'leftleft'], \
                                ['topstraight','bottomstraight', 'topleft', 'bottomleft'], \
                                    ['leftstraight','rightstraight', 'leftleft', 'rightleft'], \
                                        ['topleft','bottomleft'], \
                                            ['leftleft','rightleft'] ]
class timer(object):
    def __init__(self) -> None:
        self.start_time = dict()
        self.running_time = dict()

    def start(self, func):
        self.start_time[func] = time.time()

    def end(self, func):
        if self.start_time[func]:
            end_time = time.time()
            print(func+' lasts:'+ str(end_time-self.start_time[func]))
            if not func in self.running_time.keys():
                self.running_time[func] = [end_time-self.start_time[func]]
            else:
                self.running_time[func].extend([end_time-self.start_time[func]])
            self.start_time[func] = 0
        else:
            print(func+' not found')
    
    def report(self, func):
        pass


def dict_tolist(dict_input):
    result = []
    if isinstance(dict_input, list):
        return dict_input

    for _, s in dict_input.items():
        result.append(s)
    return result



def map_parser(xml_path):
    # import the .net.xml file containing all edge/type data
    parser = etree.XMLParser(recover=True)
    net_path = xml_path
    tree = ElementTree.parse(net_path, parser=parser)
    root = tree.getroot()

    # Collect information on the available types (if any are available).
    # This may be used when specifying some edge data.
    types_data = dict()

    for typ in root.findall('type'):
        type_id = typ.attrib['id']
        types_data[type_id] = dict()

        if 'speed' in typ.attrib:
            types_data[type_id]['speed'] = float(typ.attrib['speed'])
        else:
            types_data[type_id]['speed'] = None

        if 'numLanes' in typ.attrib:
            types_data[type_id]['numLanes'] = int(typ.attrib['numLanes'])
        else:
            types_data[type_id]['numLanes'] = None

    net_data = dict()
    next_conn_data = dict()  # forward looking connections
    prev_conn_data = dict()  # backward looking connections
    junction_data = dict() 

    # collect all information on the edges
    for edge in root.findall('edge'):
        edge_id = edge.attrib['id']

        # create a new key for this edge
        net_data[edge_id] = dict()

        # check for speed
        if 'speed' in edge:
            net_data[edge_id]['speed'] = float(edge.attrib['speed'])
        else:
            net_data[edge_id]['speed'] = None

        # if the edge has a type parameters, check that type for a
        # speed and parameter if one was not already found
        if 'type' in edge.attrib and edge.attrib['type'] in types_data:
            if net_data[edge_id]['speed'] is None:
                net_data[edge_id]['speed'] = \
                    float(types_data[edge.attrib['type']]['speed'])

        net_data[edge_id]['speed'] = None

        # collect the length from the lane sub-element in the edge, the
        # number of lanes from the number of lane elements, and if needed,
        # also collect the speed value (assuming it is there)
        net_data[edge_id]['numlane'] = 0
        for i, lane in enumerate(edge):
            net_data[edge_id]['numlane'] += 1
            if i == 0:
                net_data[edge_id]['length'] = float(lane.attrib['length'])
                if net_data[edge_id]['speed'] is None \
                        and 'speed' in lane.attrib:
                    net_data[edge_id]['speed'] = float(
                        lane.attrib['speed'])

        if 'shape' in edge.attrib:
            net_data[edge_id]['shape'] = edge.attrib['shape']
        else:
            net_data[edge_id]['shape'] = edge[0].attrib['shape']

        # if no speed value is present anywhere, set it to some default
        if net_data[edge_id]['speed'] is None:
            net_data[edge_id]['speed'] = 30
        
        ## also collecting lane data for enquire
        net_data[edge_id]['lanes'] = dict()
        for lane in edge.findall('lane'):
            lane_id = lane.attrib['id']
            net_data[edge_id]['lanes'][lane_id] = dict()
            if 'allow' in lane.attrib:
                net_data[edge_id]['lanes'][lane_id]['type'] = lane.attrib['allow']
            else:
                net_data[edge_id]['lanes'][lane_id]['type'] = None

    # collect connection data
    for connection in root.findall('connection'):
        from_edge = connection.attrib['from']
        from_lane = int(connection.attrib['fromLane'])

        if from_edge[0] != ":":
            # if the edge is not an internal link, then get the next
            # edge/lane pair from the "via" element
            # if the edge is not an internal link, then get the next
            # edge/lane pair from the "via" element
            try:
                via = connection.attrib['via'].rsplit('_', 1)
                to_edge = via[0]
                to_lane = int(via[1])
            except:
                to_edge = connection.attrib['to']
                to_lane = int(connection.attrib['toLane'])
        else:
            to_edge = connection.attrib['to']
            to_lane = int(connection.attrib['toLane'])

        if from_edge not in next_conn_data:
            next_conn_data[from_edge] = dict()

        if from_lane not in next_conn_data[from_edge]:
            next_conn_data[from_edge][from_lane] = list()

        if to_edge not in prev_conn_data:
            prev_conn_data[to_edge] = dict()

        if to_lane not in prev_conn_data[to_edge]:
            prev_conn_data[to_edge][to_lane] = list()

        next_conn_data[from_edge][from_lane].append((to_edge, to_lane))
        prev_conn_data[to_edge][to_lane].append((from_edge, from_lane))

    connection_data = {'next': next_conn_data, 'prev': prev_conn_data}

    for junction in root.findall('junction'):
        junction_id = junction.attrib['id']
        inclanes = junction.attrib['incLanes'].split(' ')
        intlanes = junction.attrib['intLanes'].split(' ')
        junction_data[junction_id] = dict()
        junction_data[junction_id]['incLanes'] = inclanes # The ids of the lanes that end at the intersection; sorted by direction, clockwise, with direction up = 0
        junction_data[junction_id]['intLanes'] = intlanes # The IDs of the lanes within the intersection
        junction_data[junction_id]['x'] = junction.attrib['x']
        junction_data[junction_id]['y'] = junction.attrib['y']
        incedge = []
        for lid in inclanes:
            incedge.extend([lid[:-2]])

        junction_data[junction_id]['incEdges'] = list(set(incedge))

    return net_data, connection_data, junction_data

def detect_all_junctions(map_xml):
    net_data, connection_data, junction_data = map_parser(map_xml)
    junction_list = []
    for JuncID, juncs in junction_data.items():
        incoming_edges = []
        inclanes = juncs['incLanes']
        for ln in inclanes:
            if len(ln)<2 or ln[0]==':':
                continue
            if ln[:-2] not in incoming_edges:
                incoming_edges.extend([ln[:-2]])
        if len(incoming_edges)>2:
            ## an intersection should at least contain three legs in total
            junction_list.extend([JuncID])
    return junction_list

if __name__ == "__main__":
    junction_list = detect_all_junctions('/code/MixedTrafficRouting/map_tools/manhattan/net.net.xml')
    print(junction_list)