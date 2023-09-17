import time 

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