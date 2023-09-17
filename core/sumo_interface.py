
import subprocess, inspect
import sumolib
import traci
import traci.constants as T # https://sumo.dlr.de/pydoc/traci.constants.html
from traci.exceptions import FatalTraCIError, TraCIException

import os, sys
sys.path.append(os.getcwd())
from core.costomized_data_structures import Namespace
from copy import deepcopy

class SubscribeDef:
    def __init__(self, tc_module, subs):
        self.tc_mod = tc_module
        self.names = [k.split('_', 1)[1].lower() for k in subs]
        self.constants = [getattr(T, k) for k in subs]

    def subscribe(self, *id):
        self.tc_mod.subscribe(*id, self.constants)
        return self

    def get(self, *id):
        res = self.tc_mod.getSubscriptionResults(*id)
        return Namespace(((n, res[v]) for n, v in zip(self.names, self.constants)))


class SUMO(object):
    def __init__(self, cfg, render=False):
        self.print_debug = False
        self.tc = None    
        self.cfg = cfg
        self.render  = render
        self.sim_step = 1
        self.sumo_cmd = self.generate_sumo()
        self.tc = self.start_sumo(self.tc)    
        self.subscribes = Namespace()
        self.setup_sub()
        self.backup_TL = dict()
        self.traffic_light_status = True

    def _print_debug(self, fun_str):
        if self.print_debug:
            print('exec: '+fun_str+' at time step: '+str(self._step))

    def setup_sub(self):
        tc = self.tc
        V = Namespace(**{k[4:].lower(): k for k, v in inspect.getmembers(T, lambda x: not callable(x)) if k.startswith('VAR_')})
        TL = Namespace(**{k[3:].lower(): k for k, v in inspect.getmembers(T, lambda x: not callable(x)) if k.startswith('TL_')})

        self.subscribes.sim = SubscribeDef(tc.simulation, [
            V.departed_vehicles_ids, V.arrived_vehicles_ids,
            V.colliding_vehicles_ids, V.loaded_vehicles_ids]).subscribe()
        self.subscribes.tl = SubscribeDef(tc.trafficlight, [
            TL.red_yellow_green_state])
        self.subscribes.veh = SubscribeDef(tc.vehicle, [
            V.road_id, V.lane_index, V.laneposition,
            V.speed, V.position, V.angle,
            V.fuelconsumption, V.noxemission, V.waiting_time])

    def val_to_str(self,x):
        return str(x).lower() if isinstance(x, bool) else str(x)

    def generate_sumo(self, **kwargs):
        # https://sumo.dlr.de/docs/SUMO.html
        sumo_args = {
            'begin': 0,
            'no-step-log': True,
            'time-to-teleport':-1,
            'collision.check-junctions': True,
            # 'random': True,
            'configuration-file': self.cfg,
            'step-length': self.sim_step,
            'log': 'sumolog'
        }
        cmd = ['sumo-gui' if self.render else 'sumo']
        for k, v in sumo_args.items():
            cmd.extend(['--%s' % k, self.val_to_str(v)] if v is not None else [])
        ## auto run
        cmd.extend(['-S'])
        print(cmd)
        return cmd

    def start_sumo(self, tc, tries=3):
        for _ in range(tries):
            try:
                if tc and not 'TRACI_NO_LOAD' in os.environ:
                    tc.load(self.sumo_cmd[1:])
                else:
                    if tc:
                        tc.close()
                    else:
                        self.port = sumolib.miscutils.getFreeSocketPort()
                    traci.start(self.sumo_cmd, port=self.port, label=str(self.port))
                    tc = traci.getConnection(str(self.port))
                return tc
            except traci.exceptions.FatalTraCIError: # Sometimes there's an unknown error while starting SUMO
                if tc:
                    tc.close()
                print('Restarting SUMO...')
                tc = None

    def reset_sumo(self):
        self.tc = self.start_sumo(self.tc)
        self.subscribes = Namespace()
        self.setup_sub()
        self.traffic_light_status = True
        return True

    def step(self):
        self.tc.simulationStep()
    
    def get_sim_info(self):
        sim_res = self.subscribes.sim.get()
        return sim_res

    def bk_tl(self):
        tl_ids = self.tc.trafficlight.getIDList()
        for tl_id in tl_ids:
            current_state = self.tc.trafficlight.getRedYellowGreenState(tl_id)
            self.backup_TL[tl_id] = deepcopy(current_state)

    def disable_all_trafficlight(self, disable_state='G'):
        ## backup TL first if not
        if len(self.backup_TL) == 0:
            self.bk_tl()
        if self.traffic_light_status:
            tl_ids = self.tc.trafficlight.getIDList()
            for tl_id in tl_ids:
                current_state = self.tc.trafficlight.getRedYellowGreenState(tl_id)
                new_state = ''
                for _ in range(len(current_state)):
                    new_state +=disable_state
                self.tc.trafficlight.setRedYellowGreenState(tl_id, new_state)
            self.traffic_light_status = False

    def restore_trafficlight(self):
        if not self.traffic_light_status:
            tl_ids = self.tc.trafficlight.getIDList()
            for tl_id in tl_ids:
                self.tc.trafficlight.setProgram(tl_id, '0')
            self.traffic_light_status = True


    def set_safety_junction(self):
        pass

    def set_veh_route(self, veh_id, route):
        if route:
            self.tc.vehicle.setRoute(veh_id, route)

    def set_tau(self, veh, tau):
        self.tc.vehicle.setTau(veh.id, tau)

    def set_max_speed_all(self, max_speed):
        self.tc.vehicletype.setMaxSpeed('DEFAULT_VEHTYPE', max_speed)

    def get_vehicle_edge(self, veh_id):
        return self.tc.vehicle.getRoadID(veh_id)

    def get_veh_waiting_time(self, veh):
        return self.tc.vehicle.getWaitingTime(veh.id), self.tc.vehicle.getAccumulatedWaitingTime(veh.id)

    def get_average_wait_time(self, edge_id):
        return self.tc.edge.getWaitingTime(edge_id)/self.tc.edge.getLastStepVehicleNumber(edge_id) if self.tc.edge.getLastStepVehicleNumber(edge_id)>0 else 0

    def get_last_step_vehicle_ids(self, edge_id):
        return self.tc.edge.getLastStepVehicleIDs(edge_id)

    def accl_control(self, veh, acc, n_acc_steps=1):
        self.tc.vehicle.slowDown(veh.id, max(0, veh.speed + acc * self.sim_step), n_acc_steps * self.sim_step)

    def set_color(self, veh, color):
        self.tc.vehicle.setColor(veh.id, color + (255,))

    def remove_veh(self, veh):
        try:
            self.tc.vehicle.remove(veh.id)
            return True
        except:
            return False

    def close(self):
        # try: traci.close()
        # except: pass
        if self.tc:
            self.tc.close()
            self.tc = None

if __name__ == "__main__":
    sumo_connection = SUMO('real_data/osm.sumocfg', render=True)
    sumo_connection.disable_all_trafficlight()
    sumo_queue = []
    for i in range(32):
        sumo_queue.extend([SUMO('real_data/osm.sumocfg')])
    while True:
        sumo_connection.tc.simulationStep()
        res = sumo_connection.get_sim_info()
        