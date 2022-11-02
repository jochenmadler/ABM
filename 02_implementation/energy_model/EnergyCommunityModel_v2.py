import random
import mesa
import numpy as np
import pandas as pd

from DataBase import *
from CommunityMember_v2 import *


def track_price_g_d(model):
    return model.schedule.agents[0].price_g_d


def track_price_g_s(model):
    return model.schedule.agents[0].price_g_s


def track_price_n_d(model):
    return model.schedule.agents[0].price_n_d


def track_price_n_s(model):
    return model.schedule.agents[0].price_n_s


def total_grid_demand(model):
    return np.sum([a.g_d_track for a in model.schedule.agents])


def total_grid_supply(model):
    return np.sum([a.g_s_track for a in model.schedule.agents])


def total_neighborhood_demand(model):
    return np.sum([a.n_d_track for a in model.schedule.agents])


def total_neighborhood_supply(model):
    return np.sum([a.n_s_track for a in model.schedule.agents])


def total_demand_actual(model):
    return np.sum([a.D_bl for a in model.schedule.agents])


def total_demand_optimal(model):
    return np.sum([a.d_bl for a in model.schedule.agents])


def pv_generation_actual(model):
    return np.sum([a.PV for a in model.schedule.agents])


def pv_generation_optimal(model):
    return np.sum([a.pv_track for a in model.schedule.agents])


# TODO: separate energy sold to/bought from grid/neighborhood for prosumers and consumers
def total_energy_traded(model):
    # return np.sum([a.g + a.n for a in model.schedule.agents if a.prosumer])
    return


def total_hb_soc(model):
    return np.sum([a.hb_soc_t for a in model.schedule.agents])


def total_hb_charging(model):
    return np.sum([a.hb_c_track for a in model.schedule.agents])


def total_hb_discharging(model):
    return np.sum([a.hb_d_track for a in model.schedule.agents])


def total_costs_consumer(model):
    return np.mean([a.costs_total for a in model.schedule.agents if not a.prosumer])


def total_costs_prosumer(model):
    return np.mean([a.costs_total for a in model.schedule.agents if a.prosumer])


# TODO: always 1 with current constraints
def compute_avg_self_consumption(model):
    return


def compute_avg_self_sufficiency(model):
    return np.mean([a.pv_track / a.D_bl if a.D_bl > 0 else a.pv_track for a in model.schedule.agents])


def compute_avg_load_shifting(model):
    #'return np.mean([(abs(a.d_bl) - abs(a.D_bl)) / a.D_bl for a in model.schedule.agents])
    return


def track_n_d_volumes(model):
    return [np.sum(model.n_volumes[i][0]) for i in range(len(model.n_volumes.keys()))]


def track_n_s_volumes(model):
    return [np.sum(model.n_volumes[i][1]) for i in range(len(model.n_volumes.keys()))]


def track_n_welfare(model):
    return model.n_welfare


def track_n_price_levels(model):
    return model.n_price_levels


def track_n_price_levels_mean(model):
    return [np.mean(model.n_price_levels[col]) for col in model.n_price_levels.columns]


class EnergyCommunityModel(mesa.Model):
    def __init__(self, simulation_data):
        # DataBase is initialized with and contains all external simulation parameters
        self.database = DataBase(self, simulation_data)
        self.G = self.database.network
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        self.internal_t = self.database.timetable[
            self.database.timetable.time == self.database.optimization_parameter['start_date'].strftime(
                '%Y-%m-%d %H:%M')].index[0]
        self.time_index = [i % len(self.database.qh_electricity_price_kWh) for i in
                           range(self.internal_t + self.schedule.steps,
                                 self.internal_t + self.schedule.steps + self.database.optimization_parameter[
                                     'optimization_steps'])]
        self.nr_n_equilibrium_price_levels = self.database.optimization_parameter['nr_n_equilibrium_price_levels']
        self.n_price_levels, self.n_volumes = None, None
        self.n_d_price_t, self.n_s_price_t = None, None
        self.n_welfare = None
        self.n_max_trading_volume_t = None
        self.n_d_volume_t, self.n_s_volume_t = None, None
        self.agent_dict = {}
        self.running = True,

        # at each time step, collect data on model and agent level
        self.datacollector = mesa.DataCollector(
            model_reporters={'price_g_d': track_price_g_d,
                             'price_g_s': track_price_g_s,
                             'price_n_d': track_price_n_d,
                             'price_n_s': track_price_g_s,
                             'total_grid_demand': total_grid_demand,
                             'total_grid_supply': total_grid_supply,
                             'n_d_volumes': track_n_d_volumes,
                             'n_s_volumes': track_n_s_volumes,
                             'n_price_levels': track_n_price_levels,
                             'n_price_levels_mean': track_n_price_levels_mean,
                             'n_welfare': track_n_welfare,
                             'total_neighborhood_demand': total_neighborhood_demand,
                             'total_neighborhood_supply': total_neighborhood_supply,
                             'total_baseload_actual_kW': total_demand_actual,
                             'total_baseload_optimal_kW': total_demand_optimal,
                             'pv_generation_actual_kW': pv_generation_actual,
                             'pv_generation_optimal_kW': pv_generation_optimal,
                             'total_home battery_soc': total_hb_soc,
                             'total_home_battery_charging': total_hb_charging,
                             'total_home_battery_discharging': total_hb_discharging,
                             'avg_load_shifting_ratio': compute_avg_load_shifting,
                             'avg_self_sufficiency_ratio': compute_avg_self_sufficiency,
                             'avg_total_costs_consumer': total_costs_consumer,
                             'avg_total_costs_prosumer': total_costs_prosumer},
            agent_reporters={'prosumer': 'prosumer',
                             'hh_type': 'residential_type',
                             'pv_percentage': 'pv_percentage',
                             'nr_evs': 'nr_evs',
                             'bl_flexibility': 'bl_flexibility',
                             'price_g_d': 'price_g_d',
                             'price_g_s': 'price_g_s',
                             'price_n_d': 'price_n_d',
                             'price_n_s': 'price_n_s',
                             'g_d': 'g_d_track',
                             'g_s': 'g_s_track',
                             'n_d': 'n_d_track',
                             'n_s': 'n_s_track',
                             'D_bl_kW': 'D_bl',
                             'd_bl_kW': 'd_bl',
                             'D_ev_kW': 'D_ev',
                             'ev_at_home_actual': 'L_ev',
                             'l_ev': 'l_ev_track',
                             'pv_max_kW': 'PV',
                             'pv_kW': 'pv_track',
                             'pv_surplus': 'pv_sur_track',
                             'hb_c_kW': 'hb_c_track',
                             'hb_d_kW': 'hb_d_track',
                             'hb_soc_kW': 'hb_soc_t',
                             'ev_c_kW': 'ev_c_track',
                             'ev_d_kW': 'ev_d_track',
                             'ev_soc_kW': 'ev_soc_t',
                             'costs_t_ct': 'costs_t',
                             'costs_total_ct': 'costs_total'
                             })

        # add agents: iterate through network and collect data from database.buildings
        centroids = np.column_stack((self.database.buildings.centroid.x, self.database.buildings.centroid.y))
        for i in range(len(self.G.nodes)):
            # sociodemographic characteristics
            hid = self.database.buildings['HID'].loc[i]
            member_type = self.database.buildings['MEMBER_TYPE'].loc[i]
            residential_type = self.database.buildings['RESIDENTIAL_TYPE'].loc[
                i] if member_type == 'residential' else None
            prosumer = True if self.database.buildings['CONSUMER_TYPE'].loc[i] == 'prosumer' else False
            # pv and home battery
            pv_percentage = self.database.buildings['ROOF_PV_PERCENTAGE'].loc[i] if prosumer else 0
            hb_soc_nom = self.database.buildings['HOME_BATTERY_CAPACITY_kW'].loc[i] if prosumer else 0
            hb_soc_max = hb_soc_nom * self.database.optimization_parameter['hb_soc_max_pct']
            hb_soc_min = hb_soc_max * self.database.optimization_parameter['hb_soc_min_pct']
            hb_soc_t0 = random.gauss((hb_soc_max + hb_soc_min) / 2, hb_soc_min / 2) if prosumer else 0
            pv_efficiency = self.database.optimization_parameter['pv_efficiency']
            roof_annual_radiation_kWh = self.database.buildings['ANNUAL_RADIATION_kWh'].loc[i]
            # ev and ev battery
            number_evs = int(self.database.buildings['NUMBER_EVS'].loc[i])
            ev_soc_nom = self.database.buildings['EV_BATTERY_CAPACITY_kW'].loc[i]
            ev_soc_max = ev_soc_nom * self.database.optimization_parameter['ev_soc_max_pct']
            ev_soc_min = ev_soc_nom * self.database.optimization_parameter['ev_soc_min_pct']
            ev_soc_t0 = random.gauss((ev_soc_max + ev_soc_min) / 2, ev_soc_min / 2)
            # load flexibility
            load_flexibility = self.database.buildings['LOAD_FLEXIBILITY'].loc[i]
            # create agent
            a = CommunityMember(hid, self, member_type, residential_type, prosumer, pv_percentage, hb_soc_max,
                                hb_soc_min, hb_soc_t0, pv_efficiency, roof_annual_radiation_kWh, load_flexibility,
                                number_evs, ev_soc_max, ev_soc_min, ev_soc_t0)
            self.schedule.add(a)
            # add to the network
            self.G.nodes.data()[i]['pos'] = (centroids[i][0], centroids[i][1])
            self.G.nodes.data()[i]['HID'] = hid
            self.grid.place_agent(a, i)
            # add to model's agent dictionary
            self.agent_dict[str(hid)] = a

    # create evenly spaced p2p trading price levels with feed-in + grid costs and day-ahead price as lower/upper bounds
    def calculate_p2p_price_levels(self):
        price_g_d, price_g_s = self.schedule.agents[0].price_g_d, self.schedule.agents[0].price_g_s
        price_n_max = price_g_d * 0.99
        price_n_min = (price_g_s + self.database.optimization_parameter['grid_fees']) * 1.01
        n_price_levels = pd.DataFrame(
            np.linspace([price_n_min[i] if price_n_min[i] < price_n_max[i] else price_n_max[i] for i in
                         range(len(price_n_min))],
                        [price_n_max[i] if price_n_max[i] > price_n_min[i] else price_n_min[i] for i in
                         range(len(price_n_max))],
                        self.nr_n_equilibrium_price_levels).T)

        return n_price_levels

    def step(self):
        # set up model variables and constraints and get external data for optimization period
        for a in self.schedule.agents:
            a.update_data()
            a.update_hems()
        # calculate p2p trading price levels
        self.n_price_levels = self.calculate_p2p_price_levels()
        self.n_volumes, n_max_volumes, self.n_welfare = dict(), dict(), dict()
        # for each price level, run agents' optimization and calculate p2p trading volume
        for i in range(self.nr_n_equilibrium_price_levels):
            # set global p2p buying/selling price level
            self.n_d_price_t = self.n_price_levels.iloc[:, i]
            self.n_s_price_t = self.n_d_price_t - self.database.optimization_parameter['grid_fees']
            # run agents' optimization, track results, and cumulate p2p buying and selling volumes
            a_n_d_volumes, a_n_s_volumes = np.zeros(len(self.time_index)), np.zeros(len(self.time_index))
            for a in self.schedule.agents:
                a_n_d, a_n_s = a.optimize_hems(i)
                a_n_d_volumes = np.add(a_n_d_volumes, a_n_d)
                a_n_s_volumes = np.add(a_n_s_volumes, a_n_s)
            self.n_volumes[i] = [a_n_d_volumes, a_n_s_volumes]
            self.n_welfare[i] = np.sum(min(a_n_d_volumes[i], a_n_s_volumes[i]) *
                                       self.n_d_price_t.reset_index(drop=True)[i] for i in range(len(a_n_s_volumes)))

        # select p2p price that maximizes price-weighted p2p trading volume (welfare), store corresponding HEMS results
        max_n_index = max(self.n_welfare, key=self.n_welfare.get)
        for a in self.schedule.agents:
            a.store_hems_result(max_n_index)
        # for current time step, obtain total p2p trading demand and supply, calculate p2p trading volume
        self.n_d_volume_t, self.n_s_volume_t = self.n_volumes[max_n_index][0][0], self.n_volumes[max_n_index][1][0]
        self.n_max_trading_volume_t = min(self.n_d_volume_t, self.n_s_volume_t)
        self.n_d_price_t = self.n_price_levels.iloc[0, max_n_index]
        self.n_s_price_t = self.n_d_price_t - self.database.optimization_parameter['grid_fees']

        # market clearing: Replace agents' initial p2p trading volume with their share of the welfare-maximizing one
        self.schedule.step()
        # collect data for model and agents, advance time_index by one (next step)
        self.datacollector.collect(self)
        self.time_index = [i % len(self.database.qh_electricity_price_kWh) for i in
                           range(self.internal_t + self.schedule.steps,
                                 self.internal_t + self.schedule.steps + self.database.optimization_parameter[
                                     'optimization_steps'])]
        return

    def run_model(self, n):
        for i in range(n):
            self.step()
