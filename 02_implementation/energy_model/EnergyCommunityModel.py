import asyncio
import mesa
import numpy as np
from DataBase import *
from CommunityMember import *


def price_g_d(model):
    return model.schedule.agents[0].price_g_d


def price_g_s(model):
    return model.schedule.agents[0].price_g_s


def price_n_levels(model):
    if model.n_price_levels is not None:
        return model.n_price_levels.loc[0]
    else:
        return


def price_n_d(model):
    return model.n_d_price_t


def price_n_s(model):
    return model.n_s_price_t


def costs_all(model):
    return sum([a.costs_t for a in model.schedule.agents])


def costs_prosumer(model):
    return sum([a.costs_t for a in model.schedule.agents if a.prosumer])


def costs_consumer(model):
    return sum([a.costs_t for a in model.schedule.agents if not a.prosumer])


def costs_gini(model):
    # according to: https://stackoverflow.com/a/61154922
    x = np.array([a.costs_total for a in model.schedule.agents])
    diffsum = 0
    x = np.array([0 if i < 0 else i for i in x])
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


def g_d_all(model):
    return sum([a.g_d_track for a in model.schedule.agents])


def g_d_prosumer(model):
    return sum([a.g_d_track for a in model.schedule.agents if a.prosumer])


def g_d_consumer(model):
    return sum([a.g_d_track for a in model.schedule.agents if not a.prosumer])


def g_s_all(model):
    return sum([a.g_s_track for a in model.schedule.agents])


def pv_all(model):
    return sum([a.pv_track for a in model.schedule.agents])


def n_d_all(model):
    return sum([a.n_d_track for a in model.schedule.agents])


def n_d_prosumer(model):
    return sum([a.n_d_track for a in model.schedule.agents if a.prosumer])


def n_d_consumer(model):
    return sum([a.n_d_track for a in model.schedule.agents if not a.prosumer])


def n_s_all(model):
    return sum([a.n_s_track for a in model.schedule.agents])


def d_bl_all(model):
    return sum([a.d_bl for a in model.schedule.agents])


def d_bl_prosumer(model):
    return sum([a.d_bl for a in model.schedule.agents if a.prosumer])


def d_bl_consumer(model):
    return sum([a.d_bl for a in model.schedule.agents if not a.prosumer])


def co2_prosumer(model):
    return sum([a.co2e_t for a in model.schedule.agents if a.prosumer])


def co2_consumer(model):
    return sum([a.co2e_t for a in model.schedule.agents if not a.prosumer])


def co2_gini(model):
    # according to: https://stackoverflow.com/a/61154922
    x = np.array([a.co2e_total for a in model.schedule.agents])
    diffsum = 0
    x = np.array([0 if i < 0 else i for i in x])
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


def hb_soc_all(model):
    return sum([a.hb_soc_t for a in model.schedule.agents])


def hb_c_all(model):
    return sum([a.hb_c_track for a in model.schedule.agents])


def hb_d_all(model):
    return sum([a.hb_d_track for a in model.schedule.agents])


def ev_soc_all(model):
    return sum([a.ev_soc_t for a in model.schedule.agents])


def ev_c_all(model):
    return sum([a.ev_c_track for a in model.schedule.agents])


def ev_d_all(model):
    return sum([a.ev_d_track for a in model.schedule.agents])


def mean_dci_all(model):
    return np.mean([abs(a.d_bl/a.D_bl -1)/a.bl_flexibility for a in model.schedule.agents])


def mean_dci_prosumer(model):
    return np.mean([abs(a.d_bl/a.D_bl -1)/a.bl_flexibility for a in model.schedule.agents if a.prosumer])


def mean_dci_consumer(model):
    return np.mean([abs(a.d_bl/a.D_bl -1)/a.bl_flexibility for a in model.schedule.agents if not a.prosumer])


def mean_n_d_share(model):
    return np.mean([a.n_d_share_tracker[-1] for a in model.schedule.agents])


def mean_n_s_share(model):
    return np.mean([a.n_s_share_tracker[-1] for a in model.schedule.agents if a.prosumer])


class EnergyCommunityModel(mesa.Model):
    def __init__(self, simulation_data):
        # DataBase is initialized with and contains all external simulation parameters
        self.database = DataBase(self, simulation_data)
        self.G = self.database.network
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        self.internal_t = int(
            self.database.timetable[self.database.timetable.time == self.database.start_date].index[0])
        self.time_index = self.update_time_index()
        self.p2p_trading = self.database.p2p_trading
        self.nr_n_equilibrium_price_levels = int(
            self.database.optimization_parameter['nr_n_equilibrium_price_levels'])
        self.n_price_levels, self.n_volumes = None, None
        self.n_d_price_t, self.n_s_price_t = None, None
        self.n_welfare = None
        self.n_max_trading_volume_t = None
        self.n_d_volume_t, self.n_s_volume_t = None, None
        self.co2_factors_mix, self.co2_factor_pv = None, None
        self.running = True,

        # at each time step, collect data on model and agent level
        self.datacollector = mesa.DataCollector(
            model_reporters={'price_g_d': price_g_d,
                             'price_g_s': price_g_s,
                             'price_n_level': price_n_levels,
                             'price_n_d': price_n_d,
                             'price_n_s': price_n_s,
                             'costs_all': costs_all,
                             'costs_prosumer': costs_prosumer,
                             'costs_consumer': costs_consumer,
                             'costs_gini': costs_gini,
                             'g_d_all': g_d_all,
                             'g_d_prosumer': g_d_prosumer,
                             'g_d_consumer': g_d_consumer,
                             'g_s_all': g_s_all,
                             'pv_all': pv_all,
                             'n_d_all': n_d_all,
                             'n_d_prosumer': n_d_prosumer,
                             'n_d_consumer': n_d_consumer,
                             'n_s_all': n_s_all,
                             'd_bl_all': d_bl_all,
                             'd_bl_prosumer': d_bl_prosumer,
                             'd_bl_consumer': d_bl_consumer,
                             'gco2e_prosumer': co2_prosumer,
                             'gco2e_consumer': co2_consumer,
                             'co2e_gini': co2_gini,
                             'hb_soc_all': hb_soc_all,
                             'hb_c_all': hb_c_all,
                             'hb_d_all': hb_d_all,
                             'ev_soc_all': ev_soc_all,
                             'ev_c_all': ev_c_all,
                             'ev_d_all': ev_d_all,
                             'mean_dci_all': mean_dci_all,
                             'mean_dci_prosumer': mean_dci_prosumer,
                             'mean_dci_consumer': mean_dci_consumer,
                             'mean_n_d_share': mean_n_d_share,
                             'mean_n_s_share': mean_n_s_share
                             },
            agent_reporters={'prosumer': 'prosumer',
                             'hh_type': 'residential_type',
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
                             'costs_total_ct': 'costs_total',
                             'gco2e_t': 'co2e_t',
                             'gco2e_total': 'co2e_total'})
        # apply optimization parameter relevant to all agents: PV capacity, home battery, and EV battery
        buildings = self.database.buildings
        pv_efficiency = self.database.optimization_parameter['pv_efficiency']
        buildings['ANNUAL_EFFECTIVE_RADIATION_kWh'] = buildings['ANNUAL_RADIATION_kWh'] * pv_efficiency
        hb_soc_pct_t0 = self.database.optimization_parameter['hb_soc_pct_t0']
        buildings['HOME_BATTERY_CAPACITY_T0_kW'] = buildings['HOME_BATTERY_CAPACITY_kW'] * hb_soc_pct_t0
        hb_soc_max_pct = self.database.optimization_parameter['hb_soc_max_pct']
        hb_soc_min_pct = self.database.optimization_parameter['hb_soc_min_pct']
        buildings['HOME_BATTERY_CAPACITY_MAX_kW'] = buildings['HOME_BATTERY_CAPACITY_kW'] * hb_soc_max_pct
        buildings['HOME_BATTERY_CAPACITY_MIN_kW'] = buildings['HOME_BATTERY_CAPACITY_kW'] * hb_soc_min_pct
        ev_soc_pct_t0 = self.database.optimization_parameter['ev_soc_pct_t0']
        buildings['EV_BATTERY_CAPACITY_T0_kW'] = buildings['EV_BATTERY_CAPACITY_kW'] * ev_soc_pct_t0
        ev_soc_max_pct = self.database.optimization_parameter['ev_soc_max_pct']
        ev_soc_min_pct = self.database.optimization_parameter['ev_soc_min_pct']
        buildings['EV_BATTERY_CAPACITY_MAX_kW'] = buildings['EV_BATTERY_CAPACITY_kW'] * ev_soc_max_pct
        buildings['EV_BATTERY_CAPACITY_MIN_kW'] = buildings['EV_BATTERY_CAPACITY_kW'] * ev_soc_min_pct
        # create agents and add them to model scheduler and network
        centroids = np.column_stack(
            (self.database.buildings.centroid.x, self.database.buildings.centroid.y))
        for i in range(len(buildings)):
            # obtain agent attributes
            data_i = buildings.loc[i].to_dict()
            hid = data_i['HID']
            member_type = data_i['MEMBER_TYPE']
            residential_type = data_i['RESIDENTIAL_TYPE'] if member_type == 'residential' else None
            prosumer = True if data_i['CONSUMER_TYPE'] == 'prosumer' else False
            annual_effective_pv_radiation_kWh = data_i['ANNUAL_EFFECTIVE_RADIATION_kWh']
            hb_soc_t0 = data_i['HOME_BATTERY_CAPACITY_T0_kW']
            hb_soc_max, hb_soc_min = data_i['HOME_BATTERY_CAPACITY_MAX_kW'], data_i['HOME_BATTERY_CAPACITY_MIN_kW']
            number_evs = int(data_i['NUMBER_EVS'])
            ev_soc_t0 = data_i['EV_BATTERY_CAPACITY_T0_kW']
            ev_soc_max, ev_soc_min = data_i['EV_BATTERY_CAPACITY_MAX_kW'], data_i['EV_BATTERY_CAPACITY_MIN_kW']
            load_flexibility = data_i['LOAD_FLEXIBILITY']
            # create agent, add to model scheduler and network
            a = CommunityMember(hid, self, member_type, residential_type, prosumer, hb_soc_max,
                                hb_soc_min, hb_soc_t0, annual_effective_pv_radiation_kWh, load_flexibility,
                                number_evs, ev_soc_max, ev_soc_min, ev_soc_t0)
            self.schedule.add(a)
            # add to the network
            self.G.nodes.data()[i]['pos'] = (centroids[i][0], centroids[i][1])
            self.G.nodes.data()[i]['HID'] = hid
            self.grid.place_agent(a, i)

    def update_time_index(self):
        return [i % len(self.database.timetable) for i in range(
            self.internal_t + self.schedule.steps,
            self.internal_t + self.schedule.steps + int(self.database.optimization_parameter['optimization_steps']))]

    # create evenly spaced p2p trading price levels with feed-in + grid costs and day-ahead price as lower/upper bounds
    def calculate_p2p_price_levels(self):
        price_g_d, price_g_s = self.schedule.agents[0].price_g_d, self.schedule.agents[0].price_g_s
        price_n_max = price_g_d * 0.99
        price_n_min = (
            price_g_s + self.database.optimization_parameter['grid_fees']) * 1.01
        n_price_levels = pd.DataFrame(
            np.linspace([price_n_min[i] if price_n_min[i] < price_n_max[i] else price_n_max[i] for i in
                         range(len(price_n_min))],
                        [price_n_max[i] if price_n_max[i] > price_n_min[i] else price_n_min[i] for i in
                         range(len(price_n_max))],
                        self.nr_n_equilibrium_price_levels).T)

        return n_price_levels

    def run_optimization(self):
        self.n_d_price_t, self.n_s_price_t = self.schedule.agents[
            0].price_g_d, self.schedule.agents[0].price_g_s
        for a in self.schedule.agents:
            a.optimize_hems(0)
            a.store_hems_result(0)
        # set p2p price to grid price for tracking
        self.n_d_price_t, self.n_s_price_t = self.schedule.agents[
            0].price_g_d, self.schedule.agents[0].price_g_s
        return

    def run_p2p_optimization(self):
        # calculate p2p trading price levels
        self.n_price_levels = self.calculate_p2p_price_levels()
        self.n_volumes, n_max_volumes, self.n_welfare = dict(), dict(), dict()
        # for each price level, run agents' optimization and calculate p2p trading volume
        for i in range(self.nr_n_equilibrium_price_levels):
            # set global p2p buying/selling price level
            self.n_d_price_t = self.n_price_levels.iloc[:, i]
            self.n_s_price_t = self.n_d_price_t - \
                self.database.optimization_parameter['grid_fees']
            # run agents' optimization, track results, and cumulate p2p buying and selling volumes
            a_n_d_volumes, a_n_s_volumes = np.zeros(
                len(self.time_index)), np.zeros(len(self.time_index))
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
        self.n_d_volume_t, self.n_s_volume_t = self.n_volumes[
            max_n_index][0][0], self.n_volumes[max_n_index][1][0]
        self.n_max_trading_volume_t = min(self.n_d_volume_t, self.n_s_volume_t)
        self.n_d_price_t = self.n_price_levels.iloc[0, max_n_index]
        self.n_s_price_t = self.n_d_price_t - \
            self.database.optimization_parameter['grid_fees']

        return

    def step(self):
        # set up model variables and constraints and get external data for optimization period
        for a in self.schedule.agents:
            a.update_data()
            a.update_hems()
        # run different optimization procedures depending on whether p2p trading is enabled or not
        if self.p2p_trading:
            self.run_p2p_optimization()
        else:
            self.run_optimization()
        # calculate agents' costs. If p2p trading: Update trading quantities according to market clearing
        self.schedule.step()
        # collect data for model and agents, advance time_index by one
        self.datacollector.collect(self)
        self.time_index = self.update_time_index()
        
        mean_n_d_share = round(np.mean([a.n_d_share_mean for a in self.schedule.agents]),2)
        mean_n_s_share = round(np.mean([a.n_s_share_mean for a in self.schedule.agents]),2)
        
        return mean_n_d_share, mean_n_s_share

    def run_model(self, n):
        for i in range(n):
            self.step()
