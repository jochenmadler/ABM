import random
import mesa
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


class CommunityMember(mesa.Agent):

    def __init__(self, unique_id, model, member_type, residential_type, prosumer, pv_percentage, hb_soc_max,
                 hb_soc_min, hb_soc_t0, pv_efficiency, roof_annual_radiation_kWh, load_flexibility,
                 number_evs, ev_soc_max, ev_soc_min, ev_soc_t0):
        super().__init__(unique_id, model)
        # agent parameter
        self.model = model
        self.member_type = member_type
        self.residential_type = residential_type
        self.prosumer = prosumer
        self.pv_percentage = pv_percentage
        self.hb_soc_max = hb_soc_max
        self.hb_soc_min = hb_soc_min
        self.hb_soc_t0 = hb_soc_t0
        self.pv_efficiency = pv_efficiency
        self.annual_effective_pv_radiation_kWh = roof_annual_radiation_kWh * self.pv_efficiency
        self.nr_evs = number_evs
        self.ev_soc_max = ev_soc_max
        self.ev_soc_min = ev_soc_min
        self.ev_soc_t0 = ev_soc_t0
        self.bl_flexibility = load_flexibility
        self.number = 0

        # optimization parameter
        self.variable_bounds = self.model.database.optimization_parameter['variable_bound']
        self.big_M = self.model.database.optimization_parameter['big_M']
        self.eps = self.model.database.optimization_parameter['eps']
        self.hb_c_max = self.model.database.optimization_parameter['hb_c_max']
        self.hb_d_max = self.model.database.optimization_parameter['hb_d_max']
        self.hb_eta_c = self.model.database.optimization_parameter['hb_eta_charging']
        self.hb_eta_d = self.model.database.optimization_parameter['hb_eta_discharging']
        self.ev_c_max = self.model.database.optimization_parameter['ev_c_max']
        self.ev_d_max = self.model.database.optimization_parameter['ev_d_max']
        self.ev_eta_c = self.model.database.optimization_parameter['ev_eta_charging']
        self.ev_eta_d = self.model.database.optimization_parameter['ev_eta_discharging']
        self.lcoe_pv = self.model.database.optimization_parameter['lcoe_pv']
        self.lcoe_hb = self.model.database.optimization_parameter['lcoe_hb']
        self.lcoe_ev = self.model.database.optimization_parameter['lcoe_ev']
        # hems model
        self.m = gp.Model(f'agent_{self.unique_id}_hems')
        self.m.Params.OutputFlag = 0

        # optimization variables
        self.optimization_results = None
        self.price_g_d, self.price_g_s, self.price_n_d, self.price_n_s = None, None, None, None
        self.g_d, self.g_s, self.n_d, self.n_s = None, None, None, None
        self.g_d_binary, self.g_s_binary = None, None
        self.n_d_binary, self.n_s_binary = None, None
        self.PV_max, self.pv, self.pv_surplus = None, None, None
        self.D_bl, self.d_bl = None, None
        self.hb_c_binary, self.hb_d_binary = None, None
        self.hb_c, self.hb_d, self.hb_soc_t = None, None, None
        self.ev_c_binary, self.ev_d_binary = None, None
        self.ev_c, self.ev_d, self.ev_soc_t = None, None, None
        self.D_ev, self.L_ev, self.ev_at_home = None, None, None
        self.costs_t = None
        self.costs_total = 0

    def update_data(self):
        # obtain latest data slice from model's time_index
        self.PV_max = self.model.database.qh_pv_generation_factors['qh_generation_factor'][self.model.time_index].apply(
            lambda x: x * self.annual_effective_pv_radiation_kWh)
        self.D_bl = self.model.database.qh_residential_load_profiles[self.residential_type][self.model.time_index]
        d_ev = self.model.database.qh_residential_ev_load_profiles['consumption_' + self.residential_type][
            self.model.time_index]
        # cut off ev demand beyond max. ev battery capacity: this energy will be (re-)charged during the journey
        self.D_ev = pd.Series([i if i < self.ev_soc_max else self.ev_soc_max for i in d_ev], name=d_ev.name,
                              index=d_ev.index) if self.nr_evs > 0 else pd.Series([0] * len(d_ev), index=d_ev.index)
        self.L_ev = self.model.database.qh_residential_ev_load_profiles['location_' + self.residential_type][
            self.model.time_index] if self.nr_evs > 0 else pd.Series([0] * len(d_ev), index=d_ev.index)
        self.price_g_d = self.model.database.qh_electricity_price_kWh['price_ct_kWh'][self.model.time_index]
        self.price_g_s = pd.Series(self.model.database.optimization_parameter['fit_tariff'], self.model.time_index,
                                   name='price_g_s')

        return

    def setup_hems(self):
        self.optimization_results = dict()
        t = self.model.internal_t + self.model.schedule.steps
        time_index = self.model.time_index
        # create variables
        self.g_d = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='g_demand', lb=0, ub=self.variable_bounds)
        self.g_s = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='g_supply', lb=0, ub=self.variable_bounds)
        self.g_d_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='g_d_binary')
        self.g_s_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='g_s_binary')
        self.n_d = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='n_demand', lb=0, ub=self.variable_bounds)
        self.n_s = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='n_supply', lb=0, ub=self.variable_bounds)
        self.n_d_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='n_d_binary')
        self.n_s_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='n_s_binary')
        self.pv_surplus = self.m.addVars(time_index, vtype=GRB.BINARY, name='pv_surplus')
        self.pv = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='pv', lb=0, ub=self.variable_bounds)
        self.hb_c = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='hb_c', lb=0, ub=self.hb_c_max)
        self.hb_d = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='hb_d', lb=0, ub=self.hb_d_max)
        self.hb_c_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='hb_c_binary')
        self.hb_d_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='hb_d_binary')
        self.hb_soc = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='hb_soc_t', lb=self.hb_soc_min,
                                     ub=self.hb_soc_max)
        self.ev_c = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='ev_c', lb=0, ub=self.ev_c_max)
        self.ev_d = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='ev_d', lb=0, ub=self.ev_d_max)
        self.ev_c_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='ev_c_binary')
        self.ev_d_binary = self.m.addVars(time_index, vtype=GRB.BINARY, name='ev_d_binary')
        self.ev_soc = self.m.addVars(time_index, vtype=GRB.CONTINUOUS, name='ev_soc_t', lb=self.ev_soc_min,
                                     ub=self.ev_soc_max)
        # self.l_ev_zero = self.m.addVars(time_index, vtype=GRB.BINARY, name='one', ub=0)
        self.ev_at_home = self.m.addVars(time_index, vtype=GRB.BINARY, name='ev_at_home')

        # create constraints
        for i in time_index:
            # prosumer constraints
            if self.prosumer:
                # possible to use less than the actually generated pv energy
                self.m.addConstr(self.pv[i] <= self.PV_max[i], name='pv_ub')
                # if pv > base load, pv_surplus = 1, else 0
                self.m.addConstr(self.pv[i] >= self.D_bl[i] + self.eps - self.big_M * (1 - self.pv_surplus[i]),
                                 name='pv_surplus_true')
                self.m.addConstr(self.pv[i] <= self.D_bl[i] + self.big_M * self.pv_surplus[i], name='pv_surplus_false')
                # indicator constraints: no energy can be sold/bought if pv deficit/surplus
                self.m.addConstr((self.pv_surplus[i] == 1) >> (self.g_d[i] <= 0), name='pv_surplus_true,g_demand_zero')
                self.m.addConstr((self.pv_surplus[i] == 1) >> (self.n_d[i] <= 0), name='pv_surplus_true,n_demand_zero')
                self.m.addConstr((self.pv_surplus[i] == 0) >> (self.g_s[i] <= 0), name='pv_surplus_false,g_supply_zero')
                self.m.addConstr((self.pv_surplus[i] == 0) >> (self.n_s[i] <= 0), name='pv_surplus_false,n_supply_zero')
                # grid: can either be sold to or bought from at any time step
                self.m.addConstr(self.g_d_binary[i] * self.big_M >= self.g_d[i])
                self.m.addConstr(self.g_s_binary[i] * self.big_M >= self.g_s[i])
                self.m.addConstr(self.g_d_binary[i] + self.g_s_binary[i] <= 1)
                # neighbors: can either be sold to or bought from at any time step
                self.m.addConstr(self.n_d_binary[i] * self.big_M >= self.n_d[i])
                self.m.addConstr(self.n_s_binary[i] * self.big_M >= self.n_s[i])
                self.m.addConstr(self.n_d_binary[i] + self.n_s_binary[i] <= 1)
                # home battery: can either be charged or discharged at any time step
                self.m.addConstr(self.hb_c_binary[i] * self.big_M >= self.hb_c[i])
                self.m.addConstr(self.hb_d_binary[i] * self.big_M >= self.hb_d[i])
                self.m.addConstr(self.hb_c_binary[i] + self.hb_d_binary[i] <= 1)
                # home battery state update considering efficiency losses
                if i == self.model.internal_t:
                    self.m.addConstr(
                        self.hb_soc[i] == self.hb_soc_t0 + self.hb_c[i] * self.hb_eta_c - self.hb_d[i] / self.hb_eta_d,
                        name='hb_state_update_t0')
                elif i == t:
                    self.m.addConstr(
                        self.hb_soc[i] == self.hb_soc_t + self.hb_c[i] * self.hb_eta_c - self.hb_d[i] / self.hb_eta_d,
                        name='hb_state_update_t')
                else:
                    self.m.addConstr(self.hb_soc[i] == self.hb_soc[i - 1] + self.hb_c[i] * self.hb_eta_c - self.hb_d[
                        i] / self.hb_eta_d, name='hb_state_update_i')

            # consumer constraints: no pv, no storing and selling of energy
            else:
                self.m.addConstr(self.pv[i] <= 0, name='no_prosumer,no_pv')
                self.m.addConstr(self.hb_c[i] <= 0, name='no_prosumer,no_hb_c')
                self.m.addConstr(self.hb_d[i] <= 0, name='no_prosumer,no_hb_d')
                self.m.addConstr(self.hb_soc[i] <= self.hb_soc_t0, name='no_prosumer,no_hb_soc')
                self.m.addConstr(self.g_s[i] <= 0, name='no_prosumer,no_g_s')
                self.m.addConstr(self.n_s[i] <= 0, name='no_prosumer,no_n_s')

            # consumer and prosumer constraints: ev and load flexibility
            # no ev: no ev (dis-)charging
            if self.nr_evs < 1:
                self.m.addConstr(self.ev_c[i] <= 0, name='no_ev,no_c')
                self.m.addConstr(self.ev_d[i] <= 0, name='no_ev,no_d')
                self.m.addConstr(self.ev_soc[i] <= self.ev_soc_t0, name='no_ev,no_ev_soc')
                self.m.addConstr(self.ev_at_home[i] <= 0, name='no_ev,no_ev_never_at_home')
            else:
                # ev: can either be charged or discharged at any time step
                self.m.addConstr(self.ev_c_binary[i] * self.big_M >= self.ev_c[i])
                self.m.addConstr(self.ev_d_binary[i] * self.big_M >= self.ev_d[i])
                self.m.addConstr(self.ev_c_binary[i] + self.ev_d_binary[i] <= 1)
                # ev: can only be charged if ev is at home: if L_ev == 0 -> ev_home = 0, else 1
                self.m.addConstr(int(self.L_ev[i]) >= self.ev_at_home[i], name='ev_at_home_false')
                self.m.addConstr(int(self.L_ev[i]) <= self.ev_at_home[i] * self.big_M, name='ev_at_home_true')

                # TODO: Delete if it works anyways
                # self.m.addConstr(int(self.L_ev[i]) >= 1 - self.big_M * (1 - self.ev_home[i]),name='ev_home_true')
                # self.m.addConstr(int(self.L_ev[i]) <= self.big_M * self.ev_home[i], name='ev_home_false')

                # if self.L_ev[i] == 1:
                #    self.m.addConstr(self.ev_c[i] >= 0, name='ev_charging_when_at_home')
                #    self.m.addConstr(self.ev_d[i] >= 0, name='ev_discharging_when_at_home')
                # else:
                #   self.m.addConstr(self.ev_c[i] <= 0, name='ev_charging_when_not_at_home')
                #  self.m.addConstr(self.ev_d[i] <= 0, name='ev_discharging_when_not_at_home')

                # ev: indicator constraints: battery can only be charged if ev is at home
                self.m.addConstr((self.ev_at_home[i] == 1) >> (self.ev_c[i] >= 0), name='ev_at_home,ev_c')
                self.m.addConstr((self.ev_at_home[i] == 1) >> (self.ev_d[i] >= 0), name='ev_at_home,ev_d')
                self.m.addConstr((self.ev_at_home[i] == 0) >> (self.ev_c[i] <= 0), name='ev_not_at_home,no_ev_c')
                self.m.addConstr((self.ev_at_home[i] == 0) >> (self.ev_d[i] <= 0), name='ev_not_at_home,no_ev_d')

                # ev: battery state update must consider efficiency losses and consumption
                if i == self.model.internal_t:
                    self.m.addConstr(
                        self.ev_soc[i] == self.ev_soc_t0 + self.ev_c[i] * self.ev_eta_c - self.ev_d[i] / self.ev_eta_d -
                        self.D_ev[i], name='ev_state_update_t0')
                elif i == t:
                    self.m.addConstr(
                        self.ev_soc[i] == self.ev_soc_t + self.ev_c[i] * self.ev_eta_c - self.ev_d[i] / self.ev_eta_d -
                        self.D_ev[i], name='ev_state_update_t')
                else:
                    self.m.addConstr(self.ev_soc[i] == self.ev_soc[i - 1] + self.ev_c[i] * self.ev_eta_c - self.ev_d[
                        i] / self.ev_eta_d - self.D_ev[i], name='ev_state_update_i')

            # load flexibility: load can deviate from true load by +/- x percent at any time step
            self.m.addConstr(
                self.g_d[i] - self.g_s[i] + self.n_d[i] - self.n_s[i] + self.pv[i] + self.hb_d[i] - self.hb_c[i] +
                self.ev_d[i] - self.ev_c[i] <= (1 + self.bl_flexibility) * self.D_bl[i], name="flexible_load_ub")
            self.m.addConstr(
                self.g_d[i] - self.g_s[i] + self.n_d[i] - self.n_s[i] + self.pv[i] + self.hb_d[i] - self.hb_c[i] +
                self.ev_d[i] - self.ev_c[i] >= (1 - self.bl_flexibility) * self.D_bl[i], name="flexible_load_lb")

        # load flexibility: load can vary, but must be satisfied within 24 hours
        self.m.addConstr(gp.quicksum(
            self.g_d[i] - self.g_s[i] + self.n_d[i] - self.n_s[i] + self.pv[i] + self.hb_d[i] - self.hb_c[i] +
            self.ev_d[i] - self.ev_c[i] for i in time_index) >= gp.quicksum(
            self.D_bl[i] for i in time_index), name='flexible_load_total')
        self.m.update()

        return

    # obtain p2p trading price from model, run optimization, store result
    def optimize_hems(self, current_p2p_price_level_nr):
        self.price_n_d, self.price_n_s = self.model.n_d_price_t, self.model.n_s_price_t
        self.m.setObjective(gp.quicksum(
            self.price_g_d[i] * self.g_d[i] - self.price_g_s[i] * self.g_s[i] + self.price_n_d[i] * self.n_d[i] -
            self.price_n_s[i] * self.n_s[i] + self.lcoe_pv * self.pv[i] + self.lcoe_hb * self.hb_c[i] + self.lcoe_hb *
            self.hb_d[i] + self.lcoe_ev * self.ev_c[i] + self.lcoe_ev * self.ev_d[i] for i in self.model.time_index),
            GRB.MINIMIZE)
        self.m.optimize()
        # obtain agent's n_d and n_s from optimization result
        agent_n_d = [i.X for i in self.m.getVars()[0 + len(self.model.time_index) * 4: len(self.model.time_index) * 5]]
        agent_n_s = [i.X for i in self.m.getVars()[0 + len(self.model.time_index) * 5: len(self.model.time_index) * 6]]
        # store optimization results for current p2p price level
        self.optimization_results[current_p2p_price_level_nr] = self.get_hems_results()

        return agent_n_d, agent_n_s

    def get_hems_results(self):
        t = self.model.internal_t + self.model.schedule.steps
        time_index = self.model.time_index
        res = dict()
        # overwrite variables: only consider t[0]
        res['price_g_d'], res['price_g_s'] = self.price_g_d[t], self.price_g_s[t]
        res['price_n_d'], res['price_n_s'] = self.price_n_d[t], self.price_n_s[t]
        res['g_d'], res['g_s'] = self.m.getVars()[len(time_index) * 0].X, self.m.getVars()[len(time_index) * 1].X
        res['n_d'], res['n_s'] = self.m.getVars()[len(time_index) * 4].X, self.m.getVars()[len(time_index) * 5].X
        res['PV_max'], res['pv'] = self.PV_max[t], self.m.getVars()[len(time_index) * 9].X
        res['pv_surplus'] = self.m.getVars()[len(time_index) * 8].X
        res['hb_c'], res['hb_d'] = self.m.getVars()[len(time_index) * 10].X, self.m.getVars()[len(time_index) * 11].X
        res['hb_soc_t'] = self.m.getVars()[len(time_index) * 14].X
        res['L_ev'], res['ev_at_home'] = self.L_ev[t], self.m.getVars()[len(time_index) * 20].X
        res['ev_c'], res['ev_d'] = self.m.getVars()[len(time_index) * 15].X, self.m.getVars()[len(time_index) * 16].X
        res['D_ev'], res['ev_soc_t'] = self.D_ev[t], self.m.getVars()[len(time_index) * 19].X
        res['D_bl'], res['d_bl'] = self.D_bl[t], res['g_d'] - res['g_s'] + res['n_d'] - res['n_s'] + res['pv'] - res[
            'hb_c'] + res['hb_d'] - res['ev_c'] + res['ev_d']
        res['costs_t'] = res['price_g_d'] * res['g_d'] - res['price_g_s'] * res['g_s'] + res['price_n_d'] * res['n_d'] - \
                         res['price_n_s'] * res['n_s'] + self.lcoe_pv * res['pv'] + \
                         self.lcoe_hb * res['hb_c'] + self.lcoe_hb * res['hb_d'] + \
                         self.lcoe_ev * res['ev_c'] + self.lcoe_ev * res['ev_d']

        return res

    def store_hems_result(self, optimal_p2p_price_level_nr):
        # set variables to values from p2p trading maximizing price level
        res = self.optimization_results[optimal_p2p_price_level_nr]
        self.price_g_d, self.price_g_s = res['price_g_d'], res['price_g_s']
        self.price_n_d, self.price_n_s = res['price_n_d'], res['price_n_s']
        self.g_d, self.g_s = res['g_d'], res['g_s']
        self.n_d, self.n_s = res['n_d'], res['n_s']
        self.PV_max, self.pv = res['PV_max'], res['pv']
        self.pv_surplus = res['pv_surplus']
        self.hb_c, self.hb_d = res['hb_c'], res['hb_d']
        self.hb_soc_t = res['hb_soc_t']
        self.L_ev, self.ev_at_home = res['L_ev'], res['ev_at_home']
        self.ev_c, self.ev_d = res['ev_c'], res['ev_d']
        self.D_ev, self.ev_soc_t = res['D_ev'], res['ev_soc_t']
        self.D_bl, self.d_bl = res['D_bl'], res['d_bl']
        # self.costs_t = res['costs_t']
        # self.costs_total += self.costs_t
        gp.disposeDefaultEnv()

        return

    def step(self):
        self.price_n_d, self.price_n_s = self.model.n_d_price_t, self.model.n_s_price_t
        # market clearing: considering p2p max trading volume, update agent's trading quantities
        a_share_n_d = self.n_d / self.model.n_d_volume_t if self.model.n_d_volume_t > 0 else 0
        a_share_n_s = self.n_s / self.model.n_s_volume_t if self.model.n_s_volume_t > 0 else 0
        self.n_d = a_share_n_d * self.model.n_max_trading_volume_t
        self.n_s = a_share_n_s * self.model.n_max_trading_volume_t
        # update agent's costs according to new quantities
        self.costs_t = self.price_g_d * (self.d_bl - self.n_d) - self.price_g_s * (self.g_s - self.n_s) + \
                       self.price_n_d * self.n_d - self.price_n_s * self.n_s + self.lcoe_pv * self.pv + \
                       self.lcoe_hb * self.hb_d + self.lcoe_hb * self.hb_c + \
                       self.lcoe_ev * self.ev_d + self.lcoe_ev * self.ev_c
        self.costs_total += self.costs_t

        return
