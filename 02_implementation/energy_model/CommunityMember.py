import mesa
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from collections import deque


class CommunityMember(mesa.Agent):

    def __init__(self, unique_id, model, member_type, residential_type, prosumer, hb_soc_max,
                 hb_soc_min, hb_soc_t0, annual_effective_pv_radiation_kWh, load_flexibility,
                 number_evs, ev_soc_max, ev_soc_min, ev_soc_t0):
        super().__init__(unique_id, model)
        # set agent parameter
        self.model = model
        self.member_type = member_type
        self.residential_type = residential_type
        self.prosumer = prosumer
        self.hb_soc_max = hb_soc_max
        self.hb_soc_min = hb_soc_min
        self.hb_soc_t0 = hb_soc_t0
        self.annual_effective_pv_radiation_kWh = annual_effective_pv_radiation_kWh
        self.nr_evs = number_evs
        self.ev_soc_max = ev_soc_max
        self.ev_soc_min = ev_soc_min
        self.ev_soc_t0 = ev_soc_t0
        self.bl_flexibility = load_flexibility

        # decision variables, used in HEMS model
        self.g_d, self.g_s, self.n_d, self.n_s = None, None, None, None
        self.g_d_binary, self.g_s_binary, self.n_d_binary, self.n_s_binary = None, None, None, None
        self.pv, self.pv_sur = None, None
        self.hb_c, self.hb_d, self.hb_c_binary, self.hb_d_binary = None, None, None, None
        self.ev_c, self.ev_d, self.ev_c_binary, self.ev_d_binary, self.l_ev = None, None, None, None, None
        # non-decision variables, used for tracking results
        self.optimization_results = None
        self.price_g_d, self.price_g_s, self.price_n_d, self.price_n_s = None, None, None, None
        self.PV, self.D_bl, self.d_bl, self.D_ev, self.L_ev = None, None, None, None, None
        self.g_d_track, self.g_s_track, self.n_d_track, self.n_s_track = None, None, None, None
        self.g_d_binary_track, self.g_s_binary_track, self.n_d_binary_track, self.n_s_binary_track = None, None, None, None
        self.pv_track, self.pv_sur_track = None, None
        self.hb_c_track, self.hb_d_track, self.hb_c_binary_track, self.hb_d_binary_track = None, None, None, None
        self.hb_soc_t = None
        self.ev_c_track, self.ev_d_track, self.ev_c_binary_track, self.ev_d_binary_track, = None, None, None, None
        self.n_d_share_tracker = deque(
            [1] * int(self.model.database.optimization_parameter['optimization_steps']))
        self.n_s_share_tracker = deque(
            [1] * int(self.model.database.optimization_parameter['optimization_steps']))
        self.n_d_share_mean, self.n_s_share_mean = None, None
        self.ev_soc_t, self.l_ev_track = None, None
        self.costs_t, self.costs_total = None, 0
        self.co2e_t, self.co2e_total = None, 0
        # set model's optimization parameter
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
        # update data and set up model
        self.update_data()
        self.setup_hems()

    def update_data(self):
        ti = self.model.time_index
        # obtain latest data slice from model's time_index
        self.PV = np.array(self.model.database.qh_pv_generation_factors['qh_generation_factor'][
                               ti]) * self.annual_effective_pv_radiation_kWh
        self.D_bl = np.array(
            self.model.database.qh_residential_load_profiles[self.residential_type][ti])
        # D_ev: Cut off ev demand beyond max. ev battery capacity - this energy will be (re-)charged during the journey
        if self.nr_evs > 0:
            self.D_ev = np.array([i if i < self.ev_soc_max else self.ev_soc_max for i in
                                  np.array(self.model.database.qh_residential_ev_load_profiles[
                                               'consumption_' + self.residential_type][ti])])
            self.L_ev = np.array(
                self.model.database.qh_residential_ev_load_profiles['location_' + self.residential_type][ti])
        else:
            self.D_ev, self.L_ev = np.zeros(len(ti)), np.zeros(len(ti))
        self.price_g_d = np.array(
            self.model.database.qh_electricity_price_kWh['price_ct_kWh'][ti])
        self.price_g_s = np.repeat(
            self.model.database.optimization_parameter['fit_tariff'], len(ti))

        return

    def setup_hems(self):
        ti = self.model.time_index
        rti = list(range(len(ti)))
        # HEMS model
        self.m = gp.Model(f'agent_{self.unique_id}_hems')
        self.m.Params.OutputFlag = 0
        # decision variables
        self.g_d = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='g_d', lb=0, ub=self.variable_bounds)
        self.g_s = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='g_s', lb=0, ub=self.variable_bounds)
        self.g_d_binary = self.m.addVars(rti, vtype=GRB.BINARY, name='g_d_bin')
        self.g_s_binary = self.m.addVars(rti, vtype=GRB.BINARY, name='g_s_bin')
        self.n_d = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='n_d', lb=0, ub=self.variable_bounds)
        self.n_s = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='n_s', lb=0, ub=self.variable_bounds)
        self.n_d_binary = self.m.addVars(rti, vtype=GRB.BINARY, name='n_d_bin')
        self.n_s_binary = self.m.addVars(rti, vtype=GRB.BINARY, name='n_s_bin')
        self.pv_sur = self.m.addVars(rti, vtype=GRB.BINARY, name='pv_sur')
        self.pv = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='pv', lb=0, ub=self.variable_bounds)
        self.hb_c = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='hb_c', lb=0, ub=self.hb_c_max)
        self.hb_d = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='hb_d', lb=0, ub=self.hb_d_max)
        self.hb_c_binary = self.m.addVars(
            rti, vtype=GRB.BINARY, name='hb_c_bin')
        self.hb_d_binary = self.m.addVars(
            rti, vtype=GRB.BINARY, name='hb_d_bin')
        self.hb_soc = self.m.addVars(rti, vtype=GRB.CONTINUOUS, name='hb_soc_t', lb=min(
            self.hb_soc_min, self.hb_soc_t0), ub=max(self.hb_soc_max, self.hb_soc_t0))
        self.ev_c = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='ev_c', lb=0, ub=self.ev_c_max)
        self.ev_d = self.m.addVars(
            rti, vtype=GRB.CONTINUOUS, name='ev_d', lb=0, ub=self.ev_d_max)
        self.ev_c_binary = self.m.addVars(
            rti, vtype=GRB.BINARY, name='ev_c_bin')
        self.ev_d_binary = self.m.addVars(
            rti, vtype=GRB.BINARY, name='ev_d_bin')
        self.ev_soc = self.m.addVars(rti, vtype=GRB.CONTINUOUS, name='ev_soc_t', lb=min(
            self.ev_soc_min, self.ev_soc_t0), ub=max(self.ev_soc_max, self.ev_soc_t0))
        self.l_ev = self.m.addVars(rti, vtype=GRB.BINARY, name='l_ev')
        # constraints
        for j in rti:
            # prosumer constraints
            if self.prosumer:
                # possible to use less than the actually generated pv energy
                self.m.addConstr(self.pv[j] <= self.PV[j], name=f'pv_ub_{j}')
                # if pv > D_bl, pv_surplus = 1, else 0
                self.m.addConstr(-(self.big_M * self.pv_sur[j]) +
                                 self.pv[j] <= self.D_bl[j], name=f'pv_sur_t_{j}')
                self.m.addConstr(self.big_M * (1 - self.pv_sur[j]) + self.pv[j] >= self.D_bl[j],
                                 name=f'pv_sur_f_{j}')
                # indicator constraints: no energy can be sold/bought if pv deficit/surplus
                self.m.addConstr((self.pv_sur[j] == 1) >> (
                    self.g_d[j] <= 0), name=f'pv_sur_t_{j} -> g_d = 0')
                self.m.addConstr((self.pv_sur[j] == 1) >> (
                    self.n_d[j] <= 0), name=f'pv_sur_t_{j} -> n_d = 0')
                self.m.addConstr((self.pv_sur[j] == 0) >> (
                    self.g_s[j] <= 0), name=f'pv_sur_f_{j} -> g_s = 0')
                self.m.addConstr((self.pv_sur[j] == 0) >> (
                    self.n_s[j] <= 0), name=f'pv_sur_f_{j} -> n_s = 0')
                # grid: can either be sold to or bought from at any time step
                self.m.addConstr(
                    self.g_d_binary[j] * self.big_M >= self.g_d[j], name=f'g_d_bin_{j}')
                self.m.addConstr(
                    self.g_s_binary[j] * self.big_M >= self.g_s[j], name=f'g_s_bin_{j}')
                self.m.addConstr(
                    self.g_d_binary[j] + self.g_s_binary[j] <= 1, name=f'g_d_bin_{j} + g_s_bin_{j}')
                # if p2p trading enabled: Neighbors can either be sold to or bought from at any time step
                if self.model.p2p_trading:
                    self.m.addConstr(
                        self.n_d_binary[j] * self.big_M >= self.n_d[j], name=f'n_d_bin_{j}')
                    self.m.addConstr(
                        self.n_s_binary[j] * self.big_M >= self.n_s[j], name=f'n_s_bin_{j}')
                    self.m.addConstr(
                        self.n_d_binary[j] + self.n_s_binary[j] <= 1, name=f'n_d_bin_{j} + n_s_bin_{j}')
                else:
                    self.m.addConstr(
                        self.n_d[j] <= 0, name=f'p2p_f -> prosumer -> n_d = 0_{j}')
                    self.m.addConstr(
                        self.n_s[j] <= 0, name=f'p2p_f -> prosumer -> n_s = 0_{j}')
                # home battery: can either be charged or discharged at any time step
                self.m.addConstr(
                    self.hb_c_binary[j] * self.big_M >= self.hb_c[j], name=f'hb_c_bin_{j}')
                self.m.addConstr(
                    self.hb_d_binary[j] * self.big_M >= self.hb_d[j], name=f'hb_d_bin_{j}')
                self.m.addConstr(
                    self.hb_c_binary[j] + self.hb_d_binary[j] <= 1, name=f'hb_c_bin_{j} + hb_d_bin_{j}')
                # home battery state update, considering efficiency losses
                if j < 1:
                    self.m.addConstr(   
                        self.hb_soc[j] - self.hb_c[j] * self.hb_eta_c +
                            self.hb_d[j] / self.hb_eta_d == self.hb_soc_t0,
                        name='hb_soc_update_0')
                else:
                    self.m.addConstr(self.hb_soc[j] == self.hb_soc[j - 1] + self.hb_c[j] * self.hb_eta_c - self.hb_d[
                        j] / self.hb_eta_d, name=f'hb_soc_update_{j}')
            # consumer constraints: no pv, no storing and selling of energy
            else:
                self.m.addConstr(
                    self.pv[j] <= 0, name=f'consumer -> pv = 0_{j}')
                self.m.addConstr(self.hb_c[j] <= 0,
                                 name=f'consumer -> hb_c = 0_{j}')
                self.m.addConstr(self.hb_d[j] <= 0,
                                 name=f'consumer -> hb_d = 0_{j}')
                self.m.addConstr(
                    self.hb_soc[j] <= self.hb_soc_t0, name=f'consumer -> hb_soc = 0_{j}')
                self.m.addConstr(self.g_s[j] <= 0,
                                 name=f'consumer -> g_s = 0_{j}')
                if not self.model.p2p_trading:
                    self.m.addConstr(
                        self.n_d[j] <= 0, name=f'p2p_f -> consumer -> n_d = 0_{j}')
                self.m.addConstr(self.n_s[j] <= 0,
                                 name=f'consumer -> n_s = 0_{j}')

            # consumer and prosumer constraints: ev and load flexibility
            # no ev: no ev (dis-)charging
            if self.nr_evs < 1:
                self.m.addConstr(self.ev_c[j] <= 0,
                                 name=f'no ev -> ev_c_{j} = 0')
                self.m.addConstr(self.ev_d[j] <= 0,
                                 name=f'no ev -> ev_d_{j} = 0')
                self.m.addConstr(
                    self.ev_soc[j] <= self.ev_soc_t0, name=f'no ev -> ev_soc_{j} = 0')
                self.m.addConstr(self.l_ev[j] <= 0,
                                 name=f'no ev -> l_ev_{j} = 0')
            else:
                # ev: can either be charged or discharged at any time step
                self.m.addConstr(
                    self.ev_c_binary[j] * self.big_M >= self.ev_c[j], name=f'ev_c_bin_{j}')
                self.m.addConstr(
                    self.ev_d_binary[j] * self.big_M >= self.ev_d[j], name=f'ev_d_bin_{j}')
                self.m.addConstr(
                    self.ev_c_binary[j] + self.ev_d_binary[j] <= 1, name=f'ev_c_bin_{j} + ev_d_bin_{j}')
                # ev: can only be charged if ev is at home: if L_ev == 0 -> l_ev = 0, else 1
                self.m.addConstr(self.l_ev[j] <= int(
                    self.L_ev[j]), name=f'l_ev_{j}_f')
                self.m.addConstr(
                    self.l_ev[j] * self.big_M >= int(self.L_ev[j]), name=f'l_ev_{j}_t')
                # ev: indicator constraints: battery can only be charged if ev is at home
                self.m.addConstr((self.l_ev[j] == 1) >> (
                    self.ev_c[j] >= 0), name=f'l_ev_{j} = 1 -> ev_c_{j} >= 0')
                self.m.addConstr((self.l_ev[j] == 1) >> (
                    self.ev_d[j] >= 0), name=f'l_ev_{j} = 1 -> ev_d_{j} >= 0')
                self.m.addConstr((self.l_ev[j] == 0) >> (
                    self.ev_c[j] <= 0), name=f'l_ev_{j} = 0 -> ev_c_{j} <= 0')
                self.m.addConstr((self.l_ev[j] == 0) >> (
                    self.ev_d[j] <= 0), name=f'l_ev_{j} = 0 -> ev_d_{j} <= 0')
                # ev: battery state update must consider efficiency losses and consumption
                if j < 1:
                    self.m.addConstr(
                        self.ev_soc[j] - self.ev_c[j] * self.ev_eta_c + self.ev_d[j] / self.ev_eta_d == self.ev_soc_t0 -
                        self.D_ev[j], name='ev_soc_update_0')
                else:
                    self.m.addConstr(-self.ev_soc[j] + (self.ev_soc[j - 1] + self.ev_c[j] * self.ev_eta_c) - self.ev_d[
                        j] / self.ev_eta_d == self.D_ev[j], name=f'ev_soc_update_{j}')
            # load flexibility: load can deviate from true load by +/- x percent at any time step
            self.m.addConstr(
                (self.g_d[j] - self.g_s[j] + self.n_d[j] - self.n_s[j] + self.pv[j] + self.hb_d[j] - self.hb_c[j] +
                 self.ev_d[j] - self.ev_c[j]) / (1 + self.bl_flexibility) <= self.D_bl[j], name=f'D_bl_{j}_ub')
            self.m.addConstr(
                (self.g_d[j] - self.g_s[j] + self.n_d[j] - self.n_s[j] + self.pv[j] + self.hb_d[j] - self.hb_c[j] +
                 self.ev_d[j] - self.ev_c[j]) / (1 - self.bl_flexibility) >= self.D_bl[j], name=f'D_bl_{j}_lb')

        # load flexibility: load can vary, but must be satisfied within 24 hours
        self.m.addConstr(gp.quicksum(
            self.g_d[j] - self.g_s[j] + self.n_d[j] - self.n_s[j] + self.pv[j] + self.hb_d[j] - self.hb_c[j] +
            self.ev_d[j] - self.ev_c[j] for j in rti) >= gp.quicksum(
            self.D_bl[j] for j in rti), name='D_bl_sum')
        self.m.update()

        return

    def update_hems(self):
        self.optimization_results = dict()
        rti = list(range(len(self.model.time_index)))
        # update HEMS' constraints with current data
        if self.prosumer:
            self.m.setAttr('RHS', [self.m.getConstrByName(
                f'pv_ub_{i}') for i in rti], self.PV)
            self.m.setAttr('RHS', [self.m.getConstrByName(
                f'pv_sur_t_{i}') for i in rti], self.D_bl)
            self.m.setAttr('RHS', [self.m.getConstrByName(
                f'pv_sur_f_{i}') for i in rti], self.D_bl - self.big_M)
            if self.model.schedule.steps > 0:
                self.m.setAttr('RHS', self.m.getConstrByName(
                    'hb_soc_update_0'), self.hb_soc_t)
        if self.nr_evs > 0:
            self.m.setAttr('RHS', [self.m.getConstrByName(
                f'l_ev_{i}_f') for i in rti], self.L_ev)
            self.m.setAttr('RHS', [self.m.getConstrByName(
                f'l_ev_{i}_t') for i in rti], self.L_ev)
            if self.model.schedule.steps > 0:
                self.m.setAttr('RHS', self.m.getConstrByName(
                    f'ev_soc_update_0'), self.ev_soc_t - self.D_ev[0])
            self.m.setAttr('RHS', [self.m.getConstrByName(
                f'ev_soc_update_{i}') for i in rti if i > 0], self.D_ev[1:])
        self.m.setAttr('RHS', [self.m.getConstrByName(
            f'D_bl_{i}_ub') for i in rti], self.D_bl)
        self.m.setAttr('RHS', [self.m.getConstrByName(
            f'D_bl_{i}_lb') for i in rti], self.D_bl)
        self.m.setAttr('RHS', self.m.getConstrByName(
            'D_bl_sum'), self.D_bl.sum())

        return

    def get_hems_results(self):
        res = dict()
        # write HEMS variables to dict, only consider values for t=0
        res['D_bl'], res['PV'], res['D_ev'], res['L_ev'] = self.D_bl[0], self.PV[0], self.D_ev[0], self.L_ev[0]
        res['price_g_d'], res['price_g_s'] = self.price_g_d[0], self.price_g_s[0]
        res['price_n_d'], res['price_n_s'] = self.price_n_d[0], self.price_n_s[0]
        res['g_d'] = [i.X for i in self.m.getVars() if 'g_d[' in i.VarName][0]
        res['g_s'] = [i.X for i in self.m.getVars() if 'g_s[' in i.VarName][0]
        res['n_d'] = [i.X for i in self.m.getVars() if 'n_d[' in i.VarName][0]
        res['n_s'] = [i.X for i in self.m.getVars() if 'n_s[' in i.VarName][0]
        res['n_d_bin'] = [i.X for i in self.m.getVars() if 'n_d_bin[' in i.VarName][0]
        res['n_s_bin'] = [i.X for i in self.m.getVars() if 'n_s_bin[' in i.VarName][0]
        res['pv'] = [i.X for i in self.m.getVars() if 'pv[' in i.VarName][0]
        res['pv_sur'] = [
            i.X for i in self.m.getVars() if 'pv_sur[' in i.VarName][0]
        res['hb_c'] = [i.X for i in self.m.getVars() if 'hb_c[' in i.VarName][0]
        res['hb_d'] = [i.X for i in self.m.getVars() if 'hb_d[' in i.VarName][0]
        res['hb_soc_t'] = [
            i.X for i in self.m.getVars() if 'hb_soc_t[' in i.VarName][0]
        res['l_ev'] = [i.X for i in self.m.getVars() if 'l_ev[' in i.VarName][0]
        res['ev_c'] = [i.X for i in self.m.getVars() if 'ev_c[' in i.VarName][0]
        res['ev_d'] = [i.X for i in self.m.getVars() if 'ev_d[' in i.VarName][0]
        res['ev_soc_t'] = [
            i.X for i in self.m.getVars() if 'ev_soc_t[' in i.VarName][0]
        res['d_bl'] = res['g_d'] - res['g_s'] + res['n_d'] - res['n_s'] + res['pv'] - res[
            'hb_c'] + res['hb_d'] - res['ev_c'] + res['ev_d']

        return res

    # obtain p2p trading price from model, run optimization, store result
    def optimize_hems(self, current_p2p_price_level_nr, p2p_heuristic=True):
        rti = list(range(len(self.model.time_index)))
        self.price_n_d, self.price_n_s = self.model.n_d_price_t, self.model.n_s_price_t
        # consider uncertainty by taking the average of past p2p energy bought/sold for future p2p energy bought/sold
        self.n_d_share_mean, self.n_s_share_mean = np.mean(
            list(self.n_d_share_tracker)), np.mean(list(self.n_s_share_tracker))
        if p2p_heuristic:
            self.m.setObjective(gp.quicksum(
                self.price_g_d[i]*self.g_d[i] - self.price_g_s[i]*self.g_s[i] +
                self.price_n_d[i]*self.n_d[i] - self.price_n_s[i]*self.n_s[i] +
                self.lcoe_pv*self.pv[i] + self.lcoe_hb*self.hb_c[i] + self.lcoe_hb*self.hb_d[i] +
                self.lcoe_ev*self.ev_c[i] + self.lcoe_ev*self.ev_d[i] +
                (1-self.n_d_share_mean)*self.n_d[i] * (self.price_g_d[i]-self.price_n_d[i]) +
                (1-self.n_s_share_mean)*self.n_s[i] * (self.price_n_s[i]-self.price_g_s[i])
                for i in rti),GRB.MINIMIZE)
        else:
            self.m.setObjective(gp.quicksum(
                self.price_g_d[i]*self.g_d[i] - self.price_g_s[i]*self.g_s[i] +
                self.price_n_d[i]*self.n_d[i] - self.price_n_s[i]*self.n_s[i] +
                self.lcoe_pv*self.pv[i] + self.lcoe_hb*self.hb_c[i] + self.lcoe_hb*self.hb_d[i] +
                self.lcoe_ev*self.ev_c[i] + self.lcoe_ev*self.ev_d[i]
                for i in rti),GRB.MINIMIZE)
        self.m.optimize()
        # obtain agent's n_d and n_s from optimization result
        agent_n_d=[i.X for i in self.m.getVars() if 'n_d[' in i.VarName]
        agent_n_s=[i.X for i in self.m.getVars() if 'n_s[' in i.VarName]
        # store optimization results for current p2p price level
        self.optimization_results[current_p2p_price_level_nr]=self.get_hems_results(
        )
        if self.model.p2p_trading:
            return agent_n_d, agent_n_s
        else:
            return

    def store_hems_result(self, optimal_p2p_price_level_nr):
        # select p2p welfare-maximizing HEMS variables set
        res=self.optimization_results[optimal_p2p_price_level_nr]
        # store results to tracking variables for data logging
        self.price_g_d, self.price_g_s=res['price_g_d'], res['price_g_s']
        self.price_n_d, self.price_n_s=res['price_n_d'], res['price_n_s']
        self.g_d_track, self.g_s_track=res['g_d'], res['g_s']
        self.n_d_track, self.n_s_track=res['n_d'], res['n_s']
        self.n_d_binary_track, self.n_s_binary_track = res['n_d_bin'], res['n_s_bin']
        self.PV, self.pv_track=res['PV'], res['pv']
        self.pv_sur_track=res['pv_sur']
        self.hb_c_track, self.hb_d_track=res['hb_c'], res['hb_d']
        self.hb_soc_t=res['hb_soc_t']
        self.L_ev, self.l_ev_track=res['L_ev'], res['l_ev']
        self.ev_c_track, self.ev_d_track=res['ev_c'], res['ev_d']
        self.D_ev, self.ev_soc_t=res['D_ev'], res['ev_soc_t']
        self.D_bl, self.d_bl=res['D_bl'], res['d_bl']

        return

    def step(self):
        if self.model.p2p_trading:
            # market clearing: considering p2p max trading volume, update agent's trading quantities
            self.price_n_d, self.price_n_s=self.model.n_d_price_t, self.model.n_s_price_t
            # determine agent's share of community's p2p energy bought/sold
            a_share_n_d=self.n_d_track / self.model.n_d_volume_t if self.model.n_d_volume_t > 0 else 0
            a_share_n_s=self.n_s_track / self.model.n_s_volume_t if self.model.n_s_volume_t > 0 else 0
            a_d_t, a_s_t=self.g_d_track + self.n_d_track, self.g_s_track + self.n_s_track
            # store energy traded with p2p before market clearing
            self.n_d_track_old, self.n_s_track_old=self.n_d_track, self.n_s_track
            # update agent's p2p energy bought/sold according to share of community's p2p energy bought/sold
            self.n_d_track=a_share_n_d * self.model.n_max_trading_volume_t
            self.n_s_track=a_share_n_s * self.model.n_max_trading_volume_t
            # update agent's grid energy bought/sold according to new p2p trading quantities
            self.g_d_track, self.g_s_track=a_d_t - self.n_d_track, a_s_t - self.n_s_track
            # store share of optimal vs. actual p2p energy bought/sold for future assessment during optimization in a LIFO queue
            if not self.prosumer or (self.prosumer and self.n_d_binary_track > 0):
                n_d_share_track_t=self.n_d_track / self.n_d_track_old if self.n_d_track_old > 0 else 0
                self.n_d_share_tracker.append(n_d_share_track_t)
                self.n_d_share_tracker.popleft()
            else:
                n_s_share_track_t=self.n_s_track / self.n_s_track_old if self.n_s_track_old > 0 else 0
                self.n_s_share_tracker.append(n_s_share_track_t)
                self.n_s_share_tracker.popleft()
        # update agent's costs according to new p2p trading quantities
        self.costs_t=self.price_g_d * self.g_d_track + self.price_n_d * self.n_d_track - self.price_g_s * self.g_s_track - \
                       self.price_n_s * self.n_s_track + self.lcoe_pv * self.pv_track + \
                       self.lcoe_hb * self.hb_d_track + self.lcoe_hb * self.hb_c_track + \
                       self.lcoe_ev * self.ev_d_track + self.lcoe_ev * self.ev_c_track
        self.costs_total += self.costs_t
        # calculate agent's co2e_t, add to its co2e_total
        co2e_mix_t=self.model.database.co2_factors_mix.iloc[self.model.time_index[0], -1]
        co2e_pv_t=self.model.database.optimization_parameter['gco2e_pv']
        if self.prosumer:
            self.co2e_t=(self.g_d_track*co2e_mix_t + \
                         (self.n_d_track + self.pv_track)*co2e_pv_t)
            - ((self.g_s_track+self.n_s_track)*co2e_pv_t)
            self.co2e_t=0 if self.co2e_t < 0 else self.co2e_t
        else:
            self.co2e_t=self.g_d_track*co2e_mix_t + self.n_d_track*co2e_pv_t
        self.co2e_total += self.co2e_t

        return
