import os

import pandas as pd
import numpy as np


# match household type based on nr_persons which are dependent on buildings' area (own assessment)
def get_household_type(building_area, annual_load_profiles):
    nr_persons = 0
    if building_area < 55:
        nr_persons = 1
    elif building_area < 70:
        nr_persons = 2
    elif building_area < 85:
        nr_persons = 3
    elif building_area < 100:
        nr_persons = 4
    else:
        nr_persons = 5
    # sample (random) household type according to nr_persons
    household_type = np.random.choice(
        annual_load_profiles[annual_load_profiles['nr_persons'] == nr_persons]['household_type'].to_list())

    return household_type


# main function to generate simulation_parameters
def setup_data(gdf, network,
               residential_load_profile_path,
               residential_ev_load_profile_path,
               electricity_price_dir,
               pv_generation_factors_path,
               optimization_parameter_path,
               co2_factor_dir,
               prices_dynamic=True,
               prices_year=2019,
               season='winter',
               p2p_trading=True):
    simulation_data = {}
    # add df with manual agent information to parameters
    simulation_data['buildings'] = gdf
    # add network to parameters
    simulation_data['network'] = network
    # read in residential load profile timeseries
    simulation_data['qh_load_profiles_kWh'] = pd.read_excel(residential_load_profile_path).iloc[:, 1:]
    # read in residential ev load profile timeseries
    simulation_data['qh_ev_load_profiles_kWh'] = pd.read_excel(residential_ev_load_profile_path).iloc[:, 1:]
    # read in electricity price timeseries
    p_y, p_d = str(prices_year), 'dynamic' if prices_dynamic else 'static'
    file_path = [i for i in os.listdir(electricity_price_dir) if f'{p_y}_{p_d}_qh_price_ct_kWh.xlsx' in i][0]
    file = pd.read_excel(electricity_price_dir + '\\' + file_path)
    simulation_data['timetable'] = file[['time']]
    simulation_data['qh_electricity_prices_ct_kWh'] = file.iloc[:, 1:]
    # read in pv generation factor timeseries
    simulation_data['qh_pv_generation_factors'] = pd.read_excel(pv_generation_factors_path).iloc[:, 1:]
    # read in manually filled optimization parameter .xlsx as dict
    parameter_df = pd.read_excel(optimization_parameter_path)
    simulation_data['optimization_parameter_dict'] = dict(zip(parameter_df.parameter, parameter_df.value))
    # read in co2e factor timeseries
    file_path = [i for i in os.listdir(co2_factor_dir) if f'{p_y}_qh_gCO2e_kWh.xlsx' in i][0]
    file = pd.read_excel(co2_factor_dir + file_path, index_col=0)
    simulation_data['co2_emission_factors'] = file
    # hardcoded start dates for the seasons: January, March, June, and September
    if season == 'winter':
        start_date = pd.to_datetime(f'{p_y}-01-01 00:00:00')
    elif season == 'spring':
        start_date = pd.to_datetime(f'{p_y}-03-01 00:00:00')
    elif season == 'summer':
        start_date = pd.to_datetime(f'{p_y}-06-01 00:00:00')
    elif season == 'fall' or season == 'autumn':
        start_date = pd.to_datetime(f'{p_y}-09-01 00:00:00')
    else:
        raise Exception('no valid season. Please try one of [winter, spring, summer, fall].')
    simulation_data['start_date'] = start_date
    simulation_data['p2p_trading'] = p2p_trading

    return simulation_data
