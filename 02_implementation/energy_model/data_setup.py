import pandas as pd
import numpy as np

# match household type based on nr_persons which are dependent on buildings' area (own assessment)
def get_household_type(building_area, annual_load_profiles):
    nr_persons = 0
    if building_area < 55: nr_persons = 1
    elif building_area < 70: nr_persons = 2
    elif building_area < 85: nr_persons = 3
    elif building_area < 100: nr_persons = 4
    else: nr_persons = 5
    # sample (random) household type according to nr_persons
    household_type = np.random.choice(annual_load_profiles[annual_load_profiles['nr_persons'] == nr_persons]['household_type'].to_list())

    return household_type

# main function to generate simulation_parameters
def setup_data(gdf, network, residential_annual_loads_path, residential_load_profile_path, residential_ev_load_profile_path ,electricity_price_path,
                     pv_generation_factors_path, optimization_parameter_path):
    simulation_data = {}
    # assign residential household load type based on building_area -> done manually in .xlsx
    #residential_annual_load_profiles = pd.read_excel(residential_annual_loads_path)
    #building_df['HOUSEHOLD_TYPE'] = [get_household_type(building_df['AREA_BUILDING_sqm'].loc[i], residential_annual_load_profiles) for i in
    #                                range(len(building_df))]

    # add df with manual agent information to parameters
    simulation_data['agent_buildings'] = gdf
    # add network to parameters
    simulation_data['network'] = network
    # read in residential load profile timeseries
    simulation_data['qh_load_profiles_kWh'] = pd.read_excel(residential_load_profile_path).iloc[:, 1:]
    # read in residential ev load profile timeseries
    simulation_data['qh_ev_load_profiles_kWh'] = pd.read_excel(residential_ev_load_profile_path).iloc[:, 1:]
    # read in electricity price timeseries
    simulation_data['qh_electricity_prices_ct_kWh'] = pd.read_excel(electricity_price_path).iloc[:, 1:]
    # read in pv generation factor timeseries
    simulation_data['qh_pv_generation_factors'] = pd.read_excel(pv_generation_factors_path).iloc[:, 1:]
    # read in manually filled optimization parameter .xlsx as dict
    parameter_df = pd.read_excel(optimization_parameter_path)
    simulation_data['optimization_parameter_dict'] = dict(zip(parameter_df.parameter, parameter_df.value))

    return simulation_data

