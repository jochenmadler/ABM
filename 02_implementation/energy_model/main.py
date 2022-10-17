from datetime import datetime, date
import mesa
import mesa.visualization.UserParam

from network_setup import *
from data_setup import *
from model_vis import *
from EnergyCommunityModel import *

# buildings data
buildings_path = "C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\\tetraeder_solar\\20220815_0833_kreis_ingolstadt_buildings_export.shp"
roofs_path = "C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\\tetraeder_solar\\20220815_0833_kreis_ingolstadt_roofs_export.shp"
# manually curated input
custom_communities_path = 'C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\custom_communities'
optimization_parameter_path = 'C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\optimization_parameters\\2022 09 06 optimization_parameter.xlsx'
# time series data
residential_load_profile_path = "C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\load_profiles\qh_residential_loads_kWh.xlsx"
residential_annual_loads_path = "C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\load_profiles\\residential_annual_loads_kWh.xlsx"
electricity_price_path = "C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\electricity_price\qh_price_2019_ct_kWh.xlsx"
pv_generation_factors_path = "C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\pv_generation\qh_pv_generation_factor.xlsx"

simulation_start_time = datetime.now().replace(microsecond=0)

# read in custom energy community with manually modified agents data
network, gdf = load_network(custom_communities_path, '2022 09 06 14h40m22s')

# set up rest of external data
parameter_package = dict()
parameter_package['simulation_data'] = setup_data(gdf,
                                                  network,
                                                  residential_annual_loads_path,
                                                  residential_load_profile_path,
                                                  electricity_price_path,
                                                  pv_generation_factors_path, optimization_parameter_path)

# set up server and model
server = mesa.visualization.ModularServer(
    EnergyCommunityModel,
    # output parameter
    get_visualization_parameter(),
    'Energy Community Model',
    # input data and parameter
    model_params=parameter_package,
)

# launch server and model
server.port = 8521
server.launch()

simulation_end_time = datetime.now().replace(microsecond=0)
simulation_run_time = simulation_end_time - simulation_start_time

print('-' * 25, f'Simulation ended. Runtime: {simulation_run_time}', '-' * 25)
