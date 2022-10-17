import mesa
from EnergyCommunityModel import *
from CommunityMember import *


def network_portrayal(G):
    portrayal = dict()
    portrayal['nodes'] = [
        {
            'size': 6,
            'color': 'Black',
            'tooltip': f'HID: {int(agents[0].unique_id)}<br>type: {agents[0].residential_type}',
        }
        for (_, agents) in G.nodes.data('agent')
    ]

    portrayal['edges'] = [
        {
            'source': int(source),
            'target': int(target),
            'color': 'Black',
            'width': 1,
        }
        for (source, target) in G.edges
    ]

    return portrayal


# visualization output
def get_visualization_parameter():
    network = mesa.visualization.NetworkModule(network_portrayal, 500, 500)
    chart0 = mesa.visualization.ChartModule([
        {'Label': 'price_grid_ct', 'Color': 'Black'}],
        data_collector_name='datacollector'
    )
    chart1 = mesa.visualization.ChartModule([
        {'Label': 'avg_total_costs_consumer', 'Color': 'grey'},
        {'Label': 'avg_total_costs_prosumer', 'Color': 'orange'}],
        data_collector_name='datacollector'
    )
    chart2 = mesa.visualization.ChartModule([
        {'Label': 'total_energy_traded_consumer_kW', 'Color': 'grey'},
        {'Label': 'total_energy_traded_prosumer_kW', 'Color': 'orange'}],
        data_collector_name='datacollector'
    )
    chart3 = mesa.visualization.ChartModule([
        {'Label': 'total_baseload_actual_kW', 'Color': 'black'},
        {'Label': 'total_baseload_optimal_kW', 'Color': 'blue'},
        {'Label': 'pv_generation_actual_kW', 'Color': 'green'},
        {'Label': 'pv_generation_optimal_kW', 'Color': 'yellow'}],
        data_collector_name='datacollector'
    )
    chart4 = mesa.visualization.ChartModule([
        {'Label': 'avg_load_shifting_ratio', 'Color': 'red'},
        {'Label': 'avg_self_sufficiency_ratio', 'Color': 'black'}],
        data_collector_name='datacollector'
    )
    chart5 = mesa.visualization.ChartModule([
        {'Label': 'total_home battery_soc', 'Color': 'black'},
        {'Label': 'total_home_battery_charging', 'Color': 'green'},
        {'Label': 'total_home_battery_discharging', 'Color': 'red'}],
        data_collector_name='datacollector'
    )

    return [network, chart0, chart1, chart2, chart3, chart4, chart5]


# visualization and model input --- optional, not yet used
def get_simulation_parameter(simulation_data):
    # parameter dict keys must correspond to EnergyCommunityModel constructor parameters
    parameters_package = dict()
    parameters_package['simulation_data'] = simulation_data

    # visual slider for setting the start week
    parameters_package['start_week'] = mesa.visualization.Slider(
        'Set starting calendar week (1-52)',
        26,
        1,
        52,
        1,
        description='Set starting week for the simulation'
    )
    # visual slider for setting the start day
    parameters_package['start_day'] = mesa.visualization.Slider(
        'Set starting day (1: Monday, 7: Sunday)',
        1,
        0,
        7,
        1,
        description='Set starting week for the simulation'
    )
    # visual slider for setting the start hour
    parameters_package['start_hour'] = mesa.visualization.Slider(
        'Set starting hour (0-23)',
        12,
        0,
        23,
        1,
        description='Set starting week for the simulation'
    )

    return parameters_package
