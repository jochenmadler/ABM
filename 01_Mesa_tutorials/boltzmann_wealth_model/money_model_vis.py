import mesa
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from money_model import *
from HistogramModule import *

def agent_portrayal(agent):
    portrayal = {
        'Shape' : 'circle',
        'Filled' : 'true',
        'r' : 0.4
    }

    if agent.wealth > 0:
        portrayal['Color'] = 'red'
        portrayal['Layer'] = 0
    else:
        portrayal['Color'] = 'grey'
        portrayal['Layer'] = 1
        portrayal['r'] = 0.3

    return portrayal

# element 1: a grid
grid = mesa.visualization.CanvasGrid(agent_portrayal,10,10,500,500)

# element 2: a histogram with agents wealth
histogram = HistogramModule(list(range(10)), 200, 500)

# element 3: a line chart with Gini coeff
chart = mesa.visualization.ChartModule([{
    'Label' : 'Gini',
    'Color' : 'Black'}],
    data_collector_name='datacollector'
)

# configure and run the server
server = mesa.visualization.ModularServer(
    MoneyModel, [grid ,chart], 'Money Model', {'N' : 50, 'width' : 10, 'height' : 10}
)
server.port = 8521 # default
server.launch()


