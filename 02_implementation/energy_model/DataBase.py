
class DataBase():
    def __init__(self, model, simulation_data):
        self.model = model
        self.buildings = None
        self.network = None
        self.qh_residential_load_profiles = None
        self.qh_residential_ev_load_profiles = None
        self.qh_electricity_price_kWh = None
        self.qh_pv_generation_factors = None
        self.timetable = None
        self.optimization_parameter = None
        self.simulation_parameter = simulation_data
        self.check_and_assign_simulation_data(simulation_data)

    # check if simulation parameters are in the right form and if yes, assign to DataBase
    def check_and_assign_simulation_data(self, simulation_data):
        if not isinstance(simulation_data, dict):
            raise Exception('ERROR: simulation parameters are not provided as dictionary')

        try:
            self.buildings = simulation_data['agent_buildings']
        except:
            raise Exception('ERROR: buildings data not part of simulation data')

        try:
            self.network = simulation_data['network']
        except:
            raise Exception('ERROR: network not part of simulation data')

        try:
            self.qh_residential_load_profiles = simulation_data['qh_load_profiles_kWh']
        except:
            raise Exception('ERROR: residential load profiles not part of simulation data')

        try:
            self.qh_residential_ev_load_profiles = simulation_data['qh_ev_load_profiles_kWh']
        except:
            raise Exception('ERROR: residential ev load profiles not part of simulation data')

        try:
            self.qh_electricity_price_kWh = simulation_data['qh_electricity_prices_ct_kWh']
        except:
            raise Exception('ERROR: electricity prices not part of simulation data')

        try:
            self.qh_pv_generation_factors = simulation_data['qh_pv_generation_factors']
        except:
            raise Exception('ERROR: pv generation factors not part of simulation data')

        try:
            self.timetable = simulation_data['qh_electricity_prices_ct_kWh'][['time']]
        except:
            raise Exception('ERROR: datetime column for timetable cannot be extracted from simulation data')

        try:
            self.optimization_parameter = simulation_data['optimization_parameter_dict']
        except:
            raise Exception('ERROR: optimization parameter dict not part of simulation data')

        return
