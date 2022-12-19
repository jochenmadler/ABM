from EnergyCommunityModel import *
from data_setup import *
from network_setup import *
import dataframe_image as dfi
import matplotlib.pyplot as plt
import pickle


def create_model(paths, input_cdir_timestamp, pr_dynamic=None, pr_year=None, season=None, p2p_trading=None, scenario_t=None):
    if scenario_t is not None:
        pr_dynamic, pr_year = scenario_t['pr_dynamic_options'][0], scenario_t['pr_years'][0]
        season, p2p_trading = scenario_t['seasons'][0], scenario_t['p2p_options'][0]
    else:
        pr_dynamic, pr_year, season, p2p_trading = pr_dynamic, pr_year, season, p2p_trading
    network, gdf = load_network(paths['custom_communities_dir'], input_cdir_timestamp)
    simulation_data = setup_data(gdf, network,
                        paths['residential_load_profile_path'],
                        paths['residential_ev_load_profile_path'],
                        paths['electricity_price_dir'],
                        paths['pv_generation_factors_path'],
                        paths['optimization_parameter_path'],
                        paths['co2_emissions_factor_dir'],
                        prices_dynamic=pr_dynamic,
                        prices_year=pr_year,
                        season=season,
                        p2p_trading=p2p_trading)
    model = EnergyCommunityModel(simulation_data)
    
    return model


def single_runner(paths, input_cdir_timestamp, output_dir_timestamp, n_steps, pr_dynamic, pr_year, season, p2p_trading):
    s1_timer = datetime.now().replace(microsecond=0)
    # load existing network (community), read in relevant time series data and create model
    model = create_model(paths, input_cdir_timestamp, pr_dynamic, pr_year, season, p2p_trading)
    f_name = f'{output_dir_timestamp}_mdf,adf_season_{season}_p2p_{str(p2p_trading)}_year_{str(pr_year)}_dyn_{str(pr_dynamic)}'
    # run model for n_steps
    print(f'--- START: {f_name}')
    for i in range(n_steps):
        s2_timer = datetime.now().replace(microsecond=0)
        n_d, n_s = model.step()
        e2_timer = datetime.now().replace(microsecond=0)
        print(f'--- Step {i}/{n_steps} done. Time: {e2_timer - s2_timer}. mean(n_d_share): {n_d}, mean(n_s_share): {n_s} ---')
    # get results
    mdf, adf = model.datacollector.get_model_vars_dataframe(), model.datacollector.get_agent_vars_dataframe()
    # write to pickle file
    os.chdir(os.path.join(paths['output_dir'], output_dir_timestamp))
    with open(f'{f_name}.pickle', 'wb') as handle:
        pickle.dump([mdf, adf], handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.chdir(paths['home_path'])
    # print success message
    e1_timer = datetime.now().replace(microsecond=0)
    print(f'--- SUCCESS: {f_name} stored. Total simulation time: {e1_timer - s1_timer} ---')
    
    return


def batch_runner(paths, sc_dict, n_steps, input_cdir_timestamp, output_dir_timestamp = None):
    # extract scenario information DoF (degrees of freedom)
    seasons, p2p_options, pr_years, pr_dynamic_options = sc_dict['seasons'], sc_dict['p2p_options'], sc_dict['pr_years'], sc_dict['pr_dynamic_options']
    # create new timestamp if none was given
    if output_dir_timestamp is None:
        output_dir_timestamp = datetime.today().strftime('%Y %m %d %Hh%Mm')
    # create new dir for results
    if output_dir_timestamp not in [i.name for i in os.scandir(paths['output_dir']) if i.is_dir()]:
        os.mkdir(os.path.join(paths['output_dir'], output_dir_timestamp))
    # run and store simulation for all scenario DoF
    for season in seasons:
        for p2p_trading in p2p_options:
            for pr_year in pr_years:
                for pr_dynamic in pr_dynamic_options:
                    single_runner(paths, input_cdir_timestamp, output_dir_timestamp, n_steps,pr_dynamic,pr_year,season,p2p_trading)
                    
    return


def measure_validity_check(measure):
    if measure not in ['welfare', 'stability', 'sustainability']:
        raise Exception(f'ERROR: {measure} not in [welfare, stability, sustainability].')
    
    return None


def season_validity_check(season):
    if season not in ['winter', 'spring', 'summer', 'fall']:
        raise Exception(f'ERROR: {season} not in [winter, spring, summer, fall.')
    
    return None


def get_output_dfs(paths, date, scenario_dict_sc):
    os.chdir(paths['output_dir'])
    if date in [i.name for i in os.scandir() if i.is_dir()]:
        os.chdir(date)
    # obtain files relevant to scenario
    files = [i for i in os.listdir() if any(f'{date}_mdf,adf_season_{s}_p2p_{p2p}_year_{y}_dyn_{d}.pickle' in i
                                            for s in scenario_dict_sc['seasons']
                                            for p2p in scenario_dict_sc['p2p_options']
                                            for y in scenario_dict_sc['pr_years']
                                            for d in scenario_dict_sc['pr_dynamic_options'])]
    # read in files and store in dict_out
    dict_out_sc = {}
    for f in files:
        with open(f, 'rb') as handle:
            dfs = pickle.load(handle)
        dict_out_sc[f.split('.pickle')[0].split('mdf,adf_')[-1]] = dfs
    os.chdir(paths['home_path'])

    return dict_out_sc


def get_data_and_scenario_info(paths, date, scenario_dict_sc):
    data_dict_sc = get_output_dfs(paths, date, scenario_dict_sc)
    seasons, years = scenario_dict_sc['seasons'], scenario_dict_sc['pr_years']
    p2p_bool = 'no' if False in scenario_dict_sc['p2p_options'] else ''
    pr_bool = 'static' if False in scenario_dict_sc['pr_dynamic_options'] else 'dynamic'

    return [data_dict_sc, seasons, years, p2p_bool, pr_bool]


def get_mdf(data_dict_sc, year, season, adf_too=False):
    # returns time-indexed mdf for certain scenario, year, and season
    f_name = [i for i in list(data_dict_sc.keys()) if str(year) in i if season in i][0]
    mdf, adf = data_dict_sc[f_name][0], data_dict_sc[f_name][1]
    season_validity_check(season)
    if season == 'winter':
        start_date = pd.to_datetime(f'{year}-01-15 00:00')
    elif season == 'spring':
        start_date = pd.to_datetime(f'{year}-03-15 00:00')
    elif season == 'summer':
        start_date = pd.to_datetime(f'{year}-06-15 00:00')
    else:
        start_date = pd.to_datetime(f'{year}-09-15 00:00')
    mdf.index = pd.date_range(start_date, freq='15T', periods=len(mdf))
    if adf_too:
        return [mdf, adf]
    else:
        return mdf
    

def get_quick_mdf(paths, date, scenario_dict_sc, adf_too = False):
    data_dict_sc, seasons, years, p2p_bool, pr_bool = get_data_and_scenario_info(paths, date, scenario_dict_sc)
    if adf_too:
        return get_mdf(data_dict_sc, years[0], seasons[0], adf_too = adf_too)
    else:
        return get_mdf(data_dict_sc, years[0], seasons[0])


def calculate_welfare_measures(mdf, specifics = False):
    df = mdf[['costs_all', 'costs_prosumer', 'costs_consumer', 'costs_gini']].describe()
    tot_costs_all = mdf['costs_all'].sum(axis=0)
    tot_costs_prosumer = mdf['costs_prosumer'].sum(axis=0)
    tot_costs_consumer = mdf['costs_consumer'].sum(axis=0)
    df.loc['sum'] = [tot_costs_all, tot_costs_prosumer, tot_costs_consumer, 0]
    df.drop(['25%', '50%', '75%'], axis=0, inplace=True)
    if not specifics:
        return df.round(2)
    else:
        return {'total_costs_all': df['costs_all']['sum'],
                'total_costs_prosumer': df['costs_prosumer']['sum'],
                'total_costs_consumer': df['costs_consumer']['sum'],
                'mean_gini': df['costs_gini']['mean']}

def calculate_stability_measures(mdf, specifics = False):
    net_g_d = pd.DataFrame(columns=['net_grid_demand'], data=mdf['g_d_all'] - mdf['g_s_all'])
    autarky = pd.DataFrame(columns=['autarky_level'], data=1 - (mdf['g_d_all'] - mdf['g_s_all']) / mdf['d_bl_all'])
    net_g_d_sum = net_g_d.net_grid_demand.sum(axis=0)
    df = pd.concat([net_g_d, autarky], axis=1).describe()
    df.loc['sum'] = [net_g_d_sum, 0]
    df.drop(['min', '25%', '50%', '75%'], axis=0, inplace=True)
    if not specifics:
        return df.round(2)
    else:
        return {'total_net_grid_demand': df['net_grid_demand']['sum'],
                'mean_net_grid_demand': df['net_grid_demand']['mean'],
                'std_net_grid_demand': df['net_grid_demand']['std'],
                'times_crit_peak_demand': 0} # TODO: define times_crit_peak_demand


def calculate_sustainability_measures(mdf, specifics = False): 
    mdf['gco2e_all'] = mdf['gco2e_consumer'] + mdf['gco2e_prosumer']
    df = mdf[['gco2e_all', 'gco2e_prosumer', 'gco2e_consumer', 'co2e_gini']].describe()
    total_co2e_all = mdf['gco2e_all'].sum(axis=0)
    total_co2e_prosumer = mdf['gco2e_prosumer'].sum(axis=0)
    total_co2e_consumer = mdf['gco2e_consumer'].sum(axis=0)
    df.loc['sum'] = [total_co2e_all, total_co2e_prosumer, total_co2e_consumer, 0]
    df.drop(['25%', '50%', '75%'], axis=0, inplace=True)
    if not specifics:
        return df.round(2)
    else:
        return {'total kgCO2e_all': df['gco2e_all']['sum'] / 1000,
                'total_kgCO2e_prosumer': df['gco2e_prosumer']['sum'] / 1000,
                'total_kgCO2e_consumer': df['gco2e_consumer']['sum'] / 1000,
                'mean_CO2e_gini': df['co2e_gini']['mean']}


def calculate_measure(paths, date, scenario_dict, measure, to_df = False):
    data_dict, seasons, years, p2p_bool, pr_bool = get_data_and_scenario_info(paths, date, scenario_dict)
    dict_out = dict()
    sc_name = f'{measure}_scenario_{pr_bool} prices and {p2p_bool} P2P trading'
    dict_out[sc_name] = dict()
    for y in years:
        dict_out[sc_name][y] = dict()
        for s in seasons:
            f_name = [i for i in list(data_dict.keys()) if str(y) in i if s in i][0]
            mdf = data_dict[f_name][0]
            # create df with measures
            if 'welfare' in measure:
                dict_out[sc_name][y][s] = calculate_welfare_measures(mdf)
            elif 'stability' in measure:
                dict_out[sc_name][y][s] = calculate_stability_measures(mdf)
            elif 'sustainability' in measure:
                dict_out[sc_name][y][s] = calculate_sustainability_measures(mdf)
            else:
                dict_out[sc_name][y][s] = None
    # if only one scenario (instead of scenario_dict), return df
    if to_df and len(years) == 1 and len(seasons) == 1:
        return dict_out[sc_name][years[0]][seasons[0]]
    else: 
        return dict_out


def get_quick_measure_kpis(paths, date_t, scenarios, measure):
    # create empty df with custom cols
    measure_validity_check(measure)
    base_cols = ['year', 'season', 'price', 'p2p', 'use case']
    if measure == 'welfare':
        cols = ['total_costs_all', 'total_costs_prosumer', 'total_costs_consumer', 'mean_gini']
    elif measure == 'stability':
        cols = ['total_net_grid_demand', 'mean_net_grid_demand', 'std_net_grid_demand']
    else:
        cols = ['total kgCO2e_all', 'total_kgCO2e_prosumer', 'total_kgCO2e_consumer', 'mean_CO2e_gini']
    df = pd.DataFrame(columns=cols)
    # go through each scenario in scenario_dict_list and insert metrics as new row
    for i in range(len(scenarios)):
        sc = scenarios[i]
        mdf_sc = get_quick_mdf(paths, date_t, sc)
        year, season = int(sc['pr_years'][0]), sc['seasons'][0]
        price = 'dynamic' if sc['pr_dynamic_options'][0] else 'static'
        p2p = 'enabled' if sc['p2p_options'][0] else 'disabled'
        ind = f'sc{i}'
        if 'enabled' in p2p:
            uc = 'uc_3: dynamic prices, p2p trading'
        elif 'dynamic' in price:
            uc = 'uc_2: dynamic prices, no p2p trading'
        else:
            uc = 'uc_1: static prices, no p2p trading'
        # insert basic scenario description as base cols
        df.loc[ind, base_cols] = year, season, price, p2p, uc
        # obtain measure cols via measure-specific functions
        if measure == 'welfare':
            df.loc[ind, cols] = calculate_welfare_measures(mdf_sc, specifics=True)
        elif measure == 'stability':
            df.loc[ind, cols] = calculate_stability_measures(mdf_sc, specifics = True)
        else:
            df.loc[ind, cols] = calculate_sustainability_measures(mdf_sc, specifics = True)
        # rearrange columns (base columns bc first, then measure columns mc)
        new_cols = [*base_cols, *cols]
        df = df[new_cols]
        # change column dtypes
        df.year = df.year.astype(int)
        
    return df



def export_measure_calculation(paths, m_dict):
    os.chdir(paths['output_dir'])
    sc_name = list(m_dict.keys())[0]
    for key_0 in m_dict:
        for key_1 in m_dict[key_0]:
            for key_2 in m_dict[key_0][key_1]:
                dfi.export(m_dict[key_0][key_1][key_2], f'{sc_name}_{key_1}_{key_2}.png')
    os.chdir(paths['home_path'])

    return


def plot_mdf(mdf, measure = None, ax = None):
    if ax is None:
        ax = plt.gca()
    l_1, l_2, l_3 = None, None, None
    if 'welfare' in measure.lower():
        l_1 = ax.plot(mdf['costs_prosumer'], label = 'prosumer', color='green')
        l_2 = ax.plot(mdf['costs_consumer'], label = 'consumer', color='blue')
        ax.set_ylabel('total energy costs [€ct]')
        ax2 = ax.twinx()
        ax2.set_ylim(0,1)
        l_3 = ax2.plot(mdf['costs_gini'], label='gini coefficient', color='black', linestyle='--', linewidth=.5)
        ax2.set_ylabel('energy costs gini coefficient')
    elif 'stability' in measure.lower():
        delta_g = mdf['g_d_all'] - mdf['g_s_all']
        l_1 = ax.plot(delta_g, label = 'net grid demand', color='blue')
        ax.set_ylabel('kWh')
    elif 'sustainability' in measure.lower():
        total_co2_emissions = mdf['gco2e_prosumer'].apply(lambda x: 0 if x < 0 else x) + mdf['gco2e_consumer']
        l_1 = ax.plot(total_co2_emissions, label = 'total emissions', color='green')
        #l_1 = ax.plot(mdf['gco2e_prosumer'].apply(lambda x: 0 if x < 0 else x), label = 'prosumer', color='green')
        #l_2 = ax.plot(mdf['gco2e_consumer'], label = 'consumer', color='blue')
        ax.set_ylabel('gCO2e')
        #ax2 = ax.twinx()
        #ax2.set_ylim(0,1)
        #l_3 = ax2.plot(mdf['co2e_gini'], label='gini coefficient', color='black', linestyle='--', linewidth=.5)
        #ax2.set_ylabel('CO2e emissions gini coefficient')
    else: return
    # subplot title and combined legend
    year, month = mdf.index.year[0], mdf.index.month[0]
    if month in [11,12,1]: season = 'winter'
    elif month in [2,3,4]: season = 'spring'
    elif month in [5,6,7]: season = 'summer'
    elif month in [8,9,10]: season = 'fall'
    else: season = 'invalid season'
    ax.set_title(f'{mdf.index.year[0]}, {season}')
    lns = l_1
    if l_2 is not None: lns += l_2
    if l_3 is not None: lns += l_3
    labs = [l.get_label() for l in lns if l is not None]
    ax.legend(lns, labs, loc = 0)
    
    return


def annotate_rows_and_cols(ax):
    # annotate rows and cols header: https://stackoverflow.com/a/25814386
    row_label = [f'{y},\n{s}' for y in years for s in seasons]
    col_label = [f'Scenario {i}' for i in range(len(scenario_dict.keys()))]
    for axs, col in zip(ax[0], col_label):
        axs.set_title(col)
    for axs, row in zip(ax[:,0], row_label):
        axs.set_ylabel(row, rotation=0, size='large')
    
    return ax


def plot_all_scenarios(paths, date, scenario_dict, measure = None):
    measure_validity_check(measure)
    scenario_info_dict = dict()
    years_max, seasons_max = 0,0
    # read in basic info for every scenario
    for sc in scenario_dict.keys():
        data_dict, seasons, years, p2p_bool, pr_bool = get_data_and_scenario_info(paths, date, scenario_dict[sc])
        if len(years) > years_max: years_max = len(years)
        if len(seasons) > seasons_max: seasons_max = len(seasons)
        scenario_info_dict[sc] = [data_dict, seasons, years, p2p_bool, pr_bool]
    # set up plot
    nrows, ncols = years_max*seasons_max, len(scenario_dict.keys())
    fig, ax = plt.subplots(figsize = (10,13.5), nrows=nrows, ncols=ncols, sharey=True)
    for col in range(len(scenario_dict.keys())):
        sc = list(scenario_dict.keys())[col]
        row = 0
        data_dict_sc, seasons, years, p2p_bool, pr_bool = scenario_info_dict[sc]
        for year in years:
            for season in seasons:
                mdf = get_mdf(data_dict_sc, year, season)
                plot_mdf(mdf, measure = measure, ax = ax[row][col])
                row += 1
    # ax = annotate_rows_and_cols(ax) -> optional
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax


def plot_three_scenarios(paths, date, scenarios, measure):
    if type(scenarios) != list or len(scenarios) != 3:
        raise Exception(f'ERROR: {scenarios} must be a list of three scenarios.')
    fig, ax = plt.subplots(figsize = (18,5), ncols=len(scenarios))  
    for i in range(len(scenarios)):
        mdf_sc = get_quick_mdf(paths, date, scenario_dict_sc=scenarios[i])
        l1 = plot_mdf(mdf_sc, measure=measure, ax = ax[i])
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig