from EnergyCommunityModel import *
from data_setup import *
from network_setup import *
import dataframe_image as dfi
import matplotlib.pyplot as plt
import pickle


def single_runner(paths, input_cdir_timestamp, output_dir_timestamp, n_steps, pr_dynamic, pr_year, season, p2p_trading):
    start_timer = datetime.now().replace(microsecond=0)
    # network: load existing network (energy community)
    network, gdf = load_network(paths['custom_communities_dir'], input_cdir_timestamp)
    # data: read in time series data relevant to the scenario
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
    # 
    f_name = f'{output_dir_timestamp}_mdf,adf_season_{season}_p2p_{str(p2p_trading)}_year_{str(pr_year)}_dyn_{str(pr_dynamic)}'
    # create and run model
    model = EnergyCommunityModel(simulation_data)
    for i in range(n_steps):
        model.step()
        print(f'--- {f_name}: Step {i}/{n_steps} done ---')
    # get results
    mdf, adf = model.datacollector.get_model_vars_dataframe(), model.datacollector.get_agent_vars_dataframe()
    # write to pickle file
    os.chdir(os.path.join(paths['output_dir'], output_dir_timestamp))
    with open(f'{f_name}.pickle', 'wb') as handle:
        pickle.dump([mdf, adf], handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.chdir(paths['home_path'])
    # print success message
    end_timer = datetime.now().replace(microsecond=0)
    print(f'--- SUCCESS: {f_name} stored. Total simulation time: {end_timer - start_timer} ---')
    
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
    start_date = pd.to_datetime(f'{year}-01-01 00:00') if 'winter' in season else pd.to_datetime(f'{year}-06-01 00:00')
    mdf.index = pd.date_range(start_date, freq='15T', periods=len(mdf))
    if adf_too:
        return [mdf, adf]
    else:
        return mdf


def calculate_welfare_measures(mdf):
    df = mdf[['costs_all', 'costs_prosumer', 'costs_consumer', 'costs_gini']].describe()
    tot_costs_all = mdf['costs_all'].sum(axis=0)
    tot_costs_prosumer = mdf['costs_prosumer'].sum(axis=0)
    tot_costs_consumer = mdf['costs_consumer'].sum(axis=0)
    df.loc['sum'] = [tot_costs_all, tot_costs_prosumer, tot_costs_consumer, 0]
    df.drop(['min', '25%', '50%', '75%'], axis=0, inplace=True)

    return df


def calculate_stability_measures(mdf):
    net_g_d = pd.DataFrame(columns=['net_grid_demand'], data=mdf['g_d_all'] - mdf['g_s_all'])
    autarky = pd.DataFrame(columns=['autarky_level'], data=1 - (mdf['g_d_all'] - mdf['g_s_all']) / mdf['d_bl_all'])
    net_g_d_sum = net_g_d.net_grid_demand.sum(axis=0)
    df = pd.concat([net_g_d, autarky], axis=1).describe()
    df.loc['sum'] = [net_g_d_sum, 0]
    df.drop(['min', '25%', '50%', '75%'], axis=0, inplace=True)

    return df


def calculate_sustainability_measures(mdf): pass
    # TODO: implement metrics


def calculate_measure(paths, date, scenario_dict, measure):
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

    return dict_out


def export_measure_calculation(paths, m_dict):
    os.chdir(paths['output_dir'])
    sc_name = list(m_dict.keys())[0]
    for key_0 in m_dict:
        for key_1 in m_dict[key_0]:
            for key_2 in m_dict[key_0][key_1]:
                dfi.export(m_dict[key_0][key_1][key_2], f'{sc_name}_{key_1}_{key_2}.png')
    os.chdir(paths['home_path'])

    return


def plot_mdf(mdf, sc, year, season, measure = None, ax = None):
    if ax is None:
        ax = plt.gca()
    l_1, l_2, l_3 = None, None, None
    if 'welfare' in measure.lower():
        l_1 = ax.plot(mdf['costs_prosumer'], label = 'prosumer', color='green')
        l_2 = ax.plot(mdf['costs_consumer'], label = 'consumer', color='blue')
        ax.set_ylabel('€ct')
        ax2 = ax.twinx()
        ax2.set_ylim(0,1)
        l_3 = ax2.plot(mdf['costs_gini'], label='gini coefficient', color='black', linestyle='--')
        ax2.set_ylabel('cost gini coeff.')
    elif 'stability' in measure.lower():
        delta_g = mdf['g_d_all'] - mdf['g_s_all']
        autarky = 1 - (mdf['g_d_all'] - mdf['g_s_all']) / mdf['d_bl_all']
        l_1 = ax.plot(delta_g, label = 'net grid demand', color='blue')
        ax.set_ylabel('kWh')
        ax2 = ax.twinx()
        l_2 = ax2.plot(autarky, label = 'autarky level', color='black', linestyle='--')
        ax2.set_ylim(0,1)
        ax2.set_ylabel('autarky level')
    elif 'sustainability' in measure.lower():
        l_1 = ax.plot(mdf['gco2e_prosumer'], label = 'prosumer', color='green')
        l_2 = ax.plot(mdf['gco2e_consumer'], label = 'consumer', color='blue')
        ax.set_ylabel('gCO2e')
        ax2 = ax.twinx()
        ax2.set_ylim(0,1)
        l_3 = ax2.plot(mdf['co2e_gini'], label='gini coefficient', color='black', linestyle='--')
        ax2.set_ylabel('CO2e gini coeff.')
    else:
        return
    # subplot title and combined legend
    ax.set_title(f'{year}, {season}')
    lns = l_1+l_2
    if l_3 is not None:
        lns += l_3
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
    if measure not in ['welfare', 'stability', 'sustainability']:
        raise Exception('ERROR: please select measure in [welfare, stability, sustainability].')
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
    fig, ax = plt.subplots(figsize = (10,14), nrows=nrows, ncols=ncols, sharey=True)
    for col in range(len(scenario_dict.keys())):
        sc = list(scenario_dict.keys())[col]
        row = 0
        data_dict_sc, seasons, years, p2p_bool, pr_bool = scenario_info_dict[sc]
        for year in years:
            for season in seasons:
                mdf = get_mdf(data_dict_sc, year, season)
                plot_mdf(mdf, year=year, sc=sc, season=season, measure = measure, ax = ax[row][col])
                row += 1
    # ax = annotate_rows_and_cols(ax) -> optional
    plt.tight_layout()

    return fig, ax