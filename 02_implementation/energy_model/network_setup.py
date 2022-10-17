import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import weights
import warnings
from datetime import datetime
import os

# get centroid coordinates for an address in the dataset
def _0_get_address_coordinates(buildings_gdf, streetname, streetnr):
    if len(buildings_gdf[(buildings_gdf.STREET == streetname) & (buildings_gdf.NUMBER == streetnr)]) == 0:
        return None
    else:
        building = buildings_gdf[(buildings_gdf.STREET == streetname) & (buildings_gdf.NUMBER == streetnr)]
        return building.centroid.x.values[0], building.centroid.y.values[0]


# filter building gdf by four addresses
def _1_filter_space_by_addresses(building_gdf, lower_left, upper_left, upper_right, lower_right, margin_pct=0.00,
                                 only_buildings=True):
    xs, ys = [], []
    # check if addresses are valid
    for address in [lower_left, upper_left, upper_right, lower_right]:
        if _0_get_address_coordinates(building_gdf, streetname=address[0], streetnr=address[1]) is None:
            print(f'invalid address: {address[0], address[1]}')
            return None
        else:

            address_coords = _0_get_address_coordinates(building_gdf, streetname=address[0], streetnr=address[1])
            xs.append(address_coords[0])
            ys.append(address_coords[1])
    # filter by min/max of xs (left/right) and ys (top/bottom), accounting for an optional margin
    top, bottom, left, right = max(ys), min(ys), min(xs), max(xs)
    y_delta, x_delta = top - bottom, right - left
    gdf_filtered = building_gdf[(building_gdf.centroid.y >= (bottom - y_delta * margin_pct)) &  # bottom
                                (building_gdf.centroid.y <= (top + y_delta * margin_pct)) &  # top
                                (building_gdf.centroid.x >= (left - x_delta * margin_pct)) &  # left
                                (building_gdf.centroid.x <= (right + x_delta * margin_pct))]  # right

    # only_buildings: return only valid addresses, no sheds, garages, etc.
    return gdf_filtered[gdf_filtered.STREET.isnull() == False] if only_buildings else gdf_filtered


# combine building gdf with roofs gdf
def _2_merge_buildings_with_roofs(buildings_gdf_quarter, roofs_gdf):
    buildings_roofs_gdf_quarter = pd.merge(roofs_gdf, buildings_gdf_quarter, on='HID')
    # rename cols
    buildings_roofs_gdf_quarter.rename(
        columns={'AREA3D_x': 'AREA3D_roof', 'PV_x': 'PV_roof', 'ST_x': 'ST_roof', 'geometry_x': 'geometry_roof',
                 'AREA3D_y': 'AREA3D_building', 'PV_y': 'PV_building', 'ST_y': 'ST_building',
                 'geometry_y': 'geometry_building'}, inplace=True)
    # create df for export with aggregated roof PV capacity
    buildings_roofs_gdf_quarter_export = buildings_roofs_gdf_quarter.groupby(['HID']).apply(
        lambda x: x[(x['PV_roof'] == 1) | (x['PV_roof'] == 2)][['AREA3D_roof', 'GLOBAL']].sum())
    buildings_roofs_gdf_quarter_export['ANNUAL_RADIATION_kWh'] = buildings_roofs_gdf_quarter_export['AREA3D_roof'] * \
                                                                 buildings_roofs_gdf_quarter_export['GLOBAL']
    buildings_roofs_gdf_quarter_export = pd.merge(buildings_roofs_gdf_quarter_export, buildings_gdf_quarter,
                                                  on='HID').drop(['AREA3D', 'PV', 'ST'], axis=1)
    buildings_roofs_gdf_quarter_export.rename(
        columns={'AREA3D_roof': 'AREA_ROOF_sqm', 'GLOBAL': 'ANNUAL_RADIATION_sqm', 'AREA': 'AREA_BUILDING_sqm'},
        inplace=True)

    return gpd.GeoDataFrame(buildings_roofs_gdf_quarter_export, crs='EPSG:25832',
                            geometry=buildings_roofs_gdf_quarter_export.geometry)


# read in raw data, construct energy community and network based on four addresses
def setup_network(buildings_file,roofs_file, lower_left=('Lärchenweg', 17), upper_left=('Lärchenweg', 8),
                  upper_right=('Altenhofstr.', 30),
                  lower_right=('Föhrenweg', 7), margin_pct=-0.05, only_buildings=True,
                  custom_communities_path='C:\\Users\joche\FIM Kernkompetenzzentrum\Paper Agent-based Modeling - Dokumente\General\\04 ABM\\03_data\custom_communities'):
    # read in shape files
    buildings_gdf = gpd.read_file(buildings_file)
    roofs_gdf = gpd.read_file(roofs_file)
    # extract buildings in custom coordinate space
    filtered_buildings_gdf = _1_filter_space_by_addresses(buildings_gdf, lower_left=lower_left, upper_left=upper_left,
                                                          upper_right=upper_right, lower_right=lower_right,
                                                          margin_pct=margin_pct, only_buildings=only_buildings)
    # combine buildings with their roofs and pv capacity
    filtered_buildings_incl_roofs_gdf = _2_merge_buildings_with_roofs(filtered_buildings_gdf, roofs_gdf)
    # create network
    centroids = np.column_stack(
        (filtered_buildings_incl_roofs_gdf.centroid.x, filtered_buildings_incl_roofs_gdf.centroid.y))
    # avoid numba warnings when constructing graph
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = weights.Gabriel.from_dataframe(filtered_buildings_incl_roofs_gdf.centroid)
        network = graph.to_networkx()
    positions = dict(zip(network.nodes, centroids))

    # save custom energy community as .shp and .xslx file
    name = ''
    for i in [i for i in [lower_left, upper_left, upper_right, lower_right]]:
        name += i[0] + str(i[1]) + '_'
    name = name.replace('.', '')[:-1]
    date = datetime.today().strftime('%Y %m %d %Hh%Mm%Ss')
    # navigate to folder and create new directory for current energy community
    os.chdir(custom_communities_path)
    os.makedirs(date)
    os.chdir(date)
    # save energy community as .shp for geometry precision and as .xlsx to modify agents' data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        filtered_buildings_incl_roofs_gdf.to_file('filtered_buildings_incl_roofs.shp')
    filtered_buildings_incl_roofs_df = pd.DataFrame(filtered_buildings_incl_roofs_gdf).iloc[:, :-1]
    filtered_buildings_incl_roofs_df['MEMBER_TYPE'] = ''
    filtered_buildings_incl_roofs_df['RESIDENTIAL_TYPE'] = ''
    filtered_buildings_incl_roofs_df['CONSUMER_TYPE'] = ''
    filtered_buildings_incl_roofs_df['ROOF_PV_PERCENTAGE'] = ''
    filtered_buildings_incl_roofs_df['HOME_BATTERY_CAPACITY_kW'] = ''
    filtered_buildings_incl_roofs_df['NUMBER_EVS'] = ''
    filtered_buildings_incl_roofs_df['EV_BATTERY_CAPACITY_kW'] = ''
    filtered_buildings_incl_roofs_df['LOAD_FLEXIBILITY'] = ''
    filtered_buildings_incl_roofs_df.to_excel('filtered_buildings_incl_roofs.xlsx')

    return


# construct network from existing custom energy community
def load_network(custom_communities_path, custom_community_folder):
    # read in .shp for geometry values and .xlsx for agents' data and merge them
    os.chdir(custom_communities_path)
    os.chdir(custom_community_folder)
    filtered_buildings_incl_roofs_gdf = gpd.read_file('filtered_buildings_incl_roofs.shp')
    filtered_buildings_incl_roofs_df = pd.read_excel('filtered_buildings_incl_roofs.xlsx').iloc[:, 1:]
    filtered_buildings_incl_roofs_gdf = pd.merge(filtered_buildings_incl_roofs_gdf[['HID', 'geometry']],
                                                 filtered_buildings_incl_roofs_df, on='HID')
    # replace missing values with zeroes
    filtered_buildings_incl_roofs_gdf.fillna(0, inplace=True)
    # create network
    centroids = np.column_stack(
        (filtered_buildings_incl_roofs_gdf.centroid.x, filtered_buildings_incl_roofs_gdf.centroid.y))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = weights.Gabriel.from_dataframe(filtered_buildings_incl_roofs_gdf.centroid)
        network = graph.to_networkx()
    positions = dict(zip(network.nodes, centroids))

    return network, filtered_buildings_incl_roofs_gdf
