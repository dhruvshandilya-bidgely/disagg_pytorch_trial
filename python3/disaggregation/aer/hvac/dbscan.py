"""
Author - Abhinav
Date - 01/07/2019
DBSCAN routine for clusters
"""

# Import python packages
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats.stats import pearsonr

# Import packages from within the pipeline
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


static_params = hvac_static_params()


def get_regression_type(row):

    """
    Function to return kind of regression type, based on best r-square value
    Parameters:
        row             (int) : Row of dataframe
    Returns:
        type_selected   (str) : Selected kind linear or root
    """

    if row['r_square_linear'] >= row['r_square_root']:
        type_selected = 'linear'
    else:
        type_selected = 'root'

    return type_selected


def get_linear_and_root_r2(unique_hvac_clusters, energy_df):

    """
    Function to get linear or square root function based r-squares in DBSCAN
    Parameters:
        unique_hvac_clusters    (np.ndarray) : Array containing all unique clusters
        energy_df               (object)     : Dataframe containing cdd energy and cluster id
    Returns:
        hvac_r2_linear          (list)       : List containing all linear r-squares
        hvac_r2_root            (list)       : List containing all root r-squares
    """

    hvac_r2_linear = []

    for cluster_id in unique_hvac_clusters:
        cluster_id_frame = energy_df[energy_df['cluster_id'] == cluster_id]
        if len(cluster_id_frame['cdd']) == 1:
            cluster_id_r2 = np.nan
        else:
            cluster_id_r2 = pearsonr(cluster_id_frame['cdd'], cluster_id_frame['energy'])[0]
        hvac_r2_linear.append(cluster_id_r2)

    hvac_r2_root = []

    for cluster_id in unique_hvac_clusters:
        cluster_id_frame = energy_df[energy_df['cluster_id'] == cluster_id]
        if len(cluster_id_frame['cdd']) == 1:
            cluster_id_r2 = np.nan
        else:
            cluster_id_r2 = pearsonr(np.sqrt(cluster_id_frame['cdd']), cluster_id_frame['energy'])[0]
        hvac_r2_root.append(cluster_id_r2)

    return hvac_r2_linear, hvac_r2_root


def get_hvac_clusters(filtered_data):
    """
    Function to identify DBSCAN based hvac clusters for each mode
    Parameters:
        filtered_data       (np.ndarray)    : Array containing filtered consumption and temperature info
    Returns:
        filtered_data_df    (object)        : Dataframe containing filtered degree day, energy and id at epoch level
        cluster_info        (dict)          : Dictionary containing dbscan cluster based key information
    """

    filtered_data_df = pd.DataFrame(filtered_data, columns=['degree_day', 'filter_cons', 'filter_day'])

    hvac_dbscan = np.c_[np.array(filtered_data_df['degree_day'][filtered_data_df['filter_day'] == 1]),
                           np.array(filtered_data_df['filter_cons'][filtered_data_df['filter_day'] == 1])]

    scaler = StandardScaler()
    hvac_eps = static_params['dbscan_init_epsilon']
    lowest_hvac_eps = static_params['dbscan_lowest_epsilon']

    eps_contenders = []
    eps_net_r2 = []
    pick_eps = {}
    dbscan_models = []

    while hvac_eps >= lowest_hvac_eps:

        try:
            hvac_scaled = scaler.fit_transform(hvac_dbscan)
            dbscan = DBSCAN(eps=hvac_eps, min_samples=static_params['dbscan_min_points_in_eps'])
            hvac_clusters = dbscan.fit_predict(hvac_scaled)
        except (ValueError, IndexError, KeyError):
            hvac_clusters = np.zeros(len(hvac_dbscan))
            cluster_decision_frame = pd.DataFrame()
            filtered_data_df['day_hvac_cluster'] = np.zeros(len(filtered_data_df))
            cluster_info = {'hvac': {}}
            cluster_info['hvac'][0] = {'regression_kind' : 'linear', 'validity' : False}
            return filtered_data_df, cluster_info

        unique_hvac_clusters = np.unique(hvac_clusters)
        hvac_cluster_density = np.array([np.nansum(hvac_clusters == cluster) / len(hvac_clusters) for cluster in unique_hvac_clusters])

        energy_df = pd.DataFrame()
        energy_df['cdd'] = hvac_dbscan[:, 0]
        energy_df['energy'] = hvac_dbscan[:, 1]
        energy_df['cluster_id'] = hvac_clusters

        hvac_r2_linear, hvac_r2_root = get_linear_and_root_r2(unique_hvac_clusters, energy_df)

        hvac_cluster_frame = pd.DataFrame()
        hvac_cluster_frame['cluster_id'] = unique_hvac_clusters
        hvac_cluster_frame['r_square_linear'] = np.array(hvac_r2_linear)
        hvac_cluster_frame['r_square_root'] = np.array(hvac_r2_root)
        hvac_cluster_frame['cluster_density'] = hvac_cluster_density

        cluster_decision_frame = hvac_cluster_frame.copy()
        cluster_decision_frame['good_cluster'] = cluster_decision_frame['cluster_density'] > \
                                                      (1 / len(cluster_decision_frame) - static_params['relax_cluster_density'])
        cluster_decision_frame['stop_searching'] = (cluster_decision_frame['good_cluster']) & \
                                                        (cluster_decision_frame[["r_square_linear", "r_square_root"]].max(axis=1) > static_params['dbscan_lowest_r2_for_cluster'])
        cluster_decision_frame = cluster_decision_frame.sort_values(by=['cluster_density'], ascending=False)
        cluster_decision_frame['regression_type'] = cluster_decision_frame.apply(get_regression_type, axis=1)

        eps_contenders.append(round(hvac_eps, 1))
        dbscan_models.append(dbscan)
        eps_net_r2.append(cluster_decision_frame[cluster_decision_frame['good_cluster']]
                          [['r_square_linear', 'r_square_root']].max(axis=1).mean())
        pick_eps[round(hvac_eps, 1)] = {'hvac_clusters': hvac_clusters, 'hvac_cluster_decision_frame': cluster_decision_frame}

        hvac_eps -= static_params['dbscan_epsilon_decrement']

    try:
        best_selected_eps = np.around(eps_contenders[eps_net_r2.index(np.nanmax(eps_net_r2))], 1)
    except (IndexError, ValueError):
        best_selected_eps = np.around(eps_contenders[0], 1)

    hvac_clusters = list(pick_eps[best_selected_eps]['hvac_clusters'])

    cluster_decision_frame = pick_eps[best_selected_eps]['hvac_cluster_decision_frame']

    edge_confition_1 = len(cluster_decision_frame[cluster_decision_frame['stop_searching'] == True]) == 1
    edge_confition_2 = len(cluster_decision_frame[cluster_decision_frame['stop_searching'] == True]) > 0
    edge_confition_3 = False

    if edge_confition_2:
        edge_confition_3 = cluster_decision_frame[cluster_decision_frame['stop_searching'] == True]['cluster_id'].iloc[0] == -1

    if edge_confition_1 and edge_confition_3:
        hvac_clusters = [-1] * len(hvac_clusters)

    filtered_data_df['day_hvac_cluster'] = [None if flag == 0 else hvac_clusters.pop(0) for flag in filtered_data_df['filter_day']]

    cluster_info = {'hvac':{}}

    for row in range(cluster_decision_frame.shape[0]):

        cluster_info['hvac'][cluster_decision_frame['cluster_id'][row]] = \
            {'regression_kind' : cluster_decision_frame['regression_type'][row],
             'validity' : cluster_decision_frame['stop_searching'][row]}

    return filtered_data_df, cluster_info


def get_distance(input_degree, input_energy, kind, cluster_info):
    """
    Function to calculate distance of a point from a line
    Parameters:
        input_degree    (np.array)         : Array carrying info about kind of cluster and its id
        input_energy    (np.array)         : Array carrying truth value of validity of cdd-consumption
        kind            (str)              : Selected kind linear or root
        cluster_info    (dict)             : Dictionary containing cluster related key information
    Returns:
        distance        (np.array)         : distance of the cdd-consumption point from the given kind of line
    """

    distance = np.Inf
    if kind == 'linear':
        distance = cluster_info['coefficient'] * input_degree + cluster_info['intercept'] - input_energy
    elif kind == 'root':
        distance = cluster_info['coefficient'] * np.sqrt(input_degree) + cluster_info['intercept'] - input_energy

    distance = abs(distance[0][0])

    return distance


def assign_cluster_by_distance(cluster_and_kind, filtered_data_df_valid, distance_comparison, cluster_info, filtered_data_df):
    """
    Function to assign the cdd/hdd and consumption pair to a hvac cluster, by distance comparison
    Parameters:
        cluster_and_kind         (list)           : tuple carrying info about kind of cluster and its id
        filtered_data_df_valid   (pd.Dataframe)   : Dataframe carrying truth value of validity of cdd-consumption
        distance_comparison      (pd.Dataframe)   : Dataframe carrying cluster wise record of distances
        cluster_info             (dict)           : Dictionary containing cluster related key information
        filtered_data            (pd.Dataframe)   : Dataframe carrying filtered cdd/hdd and consumption
    Returns:
        filtered_data            (pd.Dataframe)   : Input Dataframe Updated with cluster ids
    """

    for cluster, kind in cluster_and_kind:
        if len(filtered_data_df_valid) > 0:
            distance_comparison[cluster] = filtered_data_df_valid.apply(lambda row: get_distance(row['degree_day'], row['filter_cons'], kind, cluster_info[cluster]), axis=1)
        else:
            distance_comparison[cluster] = [None] * len(filtered_data_df)

    distance_comparison['minimum'] = distance_comparison.min(axis=1)

    for cluster_id in distance_comparison.columns[:-1]:
        distance_comparison[cluster_id][~(distance_comparison[cluster_id] == distance_comparison['minimum'])] = np.nan
        distance_comparison[cluster_id][distance_comparison[cluster_id] == distance_comparison['minimum']] = cluster_id

    del distance_comparison['minimum']
    distance_comparison = distance_comparison.replace(np.nan, -1000)
    hvac_clusters = distance_comparison.max(axis=1).astype(int).tolist()

    return hvac_clusters


def get_hvac_clusters_mtd(filtered_data, cluster_info):
    """
    Function to update filtered dataframe with cluster id in mtd mode
    Parameters:
        filtered_data   (np.ndarray)    : Array carrying filtered cdd/hdd and consumption
        cluster_info    (dict)          : Dictionary containing cluster related key information
    Returns:
        filtered_data   (pd.Dataframe)  : Dataframe carrying filtered cdd/hdd and consumption with Updated cluster ids
    """

    cluster_ids = []
    cluster_valid = []
    cluster_kind = []

    filtered_data_df = pd.DataFrame(filtered_data, columns=['degree_day', 'filter_cons', 'filter_day'])

    cluster_info = cluster_info['hvac']

    for cluster_id in cluster_info.keys():
        cluster_ids.append(cluster_id)
        cluster_valid.append(cluster_info[cluster_id]['validity'])
        cluster_kind.append(cluster_info[cluster_id]['regression_kind'])

    distance_comparison = pd.DataFrame()

    if any(cluster_valid):

        filtered_data_df_valid = filtered_data_df[filtered_data_df['filter_day'] == 1]

        valid_clusters = np.array(cluster_ids)[np.array(cluster_valid)]
        valid_kind = np.array(cluster_kind)[np.array(cluster_valid)]
        cluster_and_kind = list(zip(valid_clusters,valid_kind))

        if len(cluster_and_kind) == 1:

            hvac_clusters = [cluster_and_kind[0][0]] * filtered_data_df.shape[0]

        else:

            hvac_clusters = assign_cluster_by_distance(cluster_and_kind, filtered_data_df_valid, distance_comparison, cluster_info, filtered_data_df)

        filtered_data_df['day_hvac_cluster'] = [None if flag == 0 else hvac_clusters.pop(0) for flag in filtered_data_df['filter_day']]
    else:
        filtered_data_df['day_hvac_cluster'] = [None] * len(filtered_data_df)

    return filtered_data_df
