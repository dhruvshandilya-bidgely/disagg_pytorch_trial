"""
Author - Arpan Agrawal
Date - 21/2/19
This module runs the pool pump disaggregation and returns consumption value in MTD mode
"""

import copy
import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import label_edges
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_num_of_runs_each_day
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import find_potential_matches_all
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_run_probability_matrix
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import label_edges_time_mask_score

from python3.disaggregation.aer.poolpump.functions.estimation_utils import build_confidence_score

from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import reject_poolpump

from python3.disaggregation.aer.poolpump.functions.mtd_single_run_estimation import get_single_run_estimation
from python3.disaggregation.aer.poolpump.functions.get_variable_pp_estimation import get_variable_pp_estimation
from python3.disaggregation.aer.poolpump.functions.mtd_multiple_run_estimation import get_multiple_run_estimation


def get_pp_model(hsm_input_dict, data_clean_edges, data_nms, data_bl_removed, day_seasons, run_type, num_of_runs,
                 pp_config):
    """
    Parameters:
        hsm_input_dict      (dict)              : Dictionary containing PP history from hsm
        data_clean_edges    (np.ndarray)        : Day wise data matrix after edges cleaning
        data_nms            (np.ndarray)        : Day wise data matrix after non maximal suppression
        data_bl_removed     (np.ndarray)        : Day wise data matrix after baseload removal
        day_seasons         (np.ndarray)        : Array containing mapping of day to season
        run_type            (int)               : Run type of PP encoded as Integer
        num_of_runs         (int)               : Number of runs of PP
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        data_pp_cons
        data_pp_steps
        data_pp_diff
        runs_data
    """

    num_samples_per_hr = int(3600 / pp_config.get('sampling_rate'))

    hsm_matrix = hsm_input_dict['hsm_matrix']
    hsm_raw_matrix = hsm_input_dict['hsm_raw_matrix']
    overlap_arr = hsm_input_dict['overlap_arr']
    cons_threshold_arr = hsm_input_dict['cons_threshold_arr']

    consumption_matrix = np.zeros_like(data_clean_edges)
    structures = np.array([])

    hsm_non_overlap_idx = np.where(overlap_arr == 0)[0]
    hsm_overlap_idx = np.where(overlap_arr == 1)[0]
    new_data_idx = np.where(overlap_arr == 2)[0]
    mtd_days_idx = np.where(overlap_arr != 0)[0]

    hsm_data_bl_removed = copy.deepcopy(hsm_raw_matrix)
    hsm_data_bl_removed = np.vstack((hsm_data_bl_removed[hsm_non_overlap_idx], data_bl_removed))

    num_rows, num_cols = hsm_data_bl_removed.shape

    hsm_data_clean_edges = copy.deepcopy(hsm_matrix)
    hsm_data_clean_edges[hsm_overlap_idx] = np.where(hsm_data_clean_edges[hsm_overlap_idx] != 0,
                                                     hsm_data_clean_edges[hsm_overlap_idx],
                                                     data_clean_edges[hsm_overlap_idx])
    hsm_data_clean_edges = np.vstack((hsm_data_clean_edges, data_clean_edges[new_data_idx]))

    hsm_data_nms = copy.deepcopy(hsm_matrix)
    hsm_data_nms = np.vstack((hsm_data_nms[hsm_non_overlap_idx], data_nms))

    uncontested_matrix = copy.deepcopy(hsm_data_clean_edges)
    uncontested_matrix[mtd_days_idx] = 0

    days_label = np.zeros_like(overlap_arr)
    filled_days = np.zeros_like(overlap_arr)
    days_label[hsm_non_overlap_idx] = 1
    filled_days[hsm_non_overlap_idx] = 1
    days_label[days_label != 1] = 2

    clean_pos_edges, clean_neg_edges = label_edges(data_clean_edges, pp_config)
    clean_pairs, _, _ = find_potential_matches_all(hsm_data_bl_removed, clean_pos_edges, clean_neg_edges, num_cols,
                                                   num_samples_per_hr, pp_config)

    filtered_data = np.zeros_like(hsm_data_clean_edges)

    for pair in clean_pairs:
        filtered_data[pair[2]:pair[3], clean_pos_edges[pair[0]][2]] = clean_pos_edges[pair[0]][3]
        filtered_data[pair[2]:pair[3], clean_neg_edges[pair[1]][2]] = -(clean_neg_edges[pair[1]][3])

    time_div_dict, pp_run_matrix = get_run_probability_matrix(filtered_data, data_nms, num_samples_per_hr, pp_config)
    num_of_runs_each_day = get_num_of_runs_each_day(filtered_data)

    all_pos_edges, all_neg_edges = label_edges_time_mask_score(hsm_data_clean_edges, time_div_dict, pp_config)
    all_smooth_pos_edges, all_smooth_neg_edges = label_edges_time_mask_score(hsm_data_nms, time_div_dict,
                                                                             pp_config, min_edge_length=3)
    all_pairs, _, _ = find_potential_matches_all(hsm_data_bl_removed, all_pos_edges, all_neg_edges,
                                                 num_cols, num_samples_per_hr, pp_config)
    global_pos_edges, global_neg_edges = all_pos_edges.copy(), all_neg_edges.copy()
    global_pairs = all_pairs.copy()

    num_of_runs_matrix_from_uncontested = np.ones(shape=(num_rows, 6))
    duration_each_day = np.ones(num_rows)

    if run_type == 0:
        run_type = 'NoRun'
    elif run_type == 1:
        run_type = 'Single'
    elif run_type == 2:
        run_type = 'Multiple'
    elif run_type == 3:
        run_type = 'Variable'
    runs_tuple = (run_type, num_of_runs)
    original_runs_tuple = (run_type, num_of_runs, '1')
    if run_type == 'Multiple':
        runs_tuple = (run_type, num_of_runs, num_of_runs)
        original_runs_tuple = (run_type, num_of_runs, num_of_runs, '1')

    steps = dict()
    steps[1] = np.zeros_like(hsm_data_clean_edges[new_data_idx])
    steps[2] = np.zeros_like(hsm_data_clean_edges[new_data_idx])
    steps[3] = np.zeros_like(hsm_data_clean_edges[new_data_idx])

    input_dict = dict()
    input_dict['uncontested_matrix'] = uncontested_matrix
    input_dict['data_clean_edges'] = hsm_data_clean_edges
    input_dict['data_nms'] = hsm_data_nms
    input_dict['data_bl_removed'] = hsm_data_bl_removed
    input_dict['runs_tuple'] = runs_tuple
    input_dict['day_seasons'] = day_seasons
    input_dict['global_pos_edges'] = global_pos_edges
    input_dict['global_neg_edges'] = global_neg_edges
    input_dict['global_pairs'] = global_pairs
    input_dict['time_div_dict'] = time_div_dict
    input_dict['num_of_runs_matrix_from_uncontested'] = num_of_runs_matrix_from_uncontested
    input_dict['all_pos_edges'] = all_pos_edges
    input_dict['all_neg_edges'] = all_neg_edges
    input_dict['all_pairs'] = all_pairs
    input_dict['all_smooth_pos_edges'] = all_smooth_pos_edges
    input_dict['all_smooth_neg_edges'] = all_smooth_neg_edges

    if run_type != 'Multiple':

        uncontested_matrix, runs_tuple, consumption_matrix, pp_run_days_arr, structures = get_single_run_estimation(
            input_dict, days_label, filled_days, duration_each_day, num_of_runs_each_day, overlap_arr,
            cons_threshold_arr, pp_config)

        steps[3] = uncontested_matrix

        single_run_consumption_matrix = consumption_matrix
        single_run_days_arr = pp_run_days_arr

        if run_type == 'Variable':

            var_pair_matrix, consumption_matrix, high_amp_pairs, structures = \
                get_variable_pp_estimation(hsm_data_clean_edges, hsm_data_nms, hsm_data_bl_removed, num_samples_per_hr,
                                           pp_config)

            steps[3] = var_pair_matrix

            confidence_score, vpp_run_days_arr = build_confidence_score(hsm_data_nms, high_amp_pairs, pp_config,
                                                                        variable=True)

            consumption_matrix = consumption_matrix[mtd_days_idx]
            vpp_run_days_arr = vpp_run_days_arr[mtd_days_idx]
            steps[3] = var_pair_matrix[mtd_days_idx]
            whitespaces_to_be_filled = np.where((single_run_days_arr != 0) & (vpp_run_days_arr == 0))[0]
            consumption_matrix[whitespaces_to_be_filled, :] = single_run_consumption_matrix[whitespaces_to_be_filled, :]

    elif run_type == 'Multiple':

        uncontested_matrix, runs_tuple, consumption_matrix, structures = get_multiple_run_estimation(
            input_dict, days_label, filled_days, duration_each_day, num_of_runs_each_day, overlap_arr,
            cons_threshold_arr, pp_config)

        steps[3] = uncontested_matrix

    return consumption_matrix, steps, original_runs_tuple, structures
