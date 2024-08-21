"""
Author - Mayank Sharan
Date - 21/1/19
Get pool pump model for the given inputs
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import label_edges
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_probability
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_days_labeled
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_num_of_runs_each_day
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_runs_from_uncontested
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import find_potential_matches_all
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_run_probability_matrix
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import label_edges_time_mask_score
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import remove_small_uncontested_pairs
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import change_days_label_for_weak_runs

from python3.disaggregation.aer.poolpump.functions.get_pp_estimation import get_pp_estimation


def get_run_type(prob_single_run, prob_multiple_run, prob_var_speed, runs, min_days_for_multiple_run,
                 probability_threshold, min_max_run_days):
    """Utility to decide the type of run"""

    runs_mode = -1
    runs_max = -1

    if prob_multiple_run >= probability_threshold and np.sum(runs[2:]) >= min_days_for_multiple_run:

        run_type = 'multiple'
        runs_mode = np.argmax(runs[2:5]) + 2
        runs_max = np.where((runs[2:5] > 0) & (runs[2:5] > min_max_run_days))[0][-1] + 2
        runs_tuple = ('Multiple', runs_mode, runs_max)

    elif prob_var_speed >= probability_threshold:

        run_type = 'variable'
        runs_tuple = ('Variable', 1)

    elif prob_single_run >= probability_threshold:

        run_type = 'single'
        runs_tuple = ('Single', 1)

    else:

        run_type = 'none'
        runs_tuple = ('NoRun', 0, '0')

    return run_type, runs_tuple, runs_mode, runs_max


def get_pp_model(data_clean_edges, data_nms, data_bl_removed, day_seasons, pp_config):
    """
    Parameters:
        data_clean_edges    (np.ndarray)        : Day wise data matrix after edges cleaning
        data_nms            (np.ndarray)        : Day wise data matrix after non maximal suppression
        data_bl_removed     (np.ndarray)        : Day wise data matrix after baseload removal
        day_seasons         (np.ndarray)        : Array containing mapping of day to season
        pp_config           (dict)              : Dictionary containing all configuration variables for pool pump

    Returns:
        data_pp_cons
        data_pp_steps
        data_pp_diff
        runs_data
    """

    # Extract constants out of config file

    min_max_run_days = pp_config.get('min_max_run_days')
    probability_threshold = pp_config.get('probability_threshold')
    non_winter_run_threshold = pp_config.get('non_winter_run_threshold')
    min_days_for_multiple_run = pp_config.get('min_days_for_multiple_run')

    num_samples_per_hr = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))

    # Initialise copy of all inputs

    data_clean_edges = copy.deepcopy(data_clean_edges)
    data_bl_removed = copy.deepcopy(data_bl_removed)
    data_nms = copy.deepcopy(data_nms)
    num_rows, num_cols = data_clean_edges.shape

    # Extract run data out of clean edges data

    clean_pos_edges, clean_neg_edges = label_edges(data_clean_edges, pp_config)
    clean_pairs, _, _ = find_potential_matches_all(data_bl_removed, clean_pos_edges, clean_neg_edges, num_cols,
                                                   num_samples_per_hr, pp_config)

    filtered_data = np.zeros_like(data_clean_edges)

    for pair in clean_pairs:
        filtered_data[pair[2]:pair[3], clean_pos_edges[pair[0]][2]] = clean_pos_edges[pair[0]][3]
        filtered_data[pair[2]:pair[3], clean_neg_edges[pair[1]][2]] = -(clean_neg_edges[pair[1]][3])

    # Compute run probability matrix

    time_div_dict, pp_run_matrix = get_run_probability_matrix(filtered_data, data_nms, num_samples_per_hr, pp_config)

    # Perform label edge analysis on clean edges and nms data

    all_pos_edges, all_neg_edges = label_edges_time_mask_score(data_clean_edges, time_div_dict, pp_config)
    all_smooth_pos_edges, all_smooth_neg_edges = label_edges_time_mask_score(data_nms, time_div_dict,
                                                                             pp_config, min_edge_length=3)

    all_pairs, _, duration_each_day = find_potential_matches_all(data_bl_removed, all_pos_edges, all_neg_edges,
                                                                 num_cols, num_samples_per_hr, pp_config)

    all_pairs_matrix = np.zeros_like(data_clean_edges)

    for pair in all_pairs:
        all_pairs_matrix[pair[2]:pair[3], all_pos_edges[pair[0]][2]] = all_pos_edges[pair[0]][3]
        all_pairs_matrix[pair[2]:pair[3], all_neg_edges[pair[1]][2]] = -(all_neg_edges[pair[1]][3])

    global_pos_edges, global_neg_edges = all_pos_edges.copy(), all_neg_edges.copy()
    global_pairs = all_pairs.copy()

    # Get number of runs in a day

    num_of_runs_each_day = get_num_of_runs_each_day(all_pairs_matrix)

    # Initialize variables to send as steps

    check_in_smooth_uncontested = np.zeros_like(data_clean_edges)
    past_and_future_duration_check = np.zeros_like(data_clean_edges)
    scoring_uncontested = np.zeros_like(data_clean_edges)

    steps = dict()
    steps[1] = check_in_smooth_uncontested
    steps[2] = past_and_future_duration_check
    steps[3] = scoring_uncontested

    # Initialise consumption matrix and diff matrix

    consumption_matrix = np.zeros_like(data_clean_edges)

    # Label days using pairs

    days_label = get_days_labeled(all_pairs_matrix, num_of_runs_each_day, num_rows)
    days_label = change_days_label_for_weak_runs(num_of_runs_each_day, days_label, pp_config)
    days_label = remove_small_uncontested_pairs(days_label, pp_config)

    # Get uncontested matrix

    filled_days = np.zeros(num_rows)

    duration_each_day = np.where(days_label != 1, 0, duration_each_day)
    uncontested_matrix = np.zeros_like(data_clean_edges)
    uncontested_days = np.where(days_label == 1)[0]

    if len(uncontested_days) > 0:
        filled_days[uncontested_days] = 1
        uncontested_matrix[uncontested_days, :] = all_pairs_matrix[uncontested_days, :]

    steps[2] = uncontested_matrix.copy()

    structures = np.array([])

    if len(all_pairs) == 0:
        return consumption_matrix, steps, ('NoRun', 0, '0'), structures

    # Extract number of runs from uncontested matrix

    num_of_runs_matrix_from_uncontested = get_runs_from_uncontested(uncontested_matrix, num_cols, num_samples_per_hr,
                                                                    pp_config)
    runs = np.sum(num_of_runs_matrix_from_uncontested, axis=0)

    # Get probability of different kind of runs

    prob_single_run = get_probability(pp_run_matrix, 'single', 1, pp_config)
    if 0 < prob_single_run < probability_threshold:
        pp_run_days_arr = np.sum(np.abs(np.sign(filtered_data)), axis=1)
        if len(day_seasons) < len(pp_run_days_arr):
            day_seasons = np.r_[day_seasons,
                                np.full(shape=(len(pp_run_days_arr) - len(day_seasons),), fill_value=day_seasons[-1])]
        elif len(day_seasons) > len(pp_run_days_arr):
            day_seasons = day_seasons[:len(pp_run_days_arr)]
        non_winter_run_days = np.where((pp_run_days_arr != 0) & (day_seasons != 1))[0]
        num_of_pp_run_days = np.where(pp_run_days_arr != 0)[0]
        non_winter_run_fraction = len(non_winter_run_days) / len(num_of_pp_run_days)
        if non_winter_run_fraction >= non_winter_run_threshold:
            prob_single_run = 1

    prob_multiple_run = 0
    prob_var_speed = 0

    if prob_single_run >= probability_threshold:
        prob_multiple_run = get_probability(pp_run_matrix, 'multiple', 2, pp_config)
        prob_var_speed = get_probability(pp_run_matrix, 'variable', 3, pp_config)

    # Get the run type based on different probabilities

    run_type, runs_tuple, runs_mode, runs_max = get_run_type(prob_single_run, prob_multiple_run, prob_var_speed, runs,
                                                             min_days_for_multiple_run, probability_threshold,
                                                             min_max_run_days)

    input_dict = dict()
    input_dict['uncontested_matrix'] = uncontested_matrix
    input_dict['data_clean_edges'] = data_clean_edges
    input_dict['data_nms'] = data_nms
    input_dict['data_bl_removed'] = data_bl_removed
    input_dict['num_of_runs_matrix_from_uncontested'] = num_of_runs_matrix_from_uncontested
    input_dict['time_div_dict'] = time_div_dict
    input_dict['runs_tuple'] = runs_tuple
    input_dict['all_pos_edges'] = all_pos_edges
    input_dict['all_neg_edges'] = all_neg_edges
    input_dict['all_pairs'] = all_pairs
    input_dict['all_smooth_pos_edges'] = all_smooth_pos_edges
    input_dict['all_smooth_neg_edges'] = all_smooth_neg_edges
    input_dict['global_pos_edges'] = global_pos_edges
    input_dict['global_neg_edges'] = global_neg_edges
    input_dict['global_pairs'] = global_pairs
    input_dict['day_seasons'] = day_seasons

    consumption_matrix, steps, runs_tuple, structures = get_pp_estimation(input_dict, steps, num_of_runs_each_day,
                                                                          days_label, duration_each_day, filled_days,
                                                                          pp_config)

    return consumption_matrix, steps, runs_tuple, structures
