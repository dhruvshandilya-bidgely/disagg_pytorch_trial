"""
Author - Mayank Sharan
Date - 22/1/19
Compute estimate for single run pool pump
"""

# Import python packages

import numpy as np
import copy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_consumption_matrix

from python3.disaggregation.aer.poolpump.functions.estimation_utils import score_all_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils import post_process_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils import fill_virtual_edges
from python3.disaggregation.aer.poolpump.functions.estimation_utils import build_confidence_score

from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import find_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import check_pair_prim
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import reject_poolpump
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import fill_uniform_amplitude
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import create_structure_for_hsm
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_amp_ballpark_from_uncontested_matrix


from python3.disaggregation.aer.poolpump.functions.estimation_utils_3 import match_duration
from python3.disaggregation.aer.poolpump.functions.estimation_utils_3 import scoring_multiple_pairs


def get_single_run_estimation(input_dict, days_label, filled_days, duration_each_day, num_of_runs_each_day, pp_config):
    """
    TO be populated once we figure out the best way to pass parameters in here
    :return:
    """

    # Extract constants form config

    window_size = 30
    step_size = 30
    probability_threshold = 0.45

    uncontested_matrix = input_dict['uncontested_matrix']
    data_clean_edges = input_dict['data_clean_edges']
    data_nms = input_dict['data_nms']
    data_bl_removed = input_dict['data_bl_removed']
    num_of_runs_matrix_from_uncontested = input_dict['num_of_runs_matrix_from_uncontested']
    runs_tuple = input_dict['runs_tuple']
    all_smooth_pos_edges = input_dict['all_smooth_pos_edges']
    all_smooth_neg_edges = input_dict['all_smooth_neg_edges']

    day_seasons = input_dict['day_seasons']

    # Get few other constants needed for the run

    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))
    num_rows, num_cols = data_clean_edges.shape

    # Get rough estimate of amplitude

    amp_ballpark = get_amp_ballpark_from_uncontested_matrix(uncontested_matrix, pp_config)

    # Get indices of multiple runs

    multiple_idx = np.where((np.sum(num_of_runs_matrix_from_uncontested[:, 2:5], axis=1) > 0) |
                            (num_of_runs_matrix_from_uncontested[:, 5] > 0))[0]

    # Get indices with no runs

    no_run_idx = np.where(num_of_runs_matrix_from_uncontested[:, 0] > 0)[0]
    idx_values = np.union1d(multiple_idx, no_run_idx)

    # Modify inputs as per the above calculated indices

    uncontested_matrix[idx_values, :] = 0
    days_label[multiple_idx] = 2

    # Sort of reinitialize variables for contentious values to be populated ahead

    filled_days[idx_values] = 0
    duration_each_day[multiple_idx] = 0
    num_of_runs_each_day[:] = 1

    new_added_pairs = []

    # Iterate over windows to get final uncontested matrix that will be used to get consumption

    for window in range(0, num_rows, step_size):
        input_dict['window'] = window
        input_dict['amp_ballpark'] = amp_ballpark

        filled_previous_iterations = np.zeros_like(uncontested_matrix[window:min(num_rows, (window + window_size))])

        # Perform primary pair check

        uncontested_matrix = check_pair_prim(input_dict, data_nms, days_label, duration_each_day,
                                             num_of_runs_each_day, filled_days, filled_previous_iterations, pp_config)

        # Match future and past duration

        uncontested_matrix = match_duration(input_dict, days_label, duration_each_day, num_of_runs_each_day,
                                            filled_days, filled_previous_iterations)

        # Score pairs in multiple pairs to remove ahead

        uncontested_matrix = scoring_multiple_pairs(input_dict, days_label, duration_each_day, num_of_runs_each_day,
                                                    filled_days, filled_previous_iterations)

        # mark all contested days as EMPTY

        window_days_label = days_label[window:min(num_rows, (window + window_size))]
        window_days_label[window_days_label == 2] = 0

        uncontested_matrix, new_added_pairs = fill_virtual_edges(input_dict, uncontested_matrix, new_added_pairs,
                                                                 days_label, duration_each_day, num_of_runs_each_day,
                                                                 filled_days, filled_previous_iterations, pp_config)

        uncontested_matrix, new_added_pairs = score_all_pairs(input_dict, all_smooth_pos_edges, all_smooth_neg_edges,
                                                              new_added_pairs, filled_days, filled_previous_iterations,
                                                              pp_config)

    # Extract final pairs to be used in filling consumption

    uncontested_matrix = fill_uniform_amplitude(uncontested_matrix)
    pp_pairs = find_pairs(uncontested_matrix, pp_config)
    processed_pairs = post_process_pairs(uncontested_matrix, data_clean_edges, data_bl_removed, pp_pairs, num_cols,
                                         samples_per_hour, pp_config)

    edges_matrix = np.zeros_like(uncontested_matrix)
    for pair in processed_pairs:
        edges_matrix[pair[2]:pair[3], pair[0]] = pair[4]
        edges_matrix[pair[2]:pair[3], pair[1]] = -pair[5]

    confidence_score, pp_run_days_arr = build_confidence_score(data_nms, processed_pairs, pp_config)
    reject_winter_pp_bool = reject_poolpump(day_seasons, pp_run_days_arr)

    if reject_winter_pp_bool:
        confidence_score = str(confidence_score) + "W"
        runs_tuple = ('NoRun', 0, confidence_score)
    elif confidence_score < probability_threshold:
        confidence_score = str(confidence_score)
        runs_tuple = ('NoRun', 0, confidence_score)
    else:
        confidence_score = str(confidence_score)
        runs_tuple = runs_tuple + (confidence_score,)

    data_bl_removed_copy = copy.deepcopy(data_bl_removed)
    for pair in processed_pairs:
        pair_time_div_range = np.arange(pair[0], pair[1] + 1) % num_cols
        duration = 1
        if samples_per_hour != 4:
            duration = 2
        time_div_range = np.arange(pair[0] - duration * samples_per_hour,
                                   pair[1] + duration * samples_per_hour + 1) % num_cols
        data_bl_removed_pair = data_bl_removed_copy[pair[2]:pair[3], pair_time_div_range]
        data_bl_removed_df = data_bl_removed_copy[pair[2]:pair[3], time_div_range]
        for day in range(len(data_bl_removed_df)):
            day_df = data_bl_removed_df[day, :]
            min_val = 0
            if np.sum(day_df) != 0:
                min_val = np.min(day_df)
            data_bl_removed_pair[day, :] -= min_val
        data_bl_removed_pair[data_bl_removed_pair < 0] = 0
        data_bl_removed_copy[pair[2]:pair[3], pair_time_div_range] = data_bl_removed_pair

    consumption_matrix, cons_threshold = get_consumption_matrix(processed_pairs, data_bl_removed_copy, data_bl_removed,
                                                                samples_per_hour,
                                                                pp_config)

    structures = create_structure_for_hsm(processed_pairs, cons_threshold, samples_per_hour)

    return edges_matrix, runs_tuple, consumption_matrix, pp_run_days_arr, structures
