"""
Author - Mayank Sharan
Date - 22/1/19
Compute estimate for multiple run pool pump
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import find_potential_matches_all
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import label_edges_time_mask_score

from python3.disaggregation.aer.poolpump.functions.estimation_utils import get_section_int
from python3.disaggregation.aer.poolpump.functions.estimation_utils import score_all_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils import post_process_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils import fill_virtual_edges
from python3.disaggregation.aer.poolpump.functions.estimation_utils import get_best_scored_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils import build_confidence_score
from python3.disaggregation.aer.poolpump.functions.estimation_utils import get_filled_previous_iterations
from python3.disaggregation.aer.poolpump.functions.estimation_utils import get_overlapping_pairs_sections

from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import find_pairs
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import reject_poolpump
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import check_pair_prim
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import find_run_filled_days
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import fill_uniform_amplitude
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import retain_time_mask_score
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_global_pairs_matrix
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import restore_time_mask_score
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import create_structure_for_hsm
from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import get_amp_ballpark_from_uncontested_matrix

from python3.disaggregation.aer.poolpump.functions.estimation_utils_3 import get_pairs_idx
from python3.disaggregation.aer.poolpump.functions.estimation_utils_3 import match_duration
from python3.disaggregation.aer.poolpump.functions.estimation_utils_3 import scoring_multiple_pairs


def get_multiple_run_estimation(input_dict, days_label, filled_days, duration_each_day, num_of_runs_each_day,
                                overlap_arr, cons_threshold_arr, pp_config):
    """
    TO be populated once we figure out the best way to pass parameters in here
    :return:
    """

    # Extract constants out of config file

    window_size = pp_config.get('window_size')
    step_size = pp_config.get('window_step_size')
    probability_threshold = pp_config.get('probability_threshold')

    uncontested_matrix = input_dict['uncontested_matrix']
    data_clean_edges = input_dict['data_clean_edges']
    data_nms = input_dict['data_nms']
    data_bl_removed = input_dict['data_bl_removed']
    runs_tuple = input_dict['runs_tuple']
    day_seasons = input_dict['day_seasons']
    global_pos_edges = input_dict['global_pos_edges']
    global_neg_edges = input_dict['global_neg_edges']
    global_pairs = input_dict['global_pairs']
    all_pos_edges = input_dict['all_pos_edges']
    all_neg_edges = input_dict['all_neg_edges']

    # Get few other constants needed for the run

    mtd_days_idx = np.where(overlap_arr != 0)[0]

    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))
    num_rows, num_cols = data_clean_edges.shape

    # Get rough estimate of amplitude

    amp_ballpark = get_amp_ballpark_from_uncontested_matrix(uncontested_matrix, pp_config)
    original_uncontested_matrix = uncontested_matrix.copy()

    # Start running for multiple run

    pos_time_mask_df, neg_time_mask_df = retain_time_mask_score(all_pos_edges, all_neg_edges, num_rows, num_cols)
    new_added_pairs = list()

    start_day = mtd_days_idx[0]
    for window in range(start_day, num_rows, step_size):
        clean_union_copy = data_clean_edges.copy()
        filled_previous_iterations = np.zeros_like(uncontested_matrix[window:min(num_rows, (window + window_size))])

        input_dict['clean_union_copy'] = clean_union_copy
        input_dict['amp_ballpark'] = amp_ballpark
        input_dict['pos_time_mask_df'] = pos_time_mask_df
        input_dict['neg_time_mask_df'] = neg_time_mask_df
        input_dict['window'] = window

        uncontested_matrix = iterative_window_filling(input_dict, filled_days, days_label,
                                                      num_of_runs_each_day, duration_each_day,
                                                      filled_previous_iterations, new_added_pairs, pp_config)

    new_added_pairs = np.array(new_added_pairs, dtype=int)
    global_pairs_matrix = get_global_pairs_matrix(global_pos_edges, global_neg_edges, global_pairs)
    if len(new_added_pairs) == 0:
        new_added_pairs = np.empty(shape=(0, global_pairs_matrix.shape[1]))
    global_pairs_matrix = np.concatenate((global_pairs_matrix, new_added_pairs), axis=0)

    days_filled_with_uncommon_runs = find_run_filled_days(
        (uncontested_matrix - original_uncontested_matrix), runs_tuple[1])

    uncommon_days_arr = np.zeros(num_rows)
    uncommon_days_arr[days_filled_with_uncommon_runs] = 1
    uncommon_days_arr[:mtd_days_idx[0]] = 0
    start_arr, end_arr = find_edges(uncommon_days_arr)

    for idx in range(len(start_arr)):
        pairs_idx, grouped_schedules = get_pairs_idx(uncontested_matrix, start_arr[idx], end_arr[idx],
                                                     global_pairs_matrix)
        pairs = global_pairs_matrix[pairs_idx]
        time_div_list = [time_div for schedule_tup in grouped_schedules for time_div in schedule_tup]

        uncontested_matrix = get_best_scored_pairs(uncontested_matrix, start_arr[idx], end_arr[idx], time_div_list,
                                                   pairs, grouped_schedules, runs_tuple[1])

    # Extract final pairs to be used in filling consumption

    uncontested_matrix = uncontested_matrix[mtd_days_idx]
    data_bl_removed = data_bl_removed[mtd_days_idx]
    data_clean_edges = data_clean_edges[mtd_days_idx]
    uncontested_matrix = fill_uniform_amplitude(uncontested_matrix)
    pp_pairs = find_pairs(uncontested_matrix, pp_config)
    processed_pairs = post_process_pairs(uncontested_matrix, data_clean_edges, data_bl_removed, pp_pairs, num_cols,
                                         samples_per_hour, pp_config)

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

    consumption_matrix = np.zeros_like(data_bl_removed)
    cons_threshold = np.mean(cons_threshold_arr) / samples_per_hour
    for pair in processed_pairs:
        avg_consumption = ((pair[4] + pair[5]) / 2) / samples_per_hour

        consumption = max(cons_threshold, avg_consumption)

        # fill output matrix

        duration = (pair[1] - pair[0]) % num_cols
        time_div_reqd = np.arange(pair[0], pair[0] + duration + 1) % num_cols

        df_required = data_bl_removed[pair[2]:pair[3], time_div_reqd]
        consumption_matrix[pair[2]:pair[3], time_div_reqd] = np.where(df_required <= consumption, df_required,
                                                                      consumption)

    structures = create_structure_for_hsm(processed_pairs, cons_threshold, samples_per_hour)

    return uncontested_matrix, runs_tuple, consumption_matrix, structures


def iterative_window_filling(input_dict, filled_days, days_label, num_of_runs_each_day, duration_each_day,
                             filled_previous_iterations, new_added_pairs, pp_config):
    """Window-wise PP detection, iterated "num_of_runs" times"""

    window_size = 30
    min_window_filled_fraction = 0.7
    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))

    uncontested_matrix = input_dict['uncontested_matrix']
    clean_union_copy = input_dict['clean_union_copy']
    data_bl_removed = input_dict['data_bl_removed']
    pos_time_mask_df = input_dict['pos_time_mask_df']
    neg_time_mask_df = input_dict['neg_time_mask_df']
    time_div_dict = input_dict['time_div_dict']
    runs_tuple = input_dict['runs_tuple']
    window = input_dict['window']

    num_rows, num_cols = uncontested_matrix.shape

    for temp_idx in range(runs_tuple[1]):
        common_run_filled_days = \
            find_run_filled_days(uncontested_matrix[window:min(num_rows, window + window_size)], runs_tuple[1],
                                 common_flag=True)

        common_run_filled_days += window
        filled_days[common_run_filled_days] = 1
        days_label[common_run_filled_days] = 1

        all_pos_edges, all_neg_edges = label_edges_time_mask_score(clean_union_copy, time_div_dict, pp_config)
        all_pos_edges, all_neg_edges = restore_time_mask_score(pos_time_mask_df, neg_time_mask_df, all_pos_edges,
                                                               all_neg_edges)

        all_pairs, _, _ = find_potential_matches_all(data_bl_removed, all_pos_edges, all_neg_edges, num_cols,
                                                     samples_per_hour, pp_config)

        if len(all_pairs) == 0:
            continue

        sections = get_overlapping_pairs_sections(window, window_size, all_pairs, all_pos_edges, all_neg_edges,
                                                  num_cols)

        if len(sections) == 0:
            num_of_days_filled_in_window = len(
                np.where(np.sum(abs(uncontested_matrix[window:min(num_rows, (window + window_size))]), axis=1))[0])
            window_filled_fraction = num_of_days_filled_in_window / (min(num_rows, (window + window_size)) - window)
            if window_filled_fraction > min_window_filled_fraction:
                continue
            else:
                sections = [(0, num_cols-1)]

        uncontested_matrix, filled_previous_iterations = sectional_filling(
            input_dict, sections, days_label, filled_days, num_of_runs_each_day, duration_each_day,
            filled_previous_iterations, new_added_pairs, pp_config)

    return uncontested_matrix


def sectional_filling(input_dict, sections, days_label, filled_days, num_of_runs_each_day, duration_each_day,
                      filled_previous_iterations, new_added_pairs, pp_config):
    """Fill each section, i.e. run, of multiple run PP"""

    window_size = 30

    uncontested_matrix = input_dict['uncontested_matrix']
    clean_union_copy = input_dict['clean_union_copy']
    data_nms = input_dict['data_nms']
    data_bl_removed = input_dict['data_bl_removed']
    time_div_dict = input_dict['time_div_dict']
    window = input_dict['window']

    num_rows, num_cols = uncontested_matrix.shape
    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))

    for section in sections:
        section_int = get_section_int(section)
        days_label_copy = days_label.copy()
        filled_days_copy = filled_days.copy()
        num_of_runs_each_day_copy = num_of_runs_each_day.copy()
        num_of_runs_each_day_copy[:] = 1
        duration_each_day_copy = duration_each_day.copy()
        if section[0] < section[1]:
            clean_union_copy_section = clean_union_copy.copy()
            smooth_nms_copy = data_nms.copy()

            sectional_clean_union = np.zeros_like(clean_union_copy)
            sectional_smooth_edges = np.zeros_like(clean_union_copy)
            sectional_clean_union[:, section[0]:(section[1] + 1)] = \
                clean_union_copy_section[:, section[0]:(section[1] + 1)]
            sectional_smooth_edges[:, section[0]:(section[1] + 1)] = \
                smooth_nms_copy[:, section[0]:(section[1] + 1)]
        else:
            clean_union_copy_section = clean_union_copy.copy()
            smooth_nms_copy = data_nms.copy()
            sectional_clean_union = np.zeros_like(clean_union_copy)
            sectional_smooth_edges = np.zeros_like(clean_union_copy)
            sectional_clean_union[:, section[0]:] = clean_union_copy_section[:, section[0]:]
            sectional_clean_union[:, :(section[1] + 1)] = clean_union_copy_section[:, :(section[1] + 1)]
            sectional_smooth_edges[:, section[0]:] = smooth_nms_copy[:, section[0]:]
            sectional_smooth_edges[:, :(section[1] + 1)] = smooth_nms_copy[:, :(section[1] + 1)]

        sectional_all_pos_edges, sectional_all_neg_edges = label_edges_time_mask_score(sectional_clean_union,
                                                                                       time_div_dict, pp_config)

        sectional_all_smooth_pos_edges, sectional_all_smooth_neg_edges = label_edges_time_mask_score(
            sectional_smooth_edges, time_div_dict, pp_config, min_edge_length=3)

        if len(sectional_all_pos_edges) == 0 or len(sectional_all_neg_edges) == 0:
            continue
        sectional_all_pairs, _, _ = find_potential_matches_all(data_bl_removed, sectional_all_pos_edges,
                                                               sectional_all_neg_edges, num_cols,
                                                               samples_per_hour, pp_config)

        if len(sectional_all_pairs) == 0:
            continue

        input_dict['all_pos_edges'] = sectional_all_pos_edges
        input_dict['all_neg_edges'] = sectional_all_neg_edges
        input_dict['all_pairs'] = sectional_all_pairs
        input_dict['uncontested_matrix'] = uncontested_matrix

        sectional_uncontested_matrix = check_pair_prim(input_dict, sectional_smooth_edges, days_label_copy,
                                                       duration_each_day_copy, num_of_runs_each_day_copy,
                                                       filled_days_copy, filled_previous_iterations, pp_config)

        section_duration = (section[1] - section[0]) % num_cols
        section_time_div_arr = (np.arange(section[0], section[0] + section_duration + 1)) % num_cols
        uncontested_matrix[:, section_time_div_arr] = sectional_uncontested_matrix[:, section_time_div_arr]

        sectional_uncontested_matrix = match_duration(input_dict, days_label_copy, duration_each_day_copy,
                                                      num_of_runs_each_day_copy, filled_days_copy,
                                                      filled_previous_iterations, section=section_int)

        section_duration = (section[1] - section[0]) % num_cols
        section_time_div_arr = (np.arange(section[0], section[0] + section_duration + 1)) % num_cols
        uncontested_matrix[:, section_time_div_arr] = sectional_uncontested_matrix[:, section_time_div_arr]

        sectional_uncontested_matrix = scoring_multiple_pairs(input_dict, days_label_copy, duration_each_day_copy,
                                                              num_of_runs_each_day_copy, filled_days_copy,
                                                              filled_previous_iterations, section=section_int)

        section_duration = (section[1] - section[0]) % num_cols
        section_time_div_arr = (np.arange(section[0], section[0] + section_duration + 1)) % num_cols
        uncontested_matrix[:, section_time_div_arr] = sectional_uncontested_matrix[:, section_time_div_arr]

        window_days_label = days_label_copy[window:min(num_rows, (window + window_size))]
        window_days_label[window_days_label == 2] = 0

        sectional_uncontested_matrix, new_added_pairs = fill_virtual_edges(input_dict, uncontested_matrix,
                                                                           new_added_pairs, days_label_copy,
                                                                           duration_each_day_copy,
                                                                           num_of_runs_each_day_copy, filled_days_copy,
                                                                           filled_previous_iterations, pp_config,
                                                                           section=section_int)

        section_duration = (section[1] - section[0]) % num_cols
        section_time_div_arr = (np.arange(section[0], section[0] + section_duration + 1)) % num_cols
        uncontested_matrix[:, section_time_div_arr] = sectional_uncontested_matrix[:, section_time_div_arr]

        sectional_uncontested_matrix, new_added_pairs = score_all_pairs(input_dict,
                                                                        sectional_all_smooth_pos_edges,
                                                                        sectional_all_smooth_neg_edges,
                                                                        new_added_pairs, filled_days_copy,
                                                                        filled_previous_iterations, pp_config,
                                                                        section=section_int)

        section_duration = (section[1] - section[0]) % num_cols
        section_time_div_arr = (np.arange(section[0], section[0] + section_duration + 1)) % num_cols
        uncontested_matrix[:, section_time_div_arr] = sectional_uncontested_matrix[:, section_time_div_arr]

        section_duration = (section[1] - section[0]) % num_cols
        section_time_div_arr = (np.arange(section[0], section[0] + section_duration + 1)) % num_cols
        edges_added = uncontested_matrix[window:
                                         min(window + window_size, num_rows), section_time_div_arr]

        clean_union_copy[window:min(window + window_size, num_rows), section_time_div_arr] = np.where(
            ((clean_union_copy[window:min(window + window_size, num_rows), section_time_div_arr] != 0) &
             (edges_added != 0)), 0,
            clean_union_copy[window:min(window + window_size, num_rows), section_time_div_arr])

    filled_previous_iterations = get_filled_previous_iterations(
        uncontested_matrix[window:min(num_rows, (window + window_size))])

    return uncontested_matrix, filled_previous_iterations
