"""
Author - Mayank Sharan
Date - 30/1/19
Get pool pump estimate for variable speed pool pump
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import label_edges

from python3.disaggregation.aer.poolpump.functions.variable_pp_utils import get_patched_edges
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils import remove_unwanted_edges
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils import get_universal_details
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils import get_all_consistent_vpairs
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils import get_all_section_structures
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils import get_masked_time_divisions_vpp
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils import get_eligible_pairs_from_edges_for_vpp

from python3.disaggregation.aer.poolpump.functions.variable_pp_utils_2 import get_global_structure
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils_2 import create_all_runs_dataframe
from python3.disaggregation.aer.poolpump.functions.variable_pp_utils_2 import get_variable_run_amp_wise_pairs

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_consumption_matrix


def get_variable_pp_estimation(data_clean_edges, data_nms, data_bl_removed, num_samples_per_hr, pp_config):
    """

    :param data_clean_edges:
    :param data_nms:
    :param data_bl_removed:
    :param num_samples_per_hr:
    :param pp_config:
    :return:
    """

    #TODO: Handle for MTD days

    # Extract constants from config

    window_size = pp_config.get('window_size')
    step_size = pp_config.get('window_step_size')

    stype_ppnn = 22

    num_rows, num_cols = data_clean_edges.shape

    # All of this needs to be shifted to the main config

    vpp_local_config = {
        'samples_per_hour': num_samples_per_hr,
        'window_size': window_size,
        'step_size': step_size,
        'td_masking_min_edge_length': 30,
        'td_masking_min_filled_days': 100,
        'total_timediv_count': num_cols,
        'total_days_count': num_rows,
        'timediv_max_density_range': 5,
        'duration_outlier_limit': 20,
        'start_time_outlier_limit': 10,
        'end_time_outlier_limit': 10,
        'max_edge_patch_length': 15,
        'strong_edge_length': 30,
        'block_amp_ratio_epsilon': 0,
        'section_min_duration_in_hour': 12,
        'min_intersection_days': 1,
        'min_amplitude_ratio_threshold_high': 0.6,
        'min_amplitude_ratio_threshold_low': 0.5,
        'vpair_min_duration_in_hour': 1,
        'vpair_max_duration_in_hour': 18,
        'vpair_min_inter_duration_in_hour': 0.5,
        'vpair_amplitude_margin': 0.9,
        'vpair_relaxed_amplitude_margin': 0.7,
        'vpair_min_duration_fraction': 0.5,
        'vpair_min_day_fraction': 0.8,
        'vpair_min_window_fraction': 0.5,
        'score_weight_amplitude': 1,
        'score_weight_duration_fraction': 2,
        'score_weight_inner_window': 0.4,
        'score_weight_inner_day': 0.4,
        'score_weight_outer_window': 0.6,
        'score_weight_outer_day': 0.6,
        'score_weight_type': 0.01,
        'score_masked_td_fraction': 2,
        'score_global_vpp_type': 1

    }

    # Extract masked time divisions

    pos_masked_time_div, neg_masked_time_div = get_masked_time_divisions_vpp(vpp_local_config, data_clean_edges)

    # Remove Edges which are not in masked time divisions
    # Remove Positive Edges not within range of +/- 1 from masked time divisions

    timediv_count = vpp_local_config.get('total_timediv_count')

    pos_edge_removal_index = np.zeros(shape=(timediv_count,))

    pos_edge_removal_index[
        (np.concatenate((pos_masked_time_div, (pos_masked_time_div + 1) % timediv_count,
                         (pos_masked_time_div - 1) % timediv_count)))] = 1

    neg_edge_removal_index = np.zeros(shape=(timediv_count,))

    neg_edge_removal_index[
        (np.concatenate((neg_masked_time_div, (neg_masked_time_div + 1) % timediv_count,
                         (neg_masked_time_div - 1) % timediv_count)))] = 1

    # Get masked clean union array

    clean_union_masked_pos = copy.deepcopy(data_clean_edges)
    clean_union_masked_pos[clean_union_masked_pos < 0] = 0
    clean_union_masked_pos[:, np.where((pos_edge_removal_index == 0))] = 0

    clean_union_masked_neg = copy.deepcopy(data_clean_edges)
    clean_union_masked_neg[clean_union_masked_neg > 0] = 0
    clean_union_masked_neg[:, np.where((neg_edge_removal_index == 0))] = 0

    clean_union_masked = clean_union_masked_pos + clean_union_masked_neg

    # Section 2: Calculate Universal Duration, Start Time and End Time Bucket

    day_arr, duration_arr, start_time_arr, end_time_arr = get_universal_details(clean_union_masked, vpp_local_config)

    # Section 3: Create Buckets for start_time, end_time and Duration and Catch most prevalant Bucket
    # Find start_time, and end_time densities and pin point the location of start and end time

    start_time_density = np.zeros(vpp_local_config.get('total_timediv_count'))

    # Get unique and frequencies from start time arr

    start_time_arr_unique_vals, start_time_arr_unique_vals_frequencies = np.unique(np.array(start_time_arr),
                                                                                   return_counts=True)

    start_time_density[start_time_arr_unique_vals] = start_time_arr_unique_vals_frequencies

    # Find Best Range for this density

    start_time_range_densities = [start_time_density[np.array(
        range(xr, xr + vpp_local_config.get('timediv_max_density_range'))) % vpp_local_config.get(
            'total_timediv_count')] for xr in range(vpp_local_config.get('total_timediv_count'))]

    start_time_max_density_range_index = np.argmax(np.array([np.sum(xr) for xr in start_time_range_densities]))

    end_time_density = np.zeros(vpp_local_config.get('total_timediv_count'))

    # Get unique an dfrequencies from end time arr

    end_time_arr_unique_vals, end_time_arr_unique_vals_frequencies = np.unique(np.array(end_time_arr),
                                                                               return_counts=True)

    end_time_density[end_time_arr_unique_vals] = end_time_arr_unique_vals_frequencies

    # Find Best Range for this density

    end_time_range_densities = \
        [end_time_density[np.array(range(xr, xr + vpp_local_config.get('timediv_max_density_range'))) %
                          vpp_local_config.get('total_timediv_count')]
         for xr in range(vpp_local_config.get('total_timediv_count'))]

    end_time_max_density_range_index = np.argmax(np.array([np.sum(xr) for xr in end_time_range_densities]))

    universal_start_timediv_left = start_time_max_density_range_index
    universal_end_timediv_right = \
        (end_time_max_density_range_index +
         vpp_local_config.get('timediv_max_density_range')) % vpp_local_config.get('total_timediv_count')

    if universal_start_timediv_left > universal_end_timediv_right:
        universal_duration = vpp_local_config.get('total_timediv_count') - \
                             (universal_start_timediv_left - universal_end_timediv_right) + \
                             vpp_local_config.get('timediv_max_density_range')
    else:
        universal_duration = universal_end_timediv_right - universal_start_timediv_left - \
                             vpp_local_config.get('timediv_max_density_range')

    # Section 4: Remove all and any sections(based on duration) and then edges outside universal schedule

    df_all_runs = create_all_runs_dataframe(day_arr, duration_arr, start_time_arr, end_time_arr)
    single_schedule_edge_matrix = clean_union_masked.copy()

    # remove any sections not falling into start,end range, and then not falling duration
    df_all_runs.loc[:, 'duration_outlier'] = \
        np.absolute(df_all_runs.duration - universal_duration) > vpp_local_config.get('duration_outlier_limit')

    df_all_runs.loc[:, 'start_time_outlier'] = \
        np.absolute(df_all_runs.start_time - universal_start_timediv_left +
                    (vpp_local_config.get('timediv_max_density_range') // 2)) > \
        vpp_local_config.get('start_time_outlier_limit')

    if universal_start_timediv_left < universal_end_timediv_right:
        df_all_runs.loc[:, 'schedule_outlier'] = ~((df_all_runs.start_time < universal_end_timediv_right) &
                                                   (df_all_runs.start_time >= universal_start_timediv_left))
        df_all_runs.schedule_outlier = \
            df_all_runs.schedule_outlier | ~((df_all_runs.end_time <= universal_end_timediv_right) &
                                             (df_all_runs.end_time > universal_start_timediv_left))

    else:
        df_all_runs.loc[:, 'schedule_outlier'] = ((df_all_runs.start_time >= universal_end_timediv_right) &
                                                  (df_all_runs.start_time < universal_start_timediv_left))
        df_all_runs.schedule_outlier = \
            df_all_runs.schedule_outlier | ((df_all_runs.end_time > universal_end_timediv_right) &
                                            (df_all_runs.end_time <= universal_start_timediv_left))

    # Schedule based Outlier

    df_all_runs_schedule_outliers = df_all_runs[
        ((df_all_runs.day.duplicated(keep=False)) & df_all_runs.schedule_outlier)]

    # Only remove such schedule outliers where atleast one schedule is non-outlier

    df_temp = df_all_runs_schedule_outliers.groupby('day', as_index=False).agg(
        {'schedule_outlier': ['count', 'sum']})

    all_runs_all_schedule_outliers_days = df_temp[
        df_temp['schedule_outlier']['sum'] == df_temp['schedule_outlier']['count']].day.unique()

    df_all_runs_schedule_outliers = df_all_runs_schedule_outliers[
        ~df_all_runs_schedule_outliers.day.isin(all_runs_all_schedule_outliers_days)]

    # Delete Days Range not present in df_all_runs duplicated to single matrix:

    for index, row in df_all_runs_schedule_outliers.iterrows():
        row_duration = (row['end_time'] - row['start_time']) % vpp_local_config.get('total_timediv_count')
        row_time_div_arr = (np.arange(row['start_time'], row['start_time'] + row_duration + 1)) % vpp_local_config.get(
            'total_timediv_count')
        single_schedule_edge_matrix[row['day'], row_time_div_arr] = 0

    df_all_runs_duration_outliers = df_all_runs[
        ((df_all_runs.day.duplicated(keep=False)) & df_all_runs.duration_outlier)]

    # Only remove such duration outliers where atleast one duration is non-outlier

    df_temp = df_all_runs_duration_outliers.groupby('day', as_index=False).agg(
        {'duration_outlier': ['count', 'sum']})
    all_runs_all_duration_outliers_days = df_temp[
        df_temp['duration_outlier']['sum'] == df_temp['duration_outlier']['count']].day.unique()

    df_all_runs_duration_outliers = df_all_runs_duration_outliers[
        ~df_all_runs_duration_outliers.day.isin(all_runs_all_duration_outliers_days)]

    # Delete Days Range not present in df_all_runs duplicated to single matrix:

    for index, row in df_all_runs_duration_outliers.iterrows():
        row_duration = (row['end_time'] - row['start_time']) % vpp_local_config.get('total_timediv_count')
        row_time_div_arr = (np.arange(row['start_time'], row['start_time'] + row_duration + 1)) % vpp_local_config.get(
            'total_timediv_count')
        single_schedule_edge_matrix[row['day'], row_time_div_arr] = 0

    # Section 6: Edge Patching: Patch discontinuous edges

    pos_pse_matrix, neg_pse_matrix = get_patched_edges(single_schedule_edge_matrix, pos_masked_time_div,
                                                       vpp_local_config)

    # Section 7a: Edge Removal:Remove small unwanted edges

    cleaned_schedule_edge_matrix = remove_unwanted_edges(pos_pse_matrix, neg_pse_matrix, vpp_local_config)

    # Section 7b: Find Pairs for Cleaning Signal Consistency, Amplitude ratio, AUC for patched Matrix

    clean_pos_edges, clean_neg_edges = label_edges(cleaned_schedule_edge_matrix, pp_config)
    clean_pairs, _, _ = \
        get_eligible_pairs_from_edges_for_vpp(data_bl_removed, cleaned_schedule_edge_matrix, clean_pos_edges,
                                              clean_neg_edges, vpp_local_config.get('total_timediv_count'),
                                              num_samples_per_hr, pp_config,
                                              default_min_amp_ratio=vpp_local_config.get('block_amp_ratio_epsilon'))

    cleaned_schedule_pair_matrix = np.zeros_like(data_clean_edges)

    for pair in clean_pairs:
        cleaned_schedule_pair_matrix[pair[2]:pair[3], clean_pos_edges[pair[0]][2]] = clean_pos_edges[pair[0]][3]
        cleaned_schedule_pair_matrix[pair[2]:pair[3], clean_neg_edges[pair[1]][2]] = -(clean_neg_edges[pair[1]][3])

    # Section 8: Find Structures for all schedules, and then global structure type, and regrow required edges

    df_section_structures = get_all_section_structures(cleaned_schedule_pair_matrix, vpp_local_config)
    vpp_type = get_global_structure(df_section_structures)

    # write Global Structure

    edge_deprived_sections = df_section_structures[(df_section_structures.pos_edge_count < (vpp_type // 10)) |
                                                   (df_section_structures.neg_edge_count < (vpp_type % 10))]

    edge_refill_edge_matrix = cleaned_schedule_pair_matrix.copy()

    is_pos_timediv_masked = np.zeros(vpp_local_config.get('total_timediv_count'))
    is_pos_timediv_masked[pos_masked_time_div] = 1
    is_neg_timediv_masked = np.zeros(vpp_local_config.get('total_timediv_count'))
    is_neg_timediv_masked[neg_masked_time_div] = 1

    for index, row in edge_deprived_sections.iterrows():
        if row['pos_edge_count'] < vpp_type // 10:
            # Get Sections from timediv_masking, clean union, smooth nms and refill Matrix

            section_start_timediv, section_duration, day = row['start_time'], row['duration'], row['day']

            # Get timedivs for the section to handle rolling

            section_timedivs = np.array(range(section_start_timediv, section_start_timediv + section_duration + 1)) % (
                vpp_local_config.get('total_timediv_count'))

            pos_section_timediv_mask = is_pos_timediv_masked[section_timedivs]
            current_section = edge_refill_edge_matrix[day, section_timedivs]

            clean_union_section = data_clean_edges[day, section_timedivs]
            smooth_nms_section = data_nms[day, section_timedivs]

            # Fill New Edges from Clean Union IF timediv is masked and current section is unfilled

            clean_union_section_fill_indexes = \
                np.where((pos_section_timediv_mask == 1) & (current_section == 0) & (clean_union_section > 0))[0]
            current_section[clean_union_section_fill_indexes] = clean_union_section[
                clean_union_section_fill_indexes]

            # Fill New Edges from Smooth NMS IF timediv is masked and current section is unfilled

            smooth_nms_section_fill_indexes = \
                np.where((pos_section_timediv_mask == 1) & (current_section == 0) & (smooth_nms_section > 0))[0]
            current_section[smooth_nms_section_fill_indexes] = smooth_nms_section[smooth_nms_section_fill_indexes]
            edge_refill_edge_matrix[day, section_timedivs] = current_section

        if row['neg_edge_count'] < vpp_type % 10:
            # Get Sections from timediv_masking, clean union, smooth nms and refill Matrix

            section_start_timediv, section_duration, day = row['start_time'], row['duration'], row['day']

            # Get timedivs for the section to handle rolling

            section_timedivs = np.array(range(section_start_timediv, section_start_timediv + section_duration + 1)) % (
                vpp_local_config.get('total_timediv_count'))

            neg_section_timediv_mask = is_neg_timediv_masked[section_timedivs]
            current_section = edge_refill_edge_matrix[day, section_timedivs]
            clean_union_section = data_clean_edges[day, section_timedivs]
            smooth_nms_section = data_clean_edges[day, section_timedivs]

            # Fill New Edges from Clean Union IF timediv is masked and current section is unfilled

            clean_union_section_fill_indexes = \
                np.where((neg_section_timediv_mask == 1) & (current_section == 0) & (clean_union_section < 0))[0]
            current_section[clean_union_section_fill_indexes] = clean_union_section[
                clean_union_section_fill_indexes]

            # Fill New Edges from Smooth NMS IF timediv is masked and current section is unfilled

            smooth_nms_section_fill_indexes = \
                np.where((neg_section_timediv_mask == 1) & (current_section == 0) & (smooth_nms_section < 0))[0]
            current_section[smooth_nms_section_fill_indexes] = smooth_nms_section[smooth_nms_section_fill_indexes]
            edge_refill_edge_matrix[day, section_timedivs] = current_section

    # Section 9: Try to find all structures for all positive edges

    clean_vpairs = get_all_consistent_vpairs(vpp_local_config, data_bl_removed, edge_refill_edge_matrix, vpp_type,
                                             pp_config)

    num_cols = 6
    structures = np.empty(shape=(0, 10))
    low_amp_pairs, high_amp_pairs = np.empty(shape=(1, num_cols)), np.empty(shape=(1, num_cols))
    low_amp_pairs, high_amp_pairs, structures = get_variable_run_amp_wise_pairs(clean_vpairs, low_amp_pairs,
                                                                                high_amp_pairs, structures)

    # Section 10: Fill clean pairs in Matrix

    edge_refill_pair_matrix = fill_clean_pairs(clean_vpairs, edge_refill_edge_matrix)

    # Section 11: Try to find vpairs for Whitespaces based on all edges in Smooth and Clean Union(No Filters)

    whitespace_edge_matrix = np.zeros_like(data_clean_edges)
    nonzero_count_edge_refill_pair_matrix = np.count_nonzero(edge_refill_pair_matrix, axis=1)
    whitespace_days = np.where(nonzero_count_edge_refill_pair_matrix == 0)[0]

    if len(whitespace_days) > 1:
        whitespace_input_dict = dict()
        whitespace_input_dict['whitespace_days'] = whitespace_days
        whitespace_input_dict['data_clean_edges'] = data_clean_edges
        whitespace_input_dict['data_nms'] = data_nms
        whitespace_input_dict['data_bl_removed'] = data_bl_removed
        whitespace_input_dict['stype_ppnn'] = stype_ppnn

        edge_refill_pair_matrix, low_amp_pairs, high_amp_pairs, structures = fill_whitespaces(
            whitespace_input_dict, whitespace_edge_matrix, edge_refill_pair_matrix, low_amp_pairs, high_amp_pairs,
            structures, pp_config, vpp_local_config)

    else:

        low_amp_pairs = np.array(low_amp_pairs[1:], dtype=int)
        high_amp_pairs = np.array(high_amp_pairs[1:], dtype=int)

    # Compute consumption matrix

    low_consumption_matrix, low_cons_threshold = get_consumption_matrix(low_amp_pairs, data_bl_removed, data_bl_removed,
                                                                        num_samples_per_hr, pp_config)

    high_consumption_matrix, high_cons_threshold = get_consumption_matrix(high_amp_pairs, data_bl_removed,
                                                                          data_bl_removed,
                                                                          num_samples_per_hr, pp_config)

    consumption_matrix = np.where(low_consumption_matrix < high_consumption_matrix, high_consumption_matrix,
                                  low_consumption_matrix)

    # Modify structures to add consumption

    c1 = structures[:, 6]
    c2 = structures[:, 7]
    c3 = structures[:, 8]
    c4 = structures[:, 9]

    t1_idx = np.logical_and(c2 == 0, c3 == 0)
    t2_idx = np.logical_and(c2 == 0, c3 != 0)
    t3_idx = np.logical_and(c2 != 0, c3 == 0)
    t4_idx = np.logical_and(c2 != 0, c3 != 0)

    cons_cols = np.zeros(shape=(structures.shape[0], 3))

    high_cons_threshold *= num_samples_per_hr
    low_cons_threshold *= num_samples_per_hr

    # Fill values case by case

    # Case 1 both in the middle are zero

    cons_cols[t1_idx, 1] = np.fmax((c1[t1_idx] + c4[t1_idx]) / 2, high_cons_threshold)

    # Case 2 Inner positive 0 inner negative present

    cons_cols[t2_idx, 1] = np.fmax((c1[t2_idx] + c3[t2_idx] + c4[t2_idx]) / 2, high_cons_threshold)
    cons_cols[t2_idx, 2] = np.fmax((c3[t2_idx] + c4[t2_idx]) / 2, low_cons_threshold)

    # Case 3 Inner positive present inner negative 0

    cons_cols[t3_idx, 0] = np.fmax((c1[t3_idx] + c2[t3_idx]) / 2, low_cons_threshold)
    cons_cols[t3_idx, 1] = np.fmax((c1[t3_idx] + c2[t3_idx] + c4[t3_idx]) / 2, high_cons_threshold)

    # Case 4 Inner positive present inner negative present

    cons_cols[t4_idx, 0] = np.fmax((c1[t4_idx] + c2[t4_idx]) / 2, low_cons_threshold)
    cons_cols[t4_idx, 1] = np.fmax((c1[t4_idx] + c2[t4_idx] + c3[t4_idx] + c4[t4_idx]) / 2, high_cons_threshold)
    cons_cols[t4_idx, 2] = np.fmax((c3[t4_idx] + c4[t4_idx]) / 2, low_cons_threshold)

    structures = np.c_[structures, cons_cols]

    return edge_refill_pair_matrix, consumption_matrix, high_amp_pairs, structures


def fill_clean_pairs(clean_vpairs, edge_refill_edge_matrix):
    """Form pair matrix from clean pairs"""

    edge_refill_pair_matrix = np.zeros_like(edge_refill_edge_matrix)

    key_start_day, key_end_day, key_start_timediv, key_end_timediv = 0, 1, 2, 3
    edge_timediv = 2

    for color in clean_vpairs:
        section_vpairs = clean_vpairs[color]
        for key in section_vpairs:
            vpair_id = key.split(",")
            vpair_start_day, vpair_end_day = int(vpair_id[key_start_day]), int(vpair_id[key_end_day])
            for pos_edge in section_vpairs[key]['pos_edges']:
                edge_refill_pair_matrix[vpair_start_day:vpair_end_day + 1, pos_edge[edge_timediv]] = \
                    edge_refill_edge_matrix[vpair_start_day:vpair_end_day + 1, pos_edge[edge_timediv]]
            for neg_edge in section_vpairs[key]['neg_edges']:
                edge_refill_pair_matrix[vpair_start_day:vpair_end_day + 1, neg_edge[edge_timediv]] = \
                    edge_refill_edge_matrix[vpair_start_day:vpair_end_day + 1, neg_edge[edge_timediv]]

    return edge_refill_pair_matrix


def fill_whitespaces(whitespace_input_dict, whitespace_edge_matrix, edge_refill_pair_matrix, low_amp_pairs,
                     high_amp_pairs, structures, pp_config, vpp_local_config):
    """Fill whitespaces in the pair matrix formed"""

    key_start_day, key_end_day, key_start_timediv, key_end_timediv = 0, 1, 2, 3
    edge_timediv = 2

    whitespace_days = whitespace_input_dict['whitespace_days']
    data_clean_edges = whitespace_input_dict['data_clean_edges']
    data_nms = whitespace_input_dict['data_nms']
    data_bl_removed = whitespace_input_dict['data_bl_removed']
    stype_ppnn = whitespace_input_dict['stype_ppnn']

    for day in whitespace_days:
        whitespace_edge_matrix[day, :] = data_clean_edges[day, :]
        whitespace_edge_matrix[day, :] = np.where(whitespace_edge_matrix[day, :] > 0,
                                                  whitespace_edge_matrix[day, :], data_nms[day, :])

    whitespace_filled_vpairs = get_all_consistent_vpairs(vpp_local_config, data_bl_removed, whitespace_edge_matrix,
                                                         stype_ppnn, pp_config)

    low_amp_pairs, high_amp_pairs, structures = \
        get_variable_run_amp_wise_pairs(whitespace_filled_vpairs, low_amp_pairs, high_amp_pairs, structures)

    # Find Type for each Section
    # Section 12: Fill whitespace pairs in Clean pair matrix

    whitespace_pair_matrix = copy.deepcopy(edge_refill_pair_matrix)
    for color in whitespace_filled_vpairs:
        section_vpairs = whitespace_filled_vpairs[color]
        for key in section_vpairs:
            vpair_id = key.split(",")
            vpair_start_day, vpair_end_day = int(vpair_id[key_start_day]), int(vpair_id[key_end_day])
            for pos_edge in section_vpairs[key]['pos_edges']:
                whitespace_pair_matrix[vpair_start_day:vpair_end_day + 1, pos_edge[edge_timediv]] = \
                    whitespace_edge_matrix[vpair_start_day:vpair_end_day + 1, pos_edge[edge_timediv]]
            for neg_edge in section_vpairs[key]['neg_edges']:
                whitespace_pair_matrix[vpair_start_day:vpair_end_day + 1, neg_edge[edge_timediv]] = \
                    whitespace_edge_matrix[vpair_start_day:vpair_end_day + 1, neg_edge[edge_timediv]]

    edge_refill_pair_matrix = whitespace_pair_matrix
    low_amp_pairs = np.array(low_amp_pairs[1:], dtype=int)
    high_amp_pairs = np.array(high_amp_pairs[1:], dtype=int)

    return edge_refill_pair_matrix, low_amp_pairs, high_amp_pairs, structures
