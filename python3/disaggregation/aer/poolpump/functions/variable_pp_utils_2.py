"""
Author - Arpan Agrawal
Date - 09/04/2019
Function to find consistent vpairs for Variable PP
"""

import numpy as np
import pandas as pd
from scipy import stats


def get_global_structure(df_section_structures):
    """Utility to get global structure"""

    df_sec_struct_vals = df_section_structures.values

    global_pos_edge_count = min(stats.mode(df_sec_struct_vals[:, 4])[0][0], 2)
    global_neg_edge_count = min(stats.mode(df_sec_struct_vals[:, 5])[0][0], 2)

    return global_pos_edge_count * 10 + global_neg_edge_count


def create_all_runs_dataframe(day_arr, duration_arr, start_time_arr, end_time_arr):
    """Utility to get dataframe for usage further"""

    df = pd.DataFrame({
        'day': day_arr,
        'duration': duration_arr,
        'start_time': start_time_arr,
        'end_time': end_time_arr
    })

    return df


def check_signal_consistency(vpp_local_config, pos_edge, neg_edge, raw_data, amp=None):
    """Utility to check the consistency of signal"""

    total_time_divisions = vpp_local_config.get('total_timediv_count')

    # Write enum for addressing edges

    start_day, end_day, timediv, amplitude, sign = 0, 1, 2, 3, 4

    match_start_day = max(pos_edge[start_day], neg_edge[start_day])
    match_end_day = min(pos_edge[end_day], neg_edge[end_day])

    duration = (neg_edge[timediv] - pos_edge[timediv]) % total_time_divisions

    timediv_range = np.arange(pos_edge[timediv], pos_edge[timediv] + duration) % total_time_divisions

    data_reqd = raw_data[match_start_day: match_end_day, timediv_range]

    if amp is None:
        amp = min(pos_edge[amplitude], neg_edge[amplitude])

    amp_per_hr = amp / vpp_local_config.get('samples_per_hour')

    if vpp_local_config.get('samples_per_hour') == 1:

        min_amp = vpp_local_config.get('vpair_relaxed_amplitude_margin') * amp_per_hr
        data_check = data_reqd

    elif vpp_local_config.get('samples_per_hour') == 2:

        min_amp = vpp_local_config.get('vpair_amplitude_margin') * amp_per_hr
        data_check = data_reqd[:, 1:data_reqd.shape[1] - 1]

    else:

        min_amp = vpp_local_config.get('vpair_amplitude_margin') * amp_per_hr
        data_check = data_reqd[:, 1:data_reqd.shape[1] - 2]

    check_rows, check_cols = data_check.shape

    # Compute final values

    data_check[data_check < min_amp] = 0
    data_check = np.sign(data_check)
    duration_fraction_arr = np.sum(data_check, axis=1)

    # Compute duration fraction array

    if vpp_local_config.get('samples_per_hour') == 4:
        min_border_amp = vpp_local_config.get('vpair_relaxed_amplitude_margin') * amp_per_hr
        second_last_time_div_data = data_reqd[:, -2]

        idx_1 = second_last_time_div_data < min_border_amp
        duration_fraction_arr[idx_1] = duration_fraction_arr[idx_1] / check_cols

        idx_2 = np.logical_not(idx_1)
        duration_fraction_arr[idx_2] = (duration_fraction_arr[idx_2] + 1) / (check_cols + 1)
    else:
        duration_fraction_arr = duration_fraction_arr / check_cols

    # Compute aggregate values

    idx_dur_valid = duration_fraction_arr >= vpp_local_config.get('vpair_min_duration_fraction')
    duration_fraction_sum = np.sum(duration_fraction_arr[idx_dur_valid])
    days_carried = np.sum(idx_dur_valid)
    days_in_match = np.sum(duration_fraction_arr == 1)

    if days_carried == 0:
        day_fraction = 0
    else:
        day_fraction = duration_fraction_sum / days_carried

    match_fraction = days_in_match / (match_end_day - match_start_day + 1)

    return day_fraction, match_fraction


def get_variable_run_amp_wise_pairs(vpairs, low_amp_pairs, high_amp_pairs, structures):
    """Utility to get variable run amplitude by pair"""

    edge_col_idx = {
        'start_day': 0,
        'end_day': 1,
        'time_div': 2,
        'amp': 3,
        'sign': 4
    }

    for key in vpairs.keys():
        for sub_key in vpairs[key].keys():

            common_start_day = vpairs[key][sub_key]['common_start_day']
            # added 1 for index slicing purposes
            common_end_day = vpairs[key][sub_key]['common_end_day'] + 1

            if vpairs[key][sub_key]['type'] == 22:
                outer_pos_edge = vpairs[key][sub_key]['pos_edges'][0]
                inner_pos_edge = vpairs[key][sub_key]['pos_edges'][1]
                inner_neg_edge = vpairs[key][sub_key]['neg_edges'][0]
                outer_neg_edge = vpairs[key][sub_key]['neg_edges'][1]

                outer_pos_time_div = outer_pos_edge[edge_col_idx.get('time_div')]
                outer_pos_amp = outer_pos_edge[edge_col_idx.get('amp')]
                inner_pos_time_div = inner_pos_edge[edge_col_idx.get('time_div')]
                inner_pos_amp = inner_pos_edge[edge_col_idx.get('amp')]
                outer_neg_time_div = outer_neg_edge[edge_col_idx.get('time_div')]
                outer_neg_amp = outer_neg_edge[edge_col_idx.get('amp')]
                inner_neg_time_div = inner_neg_edge[edge_col_idx.get('time_div')]
                inner_neg_amp = inner_neg_edge[edge_col_idx.get('amp')]

                low_pos_amp_pair = np.array(
                    [outer_pos_time_div, inner_pos_time_div, common_start_day, common_end_day, outer_pos_amp,
                     outer_neg_amp])
                low_neg_amp_pair = np.array(
                    [inner_neg_time_div, outer_neg_time_div, common_start_day, common_end_day, outer_pos_amp,
                     outer_neg_amp])
                high_amp_pair = np.array([inner_pos_time_div, inner_neg_time_div, common_start_day, common_end_day,
                                          (outer_pos_amp + inner_pos_amp), (outer_neg_amp + inner_neg_amp)])
                low_amp_pairs = np.vstack((low_amp_pairs, low_pos_amp_pair, low_neg_amp_pair))
                high_amp_pairs = np.vstack((high_amp_pairs, high_amp_pair))

                structures = np.vstack((structures, np.array(
                    [common_start_day, common_end_day, outer_pos_time_div, inner_pos_time_div, inner_neg_time_div,
                     outer_neg_time_div, outer_pos_amp, inner_pos_amp, inner_neg_amp, outer_neg_amp])))

            elif vpairs[key][sub_key]['type'] == 21:
                outer_pos_edge = vpairs[key][sub_key]['pos_edges'][0]
                inner_pos_edge = vpairs[key][sub_key]['pos_edges'][1]
                outer_neg_edge = vpairs[key][sub_key]['neg_edges'][0]

                outer_pos_time_div = outer_pos_edge[edge_col_idx.get('time_div')]
                outer_pos_amp = outer_pos_edge[edge_col_idx.get('amp')]
                inner_pos_time_div = inner_pos_edge[edge_col_idx.get('time_div')]
                inner_pos_amp = inner_pos_edge[edge_col_idx.get('amp')]
                outer_neg_time_div = outer_neg_edge[edge_col_idx.get('time_div')]
                outer_neg_amp = outer_neg_edge[edge_col_idx.get('amp')]

                low_pos_amp_pair = np.array(
                    [outer_pos_time_div, inner_pos_time_div, common_start_day, common_end_day, outer_pos_amp,
                     outer_pos_amp])
                high_amp_pair = np.array([inner_pos_time_div, outer_neg_time_div, common_start_day, common_end_day,
                                          (outer_pos_amp + inner_pos_amp), outer_neg_amp])
                low_amp_pairs = np.vstack((low_amp_pairs, low_pos_amp_pair))
                high_amp_pairs = np.vstack((high_amp_pairs, high_amp_pair))

                structures = np.vstack((structures, np.array(
                    [common_start_day, common_end_day, outer_pos_time_div, inner_pos_time_div, -1,
                     outer_neg_time_div, outer_pos_amp, inner_pos_amp, 0, outer_neg_amp])))

            elif vpairs[key][sub_key]['type'] == 12:
                outer_pos_edge = vpairs[key][sub_key]['pos_edges'][0]
                inner_neg_edge = vpairs[key][sub_key]['neg_edges'][0]
                outer_neg_edge = vpairs[key][sub_key]['neg_edges'][1]

                outer_pos_time_div = outer_pos_edge[edge_col_idx.get('time_div')]
                outer_pos_amp = outer_pos_edge[edge_col_idx.get('amp')]
                outer_neg_time_div = outer_neg_edge[edge_col_idx.get('time_div')]
                outer_neg_amp = outer_neg_edge[edge_col_idx.get('amp')]
                inner_neg_time_div = inner_neg_edge[edge_col_idx.get('time_div')]
                inner_neg_amp = inner_neg_edge[edge_col_idx.get('amp')]

                low_neg_amp_pair = np.array(
                    [inner_neg_time_div, outer_neg_time_div, common_start_day, common_end_day, outer_neg_amp,
                     outer_neg_amp])
                high_amp_pair = np.array([outer_pos_time_div, inner_neg_time_div, common_start_day, common_end_day,
                                          outer_pos_amp, (outer_neg_amp + inner_neg_amp)])
                low_amp_pairs = np.vstack((low_amp_pairs, low_neg_amp_pair))
                high_amp_pairs = np.vstack((high_amp_pairs, high_amp_pair))

                structures = np.vstack((structures, np.array(
                    [common_start_day, common_end_day, outer_pos_time_div, -1, inner_neg_time_div,
                     outer_neg_time_div, outer_pos_amp, 0, inner_neg_amp, outer_neg_amp])))

            elif vpairs[key][sub_key]['type'] == 11:
                outer_pos_edge = vpairs[key][sub_key]['pos_edges'][0]
                outer_neg_edge = vpairs[key][sub_key]['neg_edges'][0]

                outer_pos_time_div = outer_pos_edge[edge_col_idx.get('time_div')]
                outer_pos_amp = outer_pos_edge[edge_col_idx.get('amp')]
                outer_neg_time_div = outer_neg_edge[edge_col_idx.get('time_div')]
                outer_neg_amp = outer_neg_edge[edge_col_idx.get('amp')]

                high_amp_pair = np.array([outer_pos_time_div, outer_neg_time_div, common_start_day, common_end_day,
                                          outer_pos_amp, outer_neg_amp])
                high_amp_pairs = np.vstack((high_amp_pairs, high_amp_pair))

                structures = np.vstack((structures, np.array(
                    [common_start_day, common_end_day, outer_pos_time_div, -1, -1,
                     outer_neg_time_div, outer_pos_amp, 0, 0, outer_neg_amp])))

    return low_amp_pairs, high_amp_pairs, structures
