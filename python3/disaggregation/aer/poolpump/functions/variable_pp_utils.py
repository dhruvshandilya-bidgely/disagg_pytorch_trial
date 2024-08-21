"""
Author - Mayank Sharan
Date - 31/1/19
Utility functions to help compute the estimate for variable pool pump
"""

# Import python packages

import copy
import numpy as np
import pandas as pd
from itertools import groupby as iter_groupby

# Import functions from within the project

from python3.disaggregation.aer.poolpump.functions.cleaning_utils import find_edges

from python3.disaggregation.aer.poolpump.functions.pp_model_utils import label_edges
from python3.disaggregation.aer.poolpump.functions.pp_model_utils import get_consistency_val
from python3.disaggregation.aer.poolpump.functions.vpairs_utils import find_consistent_vpairs


def roll_day_arr_pos_first(day_nonzero_timediv, day_nonzero_parity):
    """Utility to roll over 1d arrays"""

    # Roll over for +ve in first and -ve in last
    if day_nonzero_parity[0] < 0:

        roll_idx = np.argmax(day_nonzero_parity)
        day_nonzero_parity = np.roll(day_nonzero_parity, -roll_idx)
        day_nonzero_timediv = np.roll(day_nonzero_timediv, -roll_idx)

    elif day_nonzero_parity[-1] > 0:

        roll_idx = np.argmin(day_nonzero_parity[::-1])
        day_nonzero_parity = np.roll(day_nonzero_parity, roll_idx)
        day_nonzero_timediv = np.roll(day_nonzero_timediv, roll_idx)

    return day_nonzero_timediv, day_nonzero_parity


def get_masked_time_divisions_vpp(vpp_local_config, clean_union):
    """Utility to get masked time divisions"""

    # Get positive masked time divisions

    pos_clean_union = copy.deepcopy(clean_union)
    pos_clean_union[pos_clean_union < 0] = 0
    pos_clean_union[pos_clean_union > 0] = 1

    pos_filled_days_count = np.sum(pos_clean_union, axis=0)
    pos_filled_edge_length = np.zeros_like(pos_filled_days_count)

    for time_div in range(vpp_local_config.get('total_timediv_count')):

        start_arr, end_arr = find_edges(pos_clean_union[:, time_div])

        if len(start_arr) > 0:
            pos_filled_edge_length[time_div] = np.max(end_arr - start_arr)
        else:
            pos_filled_edge_length[time_div] = 0

    pos_masked_time_div = np.where((pos_filled_days_count >= vpp_local_config.get('td_masking_min_filled_days'))
                                   | (pos_filled_edge_length >= vpp_local_config.get('td_masking_min_edge_length')))

    # Get negative masked time divisions

    neg_clean_union = copy.deepcopy(clean_union)
    neg_clean_union[neg_clean_union > 0] = 0
    neg_clean_union[neg_clean_union < 0] = 1

    neg_filled_days_count = np.sum(neg_clean_union, axis=0)
    neg_filled_edge_length = np.zeros_like(neg_filled_days_count)

    for time_div in range(vpp_local_config.get('total_timediv_count')):

        start_arr, end_arr = find_edges(neg_clean_union[:, time_div])

        if len(start_arr) > 0:
            neg_filled_edge_length[time_div] = np.max(end_arr - start_arr)
        else:
            neg_filled_edge_length[time_div] = 0

    neg_masked_time_div = np.where((neg_filled_days_count >= vpp_local_config.get('td_masking_min_filled_days'))
                                   | (neg_filled_edge_length >= vpp_local_config.get('td_masking_min_edge_length')))

    return np.array(pos_masked_time_div), np.array(neg_masked_time_div)


def get_universal_details(clean_union_masked, vpp_local_config):
    """Utility to get details universal like start end duration and corresponding days"""

    day_arr, duration_arr, start_time_arr, end_time_arr = [], [], [], []

    edge_parity_matrix = np.sign(clean_union_masked)
    ep_sum_day = np.abs(np.sum(edge_parity_matrix, axis=1))
    ep_count_day = np.count_nonzero(edge_parity_matrix, axis=1)

    # We get duration start and end time of each section for each day

    for day in range(vpp_local_config.get('total_days_count')):

        if (ep_count_day[day] < 2) or (ep_sum_day[day] == ep_count_day[day]):
            continue

        # find indices and values for non zero indices

        edge_parity_dayarr = edge_parity_matrix[day, :]
        ep_dayarr_nonzero_timediv = np.where(~(edge_parity_dayarr == 0))[0]
        ep_dayarr_nonzero_values = edge_parity_dayarr[ep_dayarr_nonzero_timediv]

        # remove days if have one positive or negative edges or zero edges

        ep_dayarr_nonzero_timediv, ep_dayarr_nonzero_values = roll_day_arr_pos_first(ep_dayarr_nonzero_timediv,
                                                                                     ep_dayarr_nonzero_values)

        n_to_p_transition_idx = np.where((ep_dayarr_nonzero_values[:-1] - ep_dayarr_nonzero_values[1:]) == -2)[0]

        # padding start and end index to transition index

        n_to_p_transition_idx = [0] + list(n_to_p_transition_idx) + [len(ep_dayarr_nonzero_timediv) - 1]

        # loop over sections to find duration, start time and end time for each section

        for i in range(len(n_to_p_transition_idx) - 1):

            section_start_idx, section_end_idx = n_to_p_transition_idx[i], n_to_p_transition_idx[i + 1]

            section_start_timediv, section_end_timediv = \
                ep_dayarr_nonzero_timediv[section_start_idx], ep_dayarr_nonzero_timediv[section_end_idx]

            if section_end_timediv < section_start_timediv:
                duration_arr.append(vpp_local_config.get('total_timediv_count') +
                                    section_end_timediv - section_start_timediv)
            else:
                duration_arr.append(section_end_timediv - section_start_timediv)

            start_time_arr.append(section_start_timediv)
            end_time_arr.append(section_end_timediv)
            day_arr.append(day)

    return day_arr, duration_arr, start_time_arr, end_time_arr


def get_patched_edges(single_schedule_edge_matrix, pos_masked_time_div, vpp_local_config):
    """Utility to get patched edges"""

    pos_pse_matrix = single_schedule_edge_matrix.copy()
    pos_pse_matrix[pos_pse_matrix < 0] = 0

    neg_pse_matrix = single_schedule_edge_matrix.copy()
    neg_pse_matrix[neg_pse_matrix > 0] = 0

    for timediv in range(vpp_local_config.get('total_timediv_count')):

        # Do Edge Patching for Positive Matrix

        edge_start_arr, edge_end_arr = find_edges(pos_pse_matrix[:, timediv])

        # Modify Edge Start and End Array to cater patches

        patch_start_arr = np.append(np.array([0]), edge_end_arr)
        patch_end_arr = np.append(edge_start_arr, vpp_local_config.get('total_days_count'))

        patch_length_arr = patch_end_arr - patch_start_arr
        min_patch_length = vpp_local_config.get('max_edge_patch_length')
        patchable_indexes = np.where((patch_length_arr <= min_patch_length) & (patch_length_arr > 0))[0]

        for day_idx in patchable_indexes:

            patch_start, patch_end, patch_length = patch_start_arr[day_idx], patch_end_arr[day_idx], \
                                                   patch_length_arr[day_idx]

            prev_edge_length, next_edge_length = (patch_start_arr[day_idx] - patch_end_arr[max(day_idx - 1, 0)]), (
                patch_start_arr[min(day_idx + 1, len(patch_start_arr) - 1)] - patch_end_arr[day_idx])

            patch_bool, prev_edge_amplitude, next_edge_amplitude = check_patching_conditions(
                patch_start_arr, patch_end_arr, day_idx, pos_pse_matrix, timediv, pos_masked_time_div, patch_length,
                prev_edge_length, next_edge_length, vpp_local_config)

            if not patch_bool:
                continue
            # Both Conditions passed, Fill Patch

            pos_pse_matrix[patch_start:patch_end, timediv] = np.mean([prev_edge_amplitude, next_edge_amplitude])

            # Redo Patch arrays to cover filled patch

            patch_start_arr[day_idx] = patch_start_arr[max(day_idx - 1, 0)]
            patch_end_arr[day_idx] = patch_end_arr[max(day_idx - 1, 0)]

        # Do Edge Patching for Negative Matrix

        edge_start_arr, edge_end_arr = find_edges(np.abs(neg_pse_matrix[:, timediv]))

        # Modify Edge Start and End Array to cater patches

        patch_start_arr = np.append(np.array([0]), edge_end_arr)
        patch_end_arr = np.append(edge_start_arr, vpp_local_config.get('total_days_count'))

        patch_length_arr = patch_end_arr - patch_start_arr
        min_patch_length = vpp_local_config.get('max_edge_patch_length')
        patchable_indexes = np.where((patch_length_arr <= min_patch_length) & (patch_length_arr > 0))[0]

        for day_idx in patchable_indexes:
            patch_start, patch_end, patch_length = patch_start_arr[day_idx], patch_end_arr[day_idx], \
                                                   patch_length_arr[day_idx]
            # check If we should patch or not

            prev_edge_length, next_edge_length = (patch_start_arr[day_idx] - patch_end_arr[max(day_idx - 1, 0)]), (
                patch_start_arr[min(day_idx + 1, len(patch_start_arr) - 1)] - patch_end_arr[day_idx])

            patch_bool, prev_edge_amplitude, next_edge_amplitude = check_patching_conditions(
                patch_start_arr, patch_end_arr, day_idx, neg_pse_matrix, timediv, pos_masked_time_div, patch_length,
                prev_edge_length, next_edge_length, vpp_local_config)

            if not patch_bool:
                continue
            # Both Conditions passed, Fill Patch

            neg_pse_matrix[patch_start:patch_end, timediv] = np.mean([prev_edge_amplitude, next_edge_amplitude])

            # Redo Patch arrays to cover filled patch

            patch_start_arr[day_idx] = patch_start_arr[max(day_idx - 1, 0)]
            patch_end_arr[day_idx] = patch_end_arr[max(day_idx - 1, 0)]

    return pos_pse_matrix, neg_pse_matrix


def check_patching_conditions(patch_start_arr, patch_end_arr, day_idx, pse_matrix, timediv, masked_time_div,
                              patch_length, prev_edge_length, next_edge_length, vpp_local_config):
    """Checks patching conditions and returns amplitudes for patching"""

    patch_bool = True

    # Condition 1: Edges Around should be more than 30 days atleast in one direction
    min_length = 3
    if timediv in masked_time_div:
        min_length = min(5 * patch_length, vpp_local_config.get('strong_edge_length'))

    if (prev_edge_length < min_length) and (next_edge_length < min_length):
        patch_bool = False

    # Condition 2: Amplitude for neighbouring edges should be in ballpark
    prev_edge_amplitude = 0
    next_edge_amplitude = 0
    if patch_end_arr[max(day_idx - 1, 0)] <= patch_start_arr[day_idx]:
        prev_edge_amplitude = np.mean(pse_matrix[patch_end_arr[max(day_idx - 1, 0)]:patch_start_arr[day_idx], timediv])

    if patch_end_arr[day_idx] < patch_start_arr[min(day_idx + 1, len(patch_end_arr) - 1)]:
        next_edge_amplitude = np.mean(
            pse_matrix[patch_end_arr[day_idx]:patch_start_arr[min(day_idx + 1, len(patch_end_arr) - 1)], timediv])

    if (prev_edge_amplitude == 0) or (next_edge_amplitude == 0):
        patch_bool = False

    prev_edge_amplitude_abs, next_edge_amplitude_abs = np.absolute(prev_edge_amplitude), np.absolute(
        next_edge_amplitude)

    if (min(prev_edge_amplitude_abs, next_edge_amplitude_abs) / max(prev_edge_amplitude_abs,
                                                                    next_edge_amplitude_abs) < 0.8):
        patch_bool = False

    return patch_bool, prev_edge_amplitude, next_edge_amplitude


def remove_unwanted_edges(pos_pse_matrix, neg_pse_matrix, vpp_local_config):
    """Utility to remove unwanted edges"""
    for timediv in range(vpp_local_config.get('total_timediv_count')):

        # Remove Small Positive Edges

        edge_start_arr, edge_end_arr = find_edges(pos_pse_matrix[:, timediv])
        edge_length_arr = edge_end_arr - edge_start_arr

        min_edge_length = vpp_local_config.get('max_edge_patch_length')
        remove_indexes = np.where(edge_length_arr <= min_edge_length)[0]

        for day_idx in remove_indexes:
            pos_pse_matrix[edge_start_arr[day_idx]:edge_end_arr[day_idx], timediv] = 0

        # Remove Small Negative Edges

        edge_start_arr, edge_end_arr = find_edges(np.abs(neg_pse_matrix[:, timediv]))
        edge_length_arr = edge_end_arr - edge_start_arr

        min_edge_length = vpp_local_config.get('max_edge_patch_length')
        remove_indexes = np.where(edge_length_arr <= min_edge_length)[0]

        for day_idx in remove_indexes:
            neg_pse_matrix[edge_start_arr[day_idx]:edge_end_arr[day_idx], timediv] = 0

    cleaned_schedule_edge_matrix = pos_pse_matrix + neg_pse_matrix

    return cleaned_schedule_edge_matrix


def signal_consistency_bool(day_f, match_f, empty_flag, amp_ratios, neg_idx, minimum_day_signal_fraction,
                            minimum_match_signal_ratio, min_amp_ratio):
    """
    Written to reduce complexity
    """
    if empty_flag == 0 and (day_f < minimum_day_signal_fraction or match_f < minimum_match_signal_ratio):
        return False

    if empty_flag == 0 and amp_ratios[neg_idx] < min_amp_ratio:
        return False

    return True


def get_eligible_pairs_from_edges_for_vpp(data, edge_matrix, pos_edges, neg_edges, num_cols,
                                          samples_per_hr, pp_config, default_min_amp_ratio=0.5, minimum_pair_length=5,
                                          empty_flag=0):
    """Utility to get eligible pairs for vpp"""

    # Extract constants from the config

    min_duration = 1
    max_duration = 18
    min_amp_ratio_higher = 0.7
    min_pair_length_lower = 10
    min_pair_length_higher = 30
    min_amp_ratio_lower = 0.3645
    minimum_area_under_curve = 800
    minimum_match_signal_ratio = 0.5
    amp_ratio_reduction_factor = 0.9
    minimum_day_signal_fraction = 0.8

    num_rows = data.shape[0]
    min_amp_ratio = default_min_amp_ratio * (amp_ratio_reduction_factor ** 4)

    # Initialise variables for use in the function

    num_pairs_day = np.zeros(shape=(num_rows,))
    duration_days = np.zeros(shape=(num_rows,))

    matches = []
    duration_list = []
    num_pairs_list = []

    for pos_edge_idx in range(len(pos_edges)):

        pos_timediv_list = np.array(range(pos_edges[pos_edge_idx][2] - 1, pos_edges[pos_edge_idx][2] + 2)) % num_cols

        pos_timediv_derived = edge_matrix[:, pos_timediv_list]
        pos_timediv_derived[pos_timediv_derived < 0] = 0
        pos_timediv_derived = np.sum(pos_timediv_derived, axis=1)

        pos_start_time_derived, pos_end_time_derived = find_edges(np.absolute(pos_timediv_derived))
        pos_edge_index_derived = np.where((pos_start_time_derived <= pos_edges[pos_edge_idx][0]) &
                                          (pos_end_time_derived >= pos_edges[pos_edge_idx][1]))[0]

        pos_edge_length_derived = pos_end_time_derived[pos_edge_index_derived[0]] - pos_start_time_derived[
            pos_edge_index_derived[0]]

        # Initialize values from the positive edge

        pos_edge_start = pos_edges[pos_edge_idx, 0]
        pos_edge_end = pos_edges[pos_edge_idx, 1]
        pos_edge_col = pos_edges[pos_edge_idx, 2]
        pos_edge_amp = pos_edges[pos_edge_idx, 3]

        # Extract eligible negative edges

        eligible_neg_edges_bool = np.logical_and(neg_edges[:, 0] < pos_edge_end, neg_edges[:, 1] > pos_edge_start)
        eligible_neg_edges_idx = np.where(eligible_neg_edges_bool)[0]
        eligible_neg_edges = neg_edges[eligible_neg_edges_bool, :]

        # Compute first rejection criteria for edges

        min_amp = np.minimum(eligible_neg_edges[:, 3], pos_edge_amp)
        max_amp = np.maximum(eligible_neg_edges[:, 3], pos_edge_amp)
        amp_ratios = np.divide(min_amp, max_amp)

        match_start_days = np.maximum(eligible_neg_edges[:, 0], pos_edge_start)
        match_end_days = np.minimum(eligible_neg_edges[:, 1], pos_edge_end)
        pair_lengths = match_end_days - match_start_days

        # Possible bug here in precedence of and and or

        rej_idx_1 = np.logical_or(np.logical_and(empty_flag == 0, np.logical_and(pair_lengths < min_pair_length_lower,
                                                                                 amp_ratios < min_amp_ratio_higher)),
                                  np.logical_and(pair_lengths < min_pair_length_higher,
                                                 amp_ratios < min_amp_ratio_lower))
        rej_idx_1 = np.logical_not(rej_idx_1)

        # Compute rejection criteria 2

        rej_idx_2 = np.logical_not(np.logical_and(empty_flag == 0, pair_lengths < minimum_pair_length))

        # Compute rejection criteria 3 based on area under curve

        duration_arr = ((eligible_neg_edges[:, 2] - pos_edge_col) % num_cols) / samples_per_hr
        auc_arr = np.multiply(pos_edge_amp + eligible_neg_edges[:, 3], duration_arr / 2)

        rej_idx_3 = np.logical_not(auc_arr < minimum_area_under_curve)

        # Compute rejection criteria 4 based on duration

        rej_idx_4 = np.logical_not(np.logical_or(duration_arr < min_duration, duration_arr > max_duration))

        selected_neg_edge_idx_arr = np.where(np.logical_and(np.logical_and(rej_idx_1, rej_idx_2),
                                                            np.logical_and(rej_idx_3, rej_idx_4)))[0]

        for neg_idx in selected_neg_edge_idx_arr:

            neg_timediv_list = np.array(range(eligible_neg_edges[neg_idx][2] - 1,
                                              eligible_neg_edges[neg_idx][2] + 2)) % num_cols

            neg_timediv_derived = edge_matrix[:, neg_timediv_list]
            neg_timediv_derived[neg_timediv_derived > 0] = 0
            neg_timediv_derived = np.sum(neg_timediv_derived, axis=1)

            neg_start_time_derived, neg_end_time_derived = find_edges(np.absolute(neg_timediv_derived))
            neg_edge_index_derived = np.where((neg_start_time_derived <= eligible_neg_edges[neg_idx][0]) &
                                              (neg_end_time_derived >= eligible_neg_edges[neg_idx][1]))[0]

            neg_edge_length_derived = neg_end_time_derived[neg_edge_index_derived[0]] - neg_start_time_derived[
                neg_edge_index_derived[0]]

            if (min(pos_edge_length_derived, neg_edge_length_derived) / max(pos_edge_length_derived,
                                                                            neg_edge_length_derived) < 0.25):
                continue

            match_start_day = match_start_days[neg_idx]
            match_end_day = match_end_days[neg_idx]
            duration_pts = duration_arr[neg_idx] * samples_per_hr

            # Get signal consistency matrix for the given edges

            day_f, match_f = get_consistency_val(data, match_start_day, match_end_day, min_amp[neg_idx], pos_edge_col,
                                                 eligible_neg_edges[neg_idx, 2], pp_config, window=0,
                                                 samples_per_hr=samples_per_hr)

            signal_score = 0
            if day_f != float('nan') and match_f != float('nan'):
                signal_score = int(50 * (day_f + match_f))

            add_pair = signal_consistency_bool(day_f, match_f, empty_flag, amp_ratios, neg_idx,
                                               minimum_day_signal_fraction, minimum_match_signal_ratio, min_amp_ratio)

            if not add_pair:
                continue

            pair_bool_arr = np.zeros(shape=(num_rows,))
            pair_bool_arr[match_start_day: match_end_day] = 1
            num_pairs_list.append(pair_bool_arr)

            duration_array = np.zeros(shape=(num_rows,))
            duration_array[match_start_day:match_end_day] = duration_pts
            duration_list.append(duration_array)

            matches.append([pos_edge_idx, eligible_neg_edges_idx[neg_idx], match_start_day, match_end_day,
                            pair_lengths[neg_idx], duration_pts, pos_edges[pos_edge_idx, 4],
                            eligible_neg_edges[neg_idx, 4], signal_score])

    matches = np.array(matches)

    if matches.shape[0] == 0:
        return matches, num_pairs_day, duration_days

    sorted_idx = matches[:, 2].argsort()
    sorted_matches = matches[sorted_idx, :].astype(int)

    num_pairs_day = np.sum(np.array(num_pairs_list), axis=0)
    duration_days = np.max(np.array(duration_list), axis=0)

    return sorted_matches, num_pairs_day, duration_days


def get_all_section_structures(pair_matrix, vpp_local_config):
    """Utility to extract all section structures"""

    edge_parity_matrix = np.sign(pair_matrix)

    day_arr, duration_arr, start_time_arr, end_time_arr = [], [], [], []
    pos_edge_count_arr, neg_edge_count_arr = [], []

    for day in range(vpp_local_config.get('total_days_count')):

        # find indexes and values for non zero indices

        edge_parity_dayarr = edge_parity_matrix[day, :]

        # Short form for edge parity is ep

        ep_dayarr_nonzero_timediv = np.where(~(edge_parity_dayarr == 0))[0]
        ep_dayarr_nonzero_values = edge_parity_dayarr[ep_dayarr_nonzero_timediv]

        # remove days if have one positive or negative edges or zero edges

        if ((len(ep_dayarr_nonzero_timediv) < 2) | (
                np.abs(np.sum(ep_dayarr_nonzero_values)) == len(ep_dayarr_nonzero_timediv))):
            continue

        ep_dayarr_nonzero_timediv, ep_dayarr_nonzero_values = roll_day_arr_pos_first(ep_dayarr_nonzero_timediv,
                                                                                     ep_dayarr_nonzero_values)

        n_to_p_transition_idx = np.where((ep_dayarr_nonzero_values[:-1] - ep_dayarr_nonzero_values[1:]) == -2)[0]

        # padding start and end index to transition index

        n_to_p_transition_idx = [0] + list(n_to_p_transition_idx) + [len(ep_dayarr_nonzero_timediv) - 1]

        # loop over sections to find duration, start time and end time for each section

        ep_input_dict = dict()
        ep_input_dict['ep_dayarr_nonzero_values'] = ep_dayarr_nonzero_values
        ep_input_dict['ep_dayarr_nonzero_timediv'] = ep_dayarr_nonzero_timediv
        day_arr, duration_arr, start_time_arr, end_time_arr, pos_edge_count_arr, neg_edge_count_arr = \
            find_section_parameters(ep_input_dict, n_to_p_transition_idx, day, day_arr, start_time_arr, end_time_arr,
                                    pos_edge_count_arr, neg_edge_count_arr, duration_arr, vpp_local_config)

    df = pd.DataFrame({
        'day': day_arr,
        'duration': duration_arr,
        'start_time': start_time_arr,
        'end_time': end_time_arr,
        'pos_edge_count': pos_edge_count_arr,
        'neg_edge_count': neg_edge_count_arr
    })

    return df


def find_section_parameters(ep_input_dict, n_to_p_transition_idx, day, day_arr, start_time_arr, end_time_arr,
                            pos_edge_count_arr, neg_edge_count_arr, duration_arr, vpp_local_config):
    """Finds duration, start time and end times for each section"""

    ep_dayarr_nonzero_values = ep_input_dict['ep_dayarr_nonzero_values']
    ep_dayarr_nonzero_timediv = ep_input_dict['ep_dayarr_nonzero_timediv']

    for i in range(len(n_to_p_transition_idx) - 1):
        section_start_idx, section_end_idx = n_to_p_transition_idx[i], n_to_p_transition_idx[i + 1]

        if section_start_idx == 0:
            section_start_timediv = ep_dayarr_nonzero_timediv[section_start_idx]
        else:
            section_start_timediv = (ep_dayarr_nonzero_timediv[section_start_idx + 1]) % (
                vpp_local_config.get('total_timediv_count'))

        section_end_timediv = ep_dayarr_nonzero_timediv[section_end_idx]

        if len(day_arr) > 0 and day_arr[-1] == day:
            section_duration_in_hour = ((section_end_timediv - section_start_timediv) % (
                vpp_local_config.get('total_timediv_count'))) / vpp_local_config.get('samples_per_hour')

            section_distance_in_hour = ((section_start_timediv - end_time_arr[-1]) % (
                vpp_local_config.get('total_timediv_count'))) / vpp_local_config.get('samples_per_hour')

            needs_merging = (section_duration_in_hour < vpp_local_config.get('section_min_duration_in_hour')) | \
                (section_distance_in_hour < vpp_local_config.get('section_min_duration_in_hour'))

            if needs_merging:
                # Merge this section into previous section

                end_time_arr[-1] = section_end_timediv
                pos_edge_count = pos_edge_count_arr[-1] + np.count_nonzero(
                    ep_dayarr_nonzero_values[section_start_idx:section_end_idx + 1] == 1)
                pos_edge_count_arr[-1] = pos_edge_count
                neg_edge_count = neg_edge_count_arr[-1] + np.count_nonzero(
                    ep_dayarr_nonzero_values[section_start_idx + 1:section_end_idx + 1] == -1)
                neg_edge_count_arr[-1] = neg_edge_count
                duration_arr[-1] = (end_time_arr[-1] - start_time_arr[
                    -1]) % vpp_local_config.get('total_timediv_count')

                continue

        # If Section is of sufficient Length

        if len(n_to_p_transition_idx) > 2:

            start_time_arr.append(section_start_timediv)
            end_time_arr.append(section_end_timediv)

            duration = (section_end_timediv - section_start_timediv) % vpp_local_config.get('total_timediv_count')
            duration_arr.append(duration)
        else:
            start_time_arr.append(section_start_timediv)
            end_time_arr.append((section_start_timediv - 1) % vpp_local_config.get('total_timediv_count'))
            duration_arr.append(vpp_local_config.get('total_timediv_count'))

        pos_edge_count = np.count_nonzero(ep_dayarr_nonzero_values[section_start_idx:section_end_idx + 1] == 1)
        pos_edge_count_arr.append(pos_edge_count)
        neg_edge_count = np.count_nonzero(ep_dayarr_nonzero_values[section_start_idx + 1:section_end_idx + 1] == -1)
        neg_edge_count_arr.append(neg_edge_count)
        day_arr.append(day)

    return day_arr, duration_arr, start_time_arr, end_time_arr, pos_edge_count_arr, neg_edge_count_arr


def find_consistent_struct_section(vpp_local_config, section_struct_arr, edge_matrix, raw_data, pp_config,
                                   global_vpp_type=None):
    """Utility to find consistent structure for section"""

    sec_col = {
        'day': 0,
        'duration': 1,
        'start_time': 2,
        'end_time': 3,
        'pos_edge_count': 4,
        'neg_edge_count': 5,
        'final_pos_edge_count': 6,
        'final_neg_edge_count': 7,
        'type': 8,
        'color': 9,
        'is_ppnn': 10,
        'is_filled': 11,
        'is_filled_indicator': 12,
    }

    stype_pn = 11
    stype_pnn = 12
    stype_ppn = 21
    stype_ppnn = 22

    # Initiate a zero matrix for section

    section_edge_matrix = np.zeros_like(edge_matrix)
    vpp_type = np.unique(section_struct_arr[:, sec_col.get('type')])[0]

    # Fill Section Edges

    for row_idx in range(section_struct_arr.shape[0]):
        # calculate timediv range to capture rolling conditions

        timediv_range = np.arange(section_struct_arr[row_idx, sec_col.get('start_time')],
                                  section_struct_arr[row_idx, sec_col.get('start_time')] +
                                  section_struct_arr[row_idx, sec_col.get('duration')] + 1) % \
                        (vpp_local_config.get('total_timediv_count'))

        section_edge_matrix[section_struct_arr[row_idx, sec_col.get('day')], timediv_range] = \
            edge_matrix[section_struct_arr[row_idx, sec_col.get('day')], timediv_range]

    pos_edges, neg_edges = label_edges(section_edge_matrix, pp_config, min_edge_length=2)

    unique_days, unique_idx, reverse_idx = np.unique(section_struct_arr[:, sec_col.get('day')], return_index=True,
                                                     return_inverse=True)

    df_days_filled = np.c_[
        section_struct_arr[unique_idx, :], np.zeros_like(unique_days), np.zeros_like(unique_days)]

    # Find Signal Consistent vpairs for section
    # Check for PPNN Type

    vpairs_dict = {}

    if vpp_type == stype_ppnn:

        vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges, raw_data,
                                                             vpairs_dict, df_days_filled, type_val=stype_ppnn)

        # If  global arch is PPN/PNN Drop to PPN/PNN, and find all its pairs as well

        if global_vpp_type == stype_ppn or global_vpp_type == stype_pnn:
            vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges,
                                                                 raw_data, vpairs_dict, df_days_filled,
                                                                 type_val=global_vpp_type)

        # If All Days Are not filled and It's a PPNN Structure, Try to find all PPN And PNN Structures as well

        df_days_filled[:, sec_col.get('is_filled_indicator')] = df_days_filled[:, sec_col.get('is_filled')] > 0

        if not (np.sum(df_days_filled[:, sec_col.get('is_filled_indicator')]) == df_days_filled.shape[0]) and \
                global_vpp_type == stype_ppnn:
            vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges,
                                                                 raw_data,
                                                                 vpairs_dict,
                                                                 df_days_filled, type_val=stype_ppn)

            vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges,
                                                                 raw_data,
                                                                 vpairs_dict,
                                                                 df_days_filled, type_val=stype_pnn)

    elif vpp_type == stype_ppn:

        vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges, raw_data,
                                                             vpairs_dict,
                                                             df_days_filled, type_val=stype_ppn)
    elif vpp_type == stype_pnn:

        vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges, raw_data,
                                                             vpairs_dict,
                                                             df_days_filled, type_val=stype_pnn)
    else:

        vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges, raw_data,
                                                             vpairs_dict,
                                                             df_days_filled, type_val=stype_pn)

    # If All days are not filled, Drop to PN Based selection
    df_days_filled[:, sec_col.get('is_filled_indicator')] = df_days_filled[:, sec_col.get('is_filled')] > 0

    if not (np.sum(df_days_filled[:, sec_col.get('is_filled_indicator')]) == df_days_filled.shape[0]):
        vpairs_dict, df_days_filled = find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges, raw_data,
                                                             vpairs_dict,
                                                             df_days_filled, type_val=stype_pn)

    return vpairs_dict


def get_td_fraction_from_vpairs(vpp_local_config, structures_dict):
    """Utility to get td fraction from vpairs"""

    start_day, end_day, timediv = 0, 1, 2

    pos_td_indicator_matrix = np.zeros((vpp_local_config.get('total_days_count'),
                                        vpp_local_config.get('total_timediv_count')),
                                       dtype=int)

    neg_td_indicator_matrix = np.zeros((vpp_local_config.get('total_days_count'),
                                        vpp_local_config.get('total_timediv_count')),
                                       dtype=int)

    duration_indicator_arr = np.zeros((vpp_local_config.get('total_timediv_count'),), dtype=int)

    for color in structures_dict.keys():
        for vpair in structures_dict[color].keys():

            for pos_edge in structures_dict[color][vpair]['pos_edges']:
                pos_td_indicator_matrix[pos_edge[start_day]:pos_edge[end_day] + 1, pos_edge[timediv]] = 1

            for neg_edge in structures_dict[color][vpair]['neg_edges']:
                neg_td_indicator_matrix[neg_edge[start_day]:neg_edge[end_day] + 1, neg_edge[timediv]] = 1

            # Add Common Days Length to duration

            duration_indicator_arr[structures_dict[color][vpair]['duration']] += (
                structures_dict[color][vpair]['common_end_day'] - structures_dict[color][vpair][
                    'common_start_day'])

    pos_td_fraction = np.sum(pos_td_indicator_matrix, axis=0) / pos_td_indicator_matrix.shape[0]
    neg_td_fraction = np.sum(neg_td_indicator_matrix, axis=0) / neg_td_indicator_matrix.shape[0]
    duration_td_fraction = duration_indicator_arr / duration_indicator_arr.shape[0]

    return pos_td_fraction, neg_td_fraction, duration_td_fraction


def get_vpair_score(vpp_local_config, vpair, pos_td_fraction_arr, neg_td_fraction_arr, duration_arr,
                    global_vpp_type):
    """Utility to get scores for vpairs"""

    # Write a Normal Scoring function
    # For this, Scoring Function is
    # amp_ratio+ 0.5*duration_fraction + (0.3outer_day+0.2*inner_day) + (0.3outer_window+0.2inner_window)

    score = 0.

    # If Pair is in permissable limits for amplitude, then full weight for amp else, No weight for amplitude

    if vpair['amplitude_ratio'] > vpp_local_config.get('min_amplitude_ratio_threshold_high'):
        score += vpp_local_config.get('score_weight_amplitude') * vpair['amplitude_ratio']

    # Add Score for Duration based on How far a Duration is from Duration which had maximum pair_days

    max_pairdays_duration = np.argmax(duration_arr)

    closeness_fraction_from_max_pairdays_duration = 1. - (
        np.absolute(vpair['duration'] - max_pairdays_duration) / vpp_local_config.get('total_timediv_count'))

    score += vpp_local_config.get('score_weight_duration_fraction') * closeness_fraction_from_max_pairdays_duration

    score += vpp_local_config.get('score_weight_inner_window') * vpair['inner_window_fraction']
    score += vpp_local_config.get('score_weight_outer_window') * vpair['outer_window_fraction']

    score += vpp_local_config.get('score_weight_inner_day') * vpair['inner_day_fraction']
    score += vpp_local_config.get('score_weight_outer_day') * vpair['outer_day_fraction']

    score += vpp_local_config.get('score_weight_type') * vpair['type']

    # Get time division masking score(Avg score of all edges)

    td_masking_score = 0
    edge_count = 0
    timediv = 2

    for pos_edge in vpair['pos_edges']:
        td_masking_score += pos_td_fraction_arr[pos_edge[timediv]] * vpp_local_config.get(
            'score_masked_td_fraction')
        td_masking_score += pos_td_fraction_arr[(pos_edge[timediv] - 1) %
                                                vpp_local_config.get('total_timediv_count')] * \
                            vpp_local_config.get('score_masked_td_fraction') / 2
        td_masking_score += pos_td_fraction_arr[(pos_edge[timediv] + 1) %
                                                vpp_local_config.get('total_timediv_count')] * \
                            vpp_local_config.get('score_masked_td_fraction') / 2
        edge_count += 1

    for neg_edge in vpair['neg_edges']:
        td_masking_score += neg_td_fraction_arr[neg_edge[timediv]] * vpp_local_config.get(
            'score_masked_td_fraction')
        td_masking_score += neg_td_fraction_arr[(neg_edge[timediv] - 1) %
                                                vpp_local_config.get('total_timediv_count')] * \
                            vpp_local_config.get('score_masked_td_fraction') / 2
        td_masking_score += neg_td_fraction_arr[(neg_edge[timediv] + 1) %
                                                vpp_local_config.get('total_timediv_count')] * \
                            vpp_local_config.get('score_masked_td_fraction') / 2

        edge_count += 1

    score += td_masking_score / edge_count

    # Score for being a Global Structure

    if vpair['type'] == global_vpp_type:
        score += vpp_local_config.get('score_global_vpp_type')

    return score


def remove_structures(vpp_local_config, vpairs_dict, pos_td_fraction_arr, neg_td_fraction_arr, duration_arr,
                      global_vpp_type):
    """Utility to remove overlapping consistent structures for section"""

    # Fill Days Iteratively based on best Selected vpair

    df_days_filled = pd.DataFrame(columns=['day'], data=np.arange(0, vpp_local_config.get('total_days_count')))
    df_days_filled.loc[:, 'isSectionDay'] = 0
    df_days_filled.loc[:, 'isFilled'] = 0

    df_days_filled.loc[:, 'vpairFilled'] = ''

    for vpair_key in vpairs_dict.keys():
        vpair = vpairs_dict[vpair_key]
        day_range = np.arange(vpair['common_start_day'],
                              vpair['common_end_day'] + 1)
        df_days_filled.loc[df_days_filled.day.isin(day_range), 'isSectionDay'] = 1

    # Mark Days Not in Section as filled

    df_days_filled.loc[df_days_filled.isSectionDay == 0, 'isFilled'] = 1

    if len(vpairs_dict.keys()) < 2:
        return vpairs_dict

    df_input_pairs = pd.DataFrame(columns=['vpair_key'], data=list(list(vpairs_dict.keys())))

    df_input_pairs.loc[:, 'vpair_score'] = df_input_pairs.vpair_key.apply(
        lambda x: get_vpair_score(vpp_local_config, vpairs_dict[x], pos_td_fraction_arr, neg_td_fraction_arr,
                                  duration_arr,
                                  global_vpp_type))

    df_input_pairs = df_input_pairs.sort_values(by='vpair_score', ascending=False).reset_index(drop=True)

    final_vpairs_dict = {}

    for index, row in df_input_pairs.iterrows():
        # If all Section Days are filled, Then break the loop
        if df_days_filled.isFilled.sum() == df_days_filled.isFilled.count():
            break

        key, score = row['vpair_key'], row['vpair_score']

        common_start_day, common_end_day = vpairs_dict[key]['common_start_day'], vpairs_dict[key]['common_end_day']
        df_days_filled_subset = df_days_filled[
            (df_days_filled.day >= common_start_day) & (df_days_filled.day <= common_end_day)].reset_index(drop=True)
        available_days_to_fill = df_days_filled_subset[(df_days_filled_subset.isFilled == 0)].day

        # Create vpairs for each section
        start_idx = 0
        start_timediv, end_timediv = 2, 3

        for seq_num, seq in iter_groupby(df_days_filled_subset.isFilled):

            seq_len = len(list(seq))
            idx_range = np.arange(start_idx, start_idx + seq_len)
            df_vpair = df_days_filled_subset[df_days_filled_subset.index.isin(idx_range)]

            if (df_vpair.isFilled.sum() == 0) & (df_vpair.shape[0] > 0):
                # Create New Vpair in this range for final list

                new_vpair = copy.deepcopy(vpairs_dict[key])
                new_vpair['common_start_day'], new_vpair['common_end_day'] = df_vpair.day.min(), df_vpair.day.max()
                start_timediv_idx, end_timediv_idx = key.split(",")[start_timediv], key.split(",")[end_timediv]
                new_key = "%d,%d,%s,%s,%.f" % (
                    new_vpair['common_start_day'], new_vpair['common_end_day'], start_timediv_idx, end_timediv_idx,
                    new_vpair['type'])
                final_vpairs_dict[new_key] = new_vpair

            start_idx += (len(idx_range))

        # Fill available days

        df_days_filled.loc[df_days_filled.day.isin(available_days_to_fill), 'vpairFilled'] = key
        df_days_filled.loc[df_days_filled.day.isin(available_days_to_fill), 'isFilled'] = 1

    return final_vpairs_dict


def get_all_consistent_vpairs(vpp_local_config, raw_data, edge_matrix, vpp_type, pp_config):
    """Utility tot get all signal consistent vpairs"""

    # Get sections in new array

    df_section_structures = get_all_section_structures(edge_matrix, vpp_local_config)
    global_pos_edge_count, global_neg_edge_count = vpp_type // 10, vpp_type % 10

    sec_col = {
        'day': 0,
        'duration': 1,
        'start_time': 2,
        'end_time': 3,
        'pos_edge_count': 4,
        'neg_edge_count': 5,
        'final_pos_edge_count': 6,
        'final_neg_edge_count': 7,
        'type': 8,
        'color': 9,
        'is_ppnn': 10,
    }

    df_section_structures_arr = df_section_structures.values

    # Get final positive and negative edge count

    final_pos_edge_count = df_section_structures_arr[:, sec_col.get('pos_edge_count')].copy()
    final_pos_edge_count[final_pos_edge_count > 2] = global_pos_edge_count

    final_neg_edge_count = df_section_structures_arr[:, sec_col.get('neg_edge_count')].copy()
    final_neg_edge_count[final_neg_edge_count > 2] = global_neg_edge_count

    type_arr = 10 * final_pos_edge_count + final_neg_edge_count

    is_ppnn_arr = np.full(shape=(df_section_structures_arr.shape[0],), fill_value=False)

    if (global_pos_edge_count != 2) or (global_neg_edge_count != 2):
        is_ppnn_arr = np.logical_and(final_pos_edge_count == 2, final_neg_edge_count == 2)

    section_colors_arr = np.c_[df_section_structures_arr[:, sec_col.get('duration'): sec_col.get('end_time') + 1],
                               type_arr]

    if len(section_colors_arr) == 0:
        return {}

    sec_colors_unique_arr, reverse_idx = np.unique(section_colors_arr, return_inverse=True, axis=0)
    colors = np.arange(0, sec_colors_unique_arr.shape[0])

    color_arr = np.zeros_like(type_arr)
    color_arr[:] = colors[reverse_idx]

    structures_dict = {}

    df_section_structures_arr = np.c_[
        df_section_structures_arr, final_pos_edge_count, final_neg_edge_count, type_arr,
        color_arr, is_ppnn_arr]

    for color in colors:
        section_struct_arr_subset = df_section_structures_arr[color_arr == color, :]

        section_structures_dict = find_consistent_struct_section(vpp_local_config, section_struct_arr_subset,
                                                                 edge_matrix, raw_data, pp_config,
                                                                 global_vpp_type=vpp_type)
        structures_dict[color] = section_structures_dict

    # Get weight of time divisions used in scoring

    pos_td_fraction_arr, neg_td_fraction_arr, duration_arr = get_td_fraction_from_vpairs(vpp_local_config,
                                                                                         structures_dict)

    # Remove overlapping pairs from all pairs in a section

    for color in colors:
        section_structures_dict = structures_dict[color]
        clean_vpairs = remove_structures(vpp_local_config, section_structures_dict, pos_td_fraction_arr,
                                         neg_td_fraction_arr, duration_arr, global_vpp_type=vpp_type)

        structures_dict[color] = clean_vpairs

    # Remove Overlapping sections in a Day,
    # Get weight of time divisions used in scoring

    pos_td_fraction_arr_clean, neg_td_fraction_arr_clean, duration_arr_clean = \
        get_td_fraction_from_vpairs(vpp_local_config, structures_dict)

    # Remove overlapping pairs from all pairs in a section

    all_vpairs_dict = {}

    for color in colors:
        for vpair_key in structures_dict[color]:
            all_vpairs_dict[vpair_key] = structures_dict[color][vpair_key]

    all_clean_vpairs = remove_structures(vpp_local_config, all_vpairs_dict, pos_td_fraction_arr_clean,
                                         neg_td_fraction_arr_clean, duration_arr_clean, global_vpp_type=vpp_type)

    # Setting zero to be new Dummy color, TO be fixed..

    new_structures_dict = dict()
    new_structures_dict[0] = all_clean_vpairs

    return new_structures_dict
