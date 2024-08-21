"""
Author - Arpan Agrawal
Date - 04/04/2019
Function to find consistent vpairs for Variable PP
"""

import numpy as np
from functools import reduce
from itertools import permutations

from python3.disaggregation.aer.poolpump.functions.variable_pp_utils_2 import check_signal_consistency


def find_consistent_vpairs(vpp_local_config, pos_edges, neg_edges, raw_data, vpairs_dict, df_days_filled, type_val):
    """Utility to find consistent v pairs"""

    stype_pn = 11
    stype_pnn = 12
    stype_ppn = 21
    stype_ppnn = 22

    day_col = 0
    is_filled_col = 11

    # Write enum for addressing edges

    start_day, end_day, timediv, amplitude, sign = 0, 1, 2, 3, 4

    # FIND permutation for positive and negative edges

    pos_edge_index_pairs = list(permutations(range(len(pos_edges)), 2))
    neg_edge_index_pairs = list(permutations(range(len(neg_edges)), 2))

    input_dict = dict()
    input_dict['pos_edge_index_pairs'] = pos_edge_index_pairs
    input_dict['neg_edge_index_pairs'] = neg_edge_index_pairs
    input_dict['timediv'] = timediv
    input_dict['start_day'] = start_day
    input_dict['end_day'] = end_day
    input_dict['amplitude'] = amplitude
    input_dict['stype_ppnn'] = stype_ppnn
    input_dict['stype_ppn'] = stype_ppn
    input_dict['stype_pnn'] = stype_pnn
    input_dict['stype_pn'] = stype_pn
    input_dict['type_val'] = type_val
    input_dict['raw_data'] = raw_data

    # Find Signal Consistent pairs for PPNN type

    if type_val == stype_ppnn:
        vpairs_dict, df_days_filled = stype_ppnn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled,
                                                        day_col, is_filled_col, vpp_local_config)

    if type_val == stype_ppn:
        vpairs_dict, df_days_filled = stype_ppn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled,
                                                       day_col, is_filled_col, vpp_local_config)

    if type_val == stype_pnn:
        vpairs_dict, df_days_filled = stype_pnn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled,
                                                       day_col, is_filled_col, vpp_local_config)

    if type_val == stype_pn:
        vpairs_dict, df_days_filled = stype_pn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled,
                                                      day_col, is_filled_col, vpp_local_config)

    return vpairs_dict, df_days_filled


def stype_ppnn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                        vpp_local_config):
    """
    Utility for stype_ppn_checks, written to reduce number of return statements for fixing Sonar Issue
    """
    timediv = input_dict['timediv']
    start_day = input_dict['start_day']
    end_day = input_dict['end_day']
    amplitude = input_dict['amplitude']
    stype_ppnn = input_dict['stype_ppnn']
    type_val = input_dict['type_val']
    raw_data = input_dict['raw_data']

    pos_outer_idx = idx_dict['pos_outer_idx']
    pos_inner_idx = idx_dict['pos_inner_idx']
    neg_inner_idx = idx_dict['neg_inner_idx']
    neg_outer_idx = idx_dict['neg_outer_idx']

    pair_attributes = {}
    bool_val = True

    pos_outer_days = np.arange(pos_edges[pos_outer_idx][start_day], pos_edges[pos_outer_idx][end_day])
    pos_inner_days = np.arange(pos_edges[pos_inner_idx][start_day], pos_edges[pos_inner_idx][end_day])

    neg_inner_days = np.arange(neg_edges[neg_inner_idx][start_day], neg_edges[neg_inner_idx][end_day])
    neg_outer_days = np.arange(neg_edges[neg_outer_idx][start_day], neg_edges[neg_outer_idx][end_day])

    intersection_days = reduce(np.intersect1d, (pos_outer_days, pos_inner_days, neg_inner_days,
                                                neg_outer_days))
    common_start_day, common_end_day = intersection_days.min(), intersection_days.max()

    pair_attributes['common_start_day'] = common_start_day
    pair_attributes['common_end_day'] = common_end_day

    # CHECK 3: Amplitude Ballpark, (P1 + P2) should be nearly equal to (N1+N2)

    pos_amplitude_sum = pos_edges[pos_outer_idx][amplitude] + pos_edges[pos_inner_idx][amplitude]
    neg_amplitude_sum = neg_edges[neg_inner_idx][amplitude] + neg_edges[neg_outer_idx][amplitude]

    abs_neg_amplitude_sum = np.absolute(neg_amplitude_sum)

    amplitude_ratio = min(abs_neg_amplitude_sum, pos_amplitude_sum) / max(abs_neg_amplitude_sum,
                                                                          pos_amplitude_sum)
    if amplitude_ratio < vpp_local_config.get('min_amplitude_ratio_threshold_low'):
        bool_val = False

    if bool_val:
        pair_attributes['amplitude_ratio'] = amplitude_ratio

        # CHECK 4a: Duration Ballpark, 1 to 18 hrs(when developed) between last p1 and first p2 edge

        vpair_duration_in_td = (neg_edges[neg_outer_idx][timediv] - pos_edges[pos_outer_idx][
            timediv]) % vpp_local_config.get('total_timediv_count')

        vpair_duration_in_hour = vpair_duration_in_td / vpp_local_config.get('samples_per_hour')

        if ((vpair_duration_in_hour < vpp_local_config.get('vpair_min_duration_in_hour')) | (
                vpair_duration_in_hour > vpp_local_config.get('vpair_max_duration_in_hour'))):
            bool_val = False

        if bool_val:
            pair_attributes['duration'] = vpair_duration_in_td

    # CHECK 4b: Inter Edge Duration, Should be more than half hour for pair to pass

    if bool_val:
        pos_outer_pos_inner_inter_duration = \
            (pos_edges[pos_inner_idx][timediv] -
             pos_edges[pos_outer_idx][timediv]) % vpp_local_config.get('total_timediv_count')
        is_inter_duration_less = \
            pos_outer_pos_inner_inter_duration < (vpp_local_config.get('vpair_min_inter_duration_in_hour') *
                                                  vpp_local_config.get('samples_per_hour'))

        pos_inner_neg_inner_inter_duration = \
            (neg_edges[neg_inner_idx][timediv] -
             pos_edges[pos_inner_idx][timediv]) % vpp_local_config.get('total_timediv_count')

        is_inter_duration_less = is_inter_duration_less | (pos_inner_neg_inner_inter_duration <
                                                           (vpp_local_config.get('vpair_min_inter_duration_in_hour') *
                                                            vpp_local_config.get('samples_per_hour')))

        neg_inner_neg_outer_inter_duration = \
            (neg_edges[neg_outer_idx][timediv] -
             neg_edges[neg_inner_idx][timediv]) % vpp_local_config.get('total_timediv_count')

        is_inter_duration_less = is_inter_duration_less | (neg_inner_neg_outer_inter_duration <
                                                           (vpp_local_config.get('vpair_min_inter_duration_in_hour') *
                                                            vpp_local_config.get('samples_per_hour')))

        if is_inter_duration_less:
            bool_val = False

    if not bool_val:
        return bool_val, vpairs_dict, df_days_filled

    # CHECK 5: Signal Consistency For inner and outer pairs

    # Signal Consistency for Outer Pair

    pos_outer_overlapped_edge = pos_edges[pos_outer_idx].copy()
    pos_outer_overlapped_edge[start_day], pos_outer_overlapped_edge[
        end_day] = common_start_day, common_end_day

    neg_outer_overlapped_edge = neg_edges[neg_outer_idx].copy()
    neg_outer_overlapped_edge[start_day], neg_outer_overlapped_edge[
        end_day] = common_start_day, common_end_day

    outer_day_fraction, outer_window_fraction = \
        check_signal_consistency(vpp_local_config, pos_outer_overlapped_edge, neg_outer_overlapped_edge,
                                 raw_data)

    # Signal Consistency for Inner Pair

    pos_inner_overlapped_edge = pos_edges[pos_inner_idx].copy()

    pos_inner_overlapped_edge[start_day] = common_start_day
    pos_inner_overlapped_edge[end_day] = common_end_day

    neg_inner_overlapped_edge = neg_edges[neg_inner_idx].copy()

    neg_inner_overlapped_edge[start_day] = common_start_day
    neg_inner_overlapped_edge[end_day] = common_end_day

    # Setting Amp to be min(sum) positive and negative
    inner_pair_amp_threshold = min(pos_amplitude_sum, abs_neg_amplitude_sum)

    inner_day_fraction, inner_window_fraction = \
        check_signal_consistency(vpp_local_config, pos_inner_overlapped_edge, neg_inner_overlapped_edge,
                                 raw_data, amp=inner_pair_amp_threshold)

    if ((inner_day_fraction < vpp_local_config.get('vpair_min_day_fraction')) or (
            outer_day_fraction < vpp_local_config.get('vpair_min_day_fraction'))):
        bool_val = False

    if bool_val:
        pair_attributes['inner_day_fraction'], pair_attributes['outer_day_fraction'] = \
            inner_day_fraction, outer_day_fraction

    if bool_val and ((inner_window_fraction < vpp_local_config.get('vpair_min_window_fraction')) or
                     (outer_window_fraction < vpp_local_config.get('vpair_min_window_fraction'))):
        bool_val = False

    if not bool_val:
        return bool_val, vpairs_dict, df_days_filled

    pair_attributes['inner_window_fraction'], pair_attributes[
        'outer_window_fraction'] = inner_window_fraction, outer_window_fraction

    vpair_key = "%d,%d,%d,%d,%d" % (
        common_start_day, common_end_day, pos_edges[pos_outer_idx][timediv],
        neg_edges[neg_outer_idx][timediv], type_val)

    pair_attributes['type'] = stype_ppnn
    pair_attributes['pos_edges'] = pos_outer_overlapped_edge, pos_inner_overlapped_edge
    pair_attributes['neg_edges'] = neg_inner_overlapped_edge, neg_outer_overlapped_edge

    if vpair_key in vpairs_dict.keys():
        vpair_key = vpair_key + ',_r'

    vpairs_dict[vpair_key] = pair_attributes

    df_days_filled[np.isin(df_days_filled[:, day_col], intersection_days), is_filled_col] = 1

    return bool_val, vpairs_dict, df_days_filled


def stype_ppnn_checks(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                      vpp_local_config):
    """Checks for inner and outer edges of each PPNN structure"""

    timediv = input_dict['timediv']
    start_day = input_dict['start_day']
    end_day = input_dict['end_day']

    pos_outer_idx = idx_dict['pos_outer_idx']
    pos_inner_idx = idx_dict['pos_inner_idx']
    neg_inner_idx = idx_dict['neg_inner_idx']
    neg_outer_idx = idx_dict['neg_outer_idx']

    # CHECK 1: Order of edges should be consistent

    pos_outer_td, pos_inner_td = pos_edges[pos_outer_idx][timediv], pos_edges[pos_inner_idx][timediv]
    neg_inner_td, neg_outer_td = neg_edges[neg_inner_idx][timediv], neg_edges[neg_outer_idx][timediv]
    td_length = vpp_local_config.get('total_timediv_count')

    # Check p1-p2<p1-p3<p1-p4

    are_edge_td_ordered = ((pos_inner_td - pos_outer_td) % td_length < (neg_inner_td - pos_outer_td) % td_length)
    are_edge_td_ordered = \
        are_edge_td_ordered & ((neg_inner_td - pos_outer_td) % td_length < (neg_outer_td - pos_outer_td) % td_length)
    are_edge_td_ordered = \
        are_edge_td_ordered & ((neg_inner_td - pos_inner_td) % td_length < (neg_outer_td - pos_inner_td) % td_length)

    if not are_edge_td_ordered:
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    # CHECK 2: Time overlap, Total overlap should be more than 1 day

    pos_outer_days = np.arange(pos_edges[pos_outer_idx][start_day], pos_edges[pos_outer_idx][end_day])
    pos_inner_days = np.arange(pos_edges[pos_inner_idx][start_day], pos_edges[pos_inner_idx][end_day])

    neg_inner_days = np.arange(neg_edges[neg_inner_idx][start_day], neg_edges[neg_inner_idx][end_day])
    neg_outer_days = np.arange(neg_edges[neg_outer_idx][start_day], neg_edges[neg_outer_idx][end_day])

    intersection_days = reduce(np.intersect1d, (pos_outer_days, pos_inner_days, neg_inner_days,
                                                neg_outer_days))

    if len(intersection_days) < vpp_local_config.get('min_intersection_days'):
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    bool_val, vpairs_dict, df_days_filled = stype_ppnn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict,
                                                                df_days_filled, day_col, is_filled_col,
                                                                vpp_local_config)

    return bool_val, vpairs_dict, df_days_filled


def stype_ppnn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                      vpp_local_config):
    """Finds vpairs for PPNN type"""

    pos_edge_index_pairs = input_dict['pos_edge_index_pairs']
    neg_edge_index_pairs = input_dict['neg_edge_index_pairs']

    for pos_outer_idx, pos_inner_idx in pos_edge_index_pairs:
        for neg_inner_idx, neg_outer_idx in neg_edge_index_pairs:

            idx_dict = dict()
            idx_dict['pos_outer_idx'] = pos_outer_idx
            idx_dict['pos_inner_idx'] = pos_inner_idx
            idx_dict['neg_inner_idx'] = neg_inner_idx
            idx_dict['neg_outer_idx'] = neg_outer_idx
            bool_val, vpairs_dict, df_days_filled = stype_ppnn_checks(input_dict, idx_dict, pos_edges, neg_edges,
                                                                      vpairs_dict, df_days_filled, day_col,
                                                                      is_filled_col, vpp_local_config)
            if not bool_val:
                continue

    return vpairs_dict, df_days_filled


def stype_ppn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                       vpp_local_config):
    """
    Utility for stype_ppn_checks, written to reduce number of return statements for fixing Sonar Issue
    """
    timediv = input_dict['timediv']
    start_day = input_dict['start_day']
    end_day = input_dict['end_day']
    amplitude = input_dict['amplitude']
    stype_ppn = input_dict['stype_ppn']
    type_val = input_dict['type_val']
    raw_data = input_dict['raw_data']

    pos_outer_idx = idx_dict['pos_outer_idx']
    pos_inner_idx = idx_dict['pos_inner_idx']
    neg_idx = idx_dict['neg_idx']

    pair_attributes = {}
    bool_val = True

    pos_outer_days = np.arange(pos_edges[pos_outer_idx][start_day], pos_edges[pos_outer_idx][end_day])
    pos_inner_days = np.arange(pos_edges[pos_inner_idx][start_day], pos_edges[pos_inner_idx][end_day])

    neg_days = np.arange(neg_edges[neg_idx][start_day], neg_edges[neg_idx][end_day])

    intersection_days = reduce(np.intersect1d, (pos_outer_days, pos_inner_days, neg_days))
    common_start_day, common_end_day = intersection_days.min(), intersection_days.max()

    pair_attributes['common_start_day'] = common_start_day
    pair_attributes['common_end_day'] = common_end_day

    pos_amplitude_sum = pos_edges[pos_outer_idx][amplitude] + pos_edges[pos_inner_idx][amplitude]
    neg_amplitude_sum = neg_edges[neg_idx][amplitude]

    abs_neg_amplitude_sum = np.absolute(neg_amplitude_sum)

    amplitude_ratio = min(abs_neg_amplitude_sum, pos_amplitude_sum) / max(abs_neg_amplitude_sum,
                                                                          pos_amplitude_sum)

    if amplitude_ratio < vpp_local_config.get('min_amplitude_ratio_threshold_low'):
        bool_val = False

    if bool_val:
        pair_attributes['amplitude_ratio'] = amplitude_ratio

        # CHECK 4: Duration Ballpark, 1 to 18 hrs(when developed) between last p1 and first p2 edge

        vpair_duration_in_td = (neg_edges[neg_idx][timediv] - pos_edges[pos_outer_idx][
            timediv]) % vpp_local_config.get('total_timediv_count')

        vpair_duration_in_hour = vpair_duration_in_td / vpp_local_config.get('samples_per_hour')

        if ((vpair_duration_in_hour < vpp_local_config.get('vpair_min_duration_in_hour')) | (
                vpair_duration_in_hour > vpp_local_config.get('vpair_max_duration_in_hour'))):
            bool_val = False

        pair_attributes['duration'] = vpair_duration_in_td

    # CHECK 4b: Inter Edge Duration, Should be more than half hour for pair to pass

    if bool_val:
        pos_outer_pos_inner_inter_duration = \
            (pos_edges[pos_inner_idx][timediv] -
             pos_edges[pos_outer_idx][timediv]) % vpp_local_config.get('total_timediv_count')
        is_inter_duration_less = \
            pos_outer_pos_inner_inter_duration < (vpp_local_config.get('vpair_min_inter_duration_in_hour') *
                                                  vpp_local_config.get('samples_per_hour'))

        pos_inner_neg_inter_duration = (neg_edges[neg_idx][timediv] - pos_edges[pos_inner_idx][
            timediv]) % vpp_local_config.get('total_timediv_count')
        is_inter_duration_less = is_inter_duration_less | (pos_inner_neg_inter_duration <
                                                           (vpp_local_config.get('vpair_min_inter_duration_in_hour') *
                                                            vpp_local_config.get('samples_per_hour')))

        if is_inter_duration_less:
            bool_val = False

    if not bool_val:
        return bool_val, vpairs_dict, df_days_filled

    # CHECK 5: Signal Consistency For inner and outer pairs

    # Signal Consistency for Inner Pair

    pos_inner_overlapped_edge = pos_edges[pos_inner_idx].copy()
    pos_inner_overlapped_edge[start_day], pos_inner_overlapped_edge[
        end_day] = common_start_day, common_end_day

    neg_overlapped_edge = neg_edges[neg_idx].copy()
    neg_overlapped_edge[start_day], neg_overlapped_edge[
        end_day] = common_start_day, common_end_day

    # Setting Amp to be min(sum) positive and negative

    inner_pair_amp_threshold = min(pos_amplitude_sum, abs_neg_amplitude_sum)

    inner_day_fraction, inner_window_fraction = check_signal_consistency(vpp_local_config,
                                                                         pos_inner_overlapped_edge,
                                                                         neg_overlapped_edge, raw_data,
                                                                         amp=inner_pair_amp_threshold)
    # Signal Consistency for Outer Pair

    pos_outer_overlapped_edge = pos_edges[pos_outer_idx].copy()
    pos_outer_overlapped_edge[start_day], pos_outer_overlapped_edge[
        end_day] = common_start_day, common_end_day

    neg_overlapped_edge = neg_edges[neg_idx].copy()
    neg_overlapped_edge[start_day], neg_overlapped_edge[
        end_day] = common_start_day, common_end_day

    outer_day_fraction, outer_window_fraction = check_signal_consistency(vpp_local_config,
                                                                         pos_outer_overlapped_edge,
                                                                         neg_overlapped_edge, raw_data)

    if ((inner_day_fraction < vpp_local_config.get('vpair_min_day_fraction')) | (
            outer_day_fraction < vpp_local_config.get('vpair_min_day_fraction'))):
        bool_val = False

    if bool_val:
        pair_attributes['inner_day_fraction'], pair_attributes[
            'outer_day_fraction'] = inner_day_fraction, outer_day_fraction

        if ((inner_window_fraction < vpp_local_config.get('vpair_min_window_fraction')) | (
                outer_window_fraction < vpp_local_config.get('vpair_min_window_fraction'))):
            bool_val = False

    if not bool_val:
        return bool_val, vpairs_dict, df_days_filled

    pair_attributes['inner_window_fraction'], pair_attributes[
        'outer_window_fraction'] = inner_window_fraction, outer_window_fraction

    vpair_key = "%d,%d,%d,%d,%d" % (
        common_start_day, common_end_day, pos_edges[pos_outer_idx][timediv],
        neg_edges[neg_idx][timediv],
        type_val)

    pair_attributes['type'] = stype_ppn
    pair_attributes['pos_edges'] = pos_outer_overlapped_edge, pos_inner_overlapped_edge
    pair_attributes['neg_edges'] = [neg_overlapped_edge]

    if vpair_key in vpairs_dict.keys():
        vpair_key = vpair_key + ',_r'
    vpairs_dict[vpair_key] = pair_attributes

    df_days_filled[np.isin(df_days_filled[:, day_col], intersection_days), is_filled_col] = 1

    return bool_val, vpairs_dict, df_days_filled


def stype_ppn_checks(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                     vpp_local_config):
    """Checks for inner and outer edges of each PPN structure"""

    timediv = input_dict['timediv']
    start_day = input_dict['start_day']
    end_day = input_dict['end_day']

    pos_outer_idx = idx_dict['pos_outer_idx']
    pos_inner_idx = idx_dict['pos_inner_idx']
    neg_idx = idx_dict['neg_idx']

    # CHECK 1: Order of edges should be consistent

    pos_outer_td, pos_inner_td = pos_edges[pos_outer_idx][timediv], pos_edges[pos_inner_idx][timediv]
    neg_td = neg_edges[neg_idx][timediv]
    td_length = vpp_local_config.get('total_timediv_count')

    # Check p1-p2<p1-p3<p1-p4

    are_edge_td_ordered = ((pos_inner_td - pos_outer_td) % td_length < (neg_td - pos_outer_td) % td_length)

    if not are_edge_td_ordered:
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    # CHECK 2: Time overlap, Total overlap should be more than 1 day

    pos_outer_days = np.arange(pos_edges[pos_outer_idx][start_day], pos_edges[pos_outer_idx][end_day])
    pos_inner_days = np.arange(pos_edges[pos_inner_idx][start_day], pos_edges[pos_inner_idx][end_day])

    neg_days = np.arange(neg_edges[neg_idx][start_day], neg_edges[neg_idx][end_day])

    intersection_days = reduce(np.intersect1d, (pos_outer_days, pos_inner_days, neg_days))

    if len(intersection_days) < vpp_local_config.get('min_intersection_days'):
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    bool_val, vpairs_dict, df_days_filled = stype_ppn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict,
                                                               df_days_filled, day_col, is_filled_col, vpp_local_config)

    return bool_val, vpairs_dict, df_days_filled


def stype_ppn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                     vpp_local_config):
    """Finds vpairs for PPN type"""

    pos_edge_index_pairs = input_dict['pos_edge_index_pairs']

    for pos_outer_idx, pos_inner_idx in pos_edge_index_pairs:
        for neg_idx in range(len(neg_edges)):
            idx_dict = dict()
            idx_dict['pos_outer_idx'] = pos_outer_idx
            idx_dict['pos_inner_idx'] = pos_inner_idx
            idx_dict['neg_idx'] = neg_idx

            bool_val, vpairs_dict, df_days_filled = stype_ppn_checks(input_dict, idx_dict, pos_edges, neg_edges,
                                                                     vpairs_dict, df_days_filled, day_col,
                                                                     is_filled_col, vpp_local_config)
            if not bool_val:
                continue

    return vpairs_dict, df_days_filled


def stype_pnn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, pair_attributes,
                       day_col, is_filled_col, vpp_local_config):
    """
    Utility for stype_pnn_checks, written to reduce number of return statements for fixing Sonar Issue
    """
    timediv = input_dict['timediv']
    start_day = input_dict['start_day']
    end_day = input_dict['end_day']
    amplitude = input_dict['amplitude']
    stype_pnn = input_dict['stype_pnn']
    type_val = input_dict['type_val']
    raw_data = input_dict['raw_data']

    pos_idx = idx_dict['pos_idx']
    neg_inner_idx = idx_dict['neg_inner_idx']
    neg_outer_idx = idx_dict['neg_outer_idx']

    bool_val = True

    pos_days = np.arange(pos_edges[pos_idx][start_day], pos_edges[pos_idx][end_day])

    neg_inner_days = np.arange(neg_edges[neg_inner_idx][start_day], neg_edges[neg_inner_idx][end_day])
    neg_outer_days = np.arange(neg_edges[neg_outer_idx][start_day], neg_edges[neg_outer_idx][end_day])

    intersection_days = reduce(np.intersect1d, (pos_days, neg_inner_days, neg_outer_days))
    common_start_day, common_end_day = intersection_days.min(), intersection_days.max()

    pair_attributes['common_start_day'] = common_start_day
    pair_attributes['common_end_day'] = common_end_day

    # CHECK 3: Amplitude Ballpark, (P1 + P2) should be nearly equal to (N1+N2)

    pos_amplitude_sum = pos_edges[pos_idx][amplitude]
    neg_amplitude_sum = neg_edges[neg_inner_idx][amplitude] + neg_edges[neg_outer_idx][amplitude]
    abs_neg_amplitude_sum = np.absolute(neg_amplitude_sum)

    amplitude_ratio = min(abs_neg_amplitude_sum, pos_amplitude_sum) / max(abs_neg_amplitude_sum,
                                                                          pos_amplitude_sum)

    if amplitude_ratio < vpp_local_config.get('min_amplitude_ratio_threshold_low'):
        bool_val = False

    if bool_val:
        pair_attributes['amplitude_ratio'] = amplitude_ratio

    # CHECK 4: Duration Ballpark, 1 to 18 hrs(when developed) between last p1 and first p2 edge

    if bool_val:
        vpair_duration_in_td = (neg_edges[neg_outer_idx][timediv] - pos_edges[pos_idx][
            timediv]) % vpp_local_config.get('total_timediv_count')

        vpair_duration_in_hour = vpair_duration_in_td / vpp_local_config.get('samples_per_hour')

        if ((vpair_duration_in_hour < vpp_local_config.get('vpair_min_duration_in_hour')) | (
                vpair_duration_in_hour > vpp_local_config.get('vpair_max_duration_in_hour'))):
            bool_val = False

        pair_attributes['duration'] = vpair_duration_in_td

    if bool_val:

        # CHECK 4b: Inter Edge Duration, Should be more than half hour for pair to pass

        pos_neg_inner_inter_duration = \
            (neg_edges[neg_inner_idx][timediv] -
             pos_edges[pos_idx][timediv]) % vpp_local_config.get('total_timediv_count')

        is_inter_duration_less = \
            (pos_neg_inner_inter_duration < (vpp_local_config.get('vpair_min_inter_duration_in_hour') *
                                             vpp_local_config.get('samples_per_hour')))

        neg_inner_neg_outer_inter_duration = \
            (neg_edges[neg_outer_idx][timediv] -
             neg_edges[neg_inner_idx][timediv]) % vpp_local_config.get('total_timediv_count')

        is_inter_duration_less = is_inter_duration_less | (neg_inner_neg_outer_inter_duration <
                                                           (vpp_local_config.get('vpair_min_inter_duration_in_hour') *
                                                            vpp_local_config.get('samples_per_hour')))

        if is_inter_duration_less:
            bool_val = False

    if not bool_val:
        return bool_val, vpairs_dict, df_days_filled

    # CHECK 5: Signal Consistency For inner and outer pairs

    # Signal Consistency for Inner Pair

    pos_overlapped_edge = pos_edges[pos_idx].copy()
    pos_overlapped_edge[start_day], pos_overlapped_edge[
        end_day] = common_start_day, common_end_day

    neg_inner_overlapped_edge = neg_edges[neg_inner_idx].copy()
    neg_inner_overlapped_edge[start_day], neg_inner_overlapped_edge[
        end_day] = common_start_day, common_end_day

    # Setting Amp to be min(sum) positive and negative

    inner_pair_amp_threshold = min(pos_amplitude_sum, abs_neg_amplitude_sum)

    inner_day_fraction, inner_window_fraction = \
        check_signal_consistency(vpp_local_config, pos_overlapped_edge, neg_inner_overlapped_edge,
                                 raw_data,
                                 amp=inner_pair_amp_threshold)

    # Signal Consistency for Outer Pair

    pos_overlapped_edge = pos_edges[pos_idx].copy()
    pos_overlapped_edge[start_day], pos_overlapped_edge[
        end_day] = common_start_day, common_end_day

    neg_outer_overlapped_edge = neg_edges[neg_outer_idx].copy()
    neg_outer_overlapped_edge[start_day], neg_outer_overlapped_edge[
        end_day] = common_start_day, common_end_day

    outer_day_fraction, outer_window_fraction = \
        check_signal_consistency(vpp_local_config, pos_overlapped_edge, neg_outer_overlapped_edge,
                                 raw_data)

    if ((inner_day_fraction < vpp_local_config.get('vpair_min_day_fraction')) or (
            outer_day_fraction < vpp_local_config.get('vpair_min_day_fraction'))):
        bool_val = False

    if bool_val:
        pair_attributes['inner_day_fraction'], pair_attributes['outer_day_fraction'] = \
            inner_day_fraction, outer_day_fraction

        if ((inner_window_fraction < vpp_local_config.get('vpair_min_window_fraction')) or (
                outer_window_fraction < vpp_local_config.get('vpair_min_window_fraction'))):
            bool_val = False

    if not bool_val:
        return bool_val, vpairs_dict, df_days_filled

    pair_attributes['inner_window_fraction'], pair_attributes[
        'outer_window_fraction'] = inner_window_fraction, outer_window_fraction

    vpair_key = "%d,%d,%d,%d,%d" % (
        common_start_day, common_end_day, pos_edges[pos_idx][timediv],
        neg_edges[neg_outer_idx][timediv],
        type_val)

    pair_attributes['type'] = stype_pnn
    pair_attributes['pos_edges'] = [pos_overlapped_edge]
    pair_attributes['neg_edges'] = neg_inner_overlapped_edge, neg_outer_overlapped_edge

    if vpair_key in vpairs_dict.keys():
        vpair_key = vpair_key + ',_r'

    vpairs_dict[vpair_key] = pair_attributes

    df_days_filled[np.isin(df_days_filled[:, day_col], intersection_days), is_filled_col] = 1

    return bool_val, vpairs_dict, df_days_filled


def stype_pnn_checks(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                     vpp_local_config):
    """Checks for inner and outer edges of each PNN structure"""

    timediv = input_dict['timediv']
    start_day = input_dict['start_day']
    end_day = input_dict['end_day']

    pos_idx = idx_dict['pos_idx']
    neg_inner_idx = idx_dict['neg_inner_idx']
    neg_outer_idx = idx_dict['neg_outer_idx']

    pair_attributes = {}

    # CHECK 1: Order of edges should be consistent

    pos_td = pos_edges[pos_idx][timediv]
    neg_inner_td, neg_outer_td = neg_edges[neg_inner_idx][timediv], neg_edges[neg_outer_idx][timediv]
    td_length = vpp_local_config.get('total_timediv_count')

    # Check p1-p2<p1-p3<p1-p4

    are_edge_td_ordered = ((neg_inner_td - pos_td) % td_length < (neg_outer_td - pos_td) % td_length)

    if not are_edge_td_ordered:
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    # CHECK 2: Time overlap, Total overlap should be more than 1 day

    pos_days = np.arange(pos_edges[pos_idx][start_day], pos_edges[pos_idx][end_day])

    neg_inner_days = np.arange(neg_edges[neg_inner_idx][start_day], neg_edges[neg_inner_idx][end_day])
    neg_outer_days = np.arange(neg_edges[neg_outer_idx][start_day], neg_edges[neg_outer_idx][end_day])

    intersection_days = reduce(np.intersect1d, (pos_days, neg_inner_days, neg_outer_days))
    if len(intersection_days) < vpp_local_config.get('min_intersection_days'):
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    bool_val, vpairs_dict, df_days_filled = stype_pnn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict,
                                                               df_days_filled, pair_attributes, day_col, is_filled_col,
                                                               vpp_local_config)

    return bool_val, vpairs_dict, df_days_filled


def stype_pnn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                     vpp_local_config):
    """Finds vpairs for PNN type"""

    neg_edge_index_pairs = input_dict['neg_edge_index_pairs']

    for pos_idx in range(len(pos_edges)):
        for neg_inner_idx, neg_outer_idx in neg_edge_index_pairs:

            idx_dict = dict()
            idx_dict['pos_idx'] = pos_idx
            idx_dict['neg_inner_idx'] = neg_inner_idx
            idx_dict['neg_outer_idx'] = neg_outer_idx
            bool_val, vpairs_dict, df_days_filled = stype_pnn_checks(input_dict, idx_dict, pos_edges, neg_edges,
                                                                     vpairs_dict, df_days_filled, day_col,
                                                                     is_filled_col, vpp_local_config)
            if not bool_val:
                continue

    return vpairs_dict, df_days_filled


def stype_pn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, pair_attributes,
                      day_col, is_filled_col, vpp_local_config):
    """
    Utility for stype_pn_checks, written to reduce number of return statements for fixing Sonar Issue
    """
    timediv = input_dict['timediv']
    start_day = input_dict['start_day']
    end_day = input_dict['end_day']
    stype_pn = input_dict['stype_pn']
    type_val = input_dict['type_val']
    raw_data = input_dict['raw_data']

    pos_idx = idx_dict['pos_idx']
    neg_idx = idx_dict['neg_idx']

    bool_val = True

    # CHECK 4: Duration Ballpark, 1 to 18 hrs(when developed) between last p1 and first p2 edge

    vpair_duration_in_td = (neg_edges[neg_idx][timediv] - pos_edges[pos_idx][
        timediv]) % vpp_local_config.get('total_timediv_count')

    vpair_duration_in_hour = vpair_duration_in_td / vpp_local_config.get('samples_per_hour')

    if ((vpair_duration_in_hour < vpp_local_config.get('vpair_min_duration_in_hour')) | (
            vpair_duration_in_hour > vpp_local_config.get('vpair_max_duration_in_hour'))):
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    pair_attributes['duration'] = vpair_duration_in_td

    # CHECK 5: Signal Consistency For inner and outer pairs

    # Signal Consistency for Inner Pair

    pos_days = np.arange(pos_edges[pos_idx][start_day], pos_edges[pos_idx][end_day])
    neg_days = np.arange(neg_edges[neg_idx][start_day], neg_edges[neg_idx][end_day])
    intersection_days = reduce(np.intersect1d, (pos_days, neg_days))
    common_start_day, common_end_day = intersection_days.min(), intersection_days.max()
    pos_overlapped_edge = pos_edges[pos_idx].copy()
    pos_overlapped_edge[start_day], pos_overlapped_edge[
        end_day] = common_start_day, common_end_day

    neg_overlapped_edge = neg_edges[neg_idx].copy()
    neg_overlapped_edge[start_day], neg_overlapped_edge[
        end_day] = common_start_day, common_end_day

    inner_day_fraction, inner_window_fraction = check_signal_consistency(vpp_local_config,
                                                                         pos_overlapped_edge,
                                                                         neg_overlapped_edge, raw_data)
    # Signal Consistency for Outer Pair(As this is PN Pair, Both Inner and Outer are same pairs)

    outer_day_fraction, outer_window_fraction = inner_day_fraction, inner_window_fraction

    if ((inner_day_fraction < vpp_local_config.get('vpair_min_day_fraction')) | (
            outer_day_fraction < vpp_local_config.get('vpair_min_day_fraction'))):
        bool_val = False

    if bool_val:
        pair_attributes['inner_day_fraction'], pair_attributes[
            'outer_day_fraction'] = inner_day_fraction, outer_day_fraction

        if ((inner_window_fraction < vpp_local_config.get('vpair_min_window_fraction')) | (
                outer_window_fraction < vpp_local_config.get('vpair_min_window_fraction'))):
            bool_val = False

    if not bool_val:
        return bool_val, vpairs_dict, df_days_filled

    pair_attributes['inner_window_fraction'], pair_attributes[
        'outer_window_fraction'] = inner_window_fraction, outer_window_fraction

    vpair_key = "%d,%d,%d,%d,%d" % (common_start_day, common_end_day, pos_edges[pos_idx][timediv],
                                    neg_edges[neg_idx][timediv], type_val)

    pair_attributes['type'] = stype_pn
    pair_attributes['pos_edges'] = [pos_overlapped_edge]
    pair_attributes['neg_edges'] = [neg_overlapped_edge]

    if vpair_key in vpairs_dict.keys():
        vpair_key = vpair_key + ',_r'

    vpairs_dict[vpair_key] = pair_attributes

    df_days_filled[np.isin(df_days_filled[:, day_col], intersection_days), is_filled_col] = 1

    return bool_val, vpairs_dict, df_days_filled


def stype_pn_checks(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                    vpp_local_config):
    """Checks for inner and outer edges of each PN structure"""

    start_day = input_dict['start_day']
    end_day = input_dict['end_day']
    amplitude = input_dict['amplitude']
    stype_pn = input_dict['stype_pn']
    type_val = input_dict['type_val']

    pos_idx = idx_dict['pos_idx']
    neg_idx = idx_dict['neg_idx']

    pair_attributes = {}

    # CHECK 1: Order of edges should be consistent(Obselete for PN Pair)

    # CHECK 2: Time overlap, Total overlap should be more than 1 day

    pos_days = np.arange(pos_edges[pos_idx][start_day], pos_edges[pos_idx][end_day])
    neg_days = np.arange(neg_edges[neg_idx][start_day], neg_edges[neg_idx][end_day])

    intersection_days = reduce(np.intersect1d, (pos_days, neg_days))

    if len(intersection_days) < vpp_local_config.get('min_intersection_days'):
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    common_start_day, common_end_day = intersection_days.min(), intersection_days.max()
    pair_attributes['common_start_day'] = common_start_day
    pair_attributes['common_end_day'] = common_end_day

    # CHECK 3: Amplitude Ballpark, (P1 + P2) should be nearly equal to (N1+N2)

    pos_amplitude_sum = pos_edges[pos_idx][amplitude]
    neg_amplitude_sum = neg_edges[neg_idx][amplitude]

    abs_neg_amplitude_sum = np.absolute(neg_amplitude_sum)

    amplitude_ratio = min(abs_neg_amplitude_sum, pos_amplitude_sum) / max(abs_neg_amplitude_sum,
                                                                          pos_amplitude_sum)

    allow_amp_ratio_for_pn = vpp_local_config.get('min_amplitude_ratio_threshold_high') / 2

    if amplitude_ratio < vpp_local_config.get('min_amplitude_ratio_threshold_high') and (type_val == stype_pn) and (
            amplitude_ratio < allow_amp_ratio_for_pn):
        bool_val = False
        return bool_val, vpairs_dict, df_days_filled

    pair_attributes['amplitude_ratio'] = amplitude_ratio

    bool_val, vpairs_dict, df_days_filled = stype_pn_checks_1(input_dict, idx_dict, pos_edges, neg_edges, vpairs_dict,
                                                              df_days_filled, pair_attributes, day_col, is_filled_col,
                                                              vpp_local_config)
    return bool_val, vpairs_dict, df_days_filled


def stype_pn_vpairs(input_dict, pos_edges, neg_edges, vpairs_dict, df_days_filled, day_col, is_filled_col,
                    vpp_local_config):
    """Finds vpairs for PN type"""

    for pos_idx in range(len(pos_edges)):
        for neg_idx in range(len(neg_edges)):

            idx_dict = dict()
            idx_dict['pos_idx'] = pos_idx
            idx_dict['neg_idx'] = neg_idx
            bool_val, vpairs_dict, df_days_filled = stype_pn_checks(input_dict, idx_dict, pos_edges, neg_edges,
                                                                    vpairs_dict, df_days_filled, day_col, is_filled_col,
                                                                    vpp_local_config)
            if not bool_val:
                continue

    return vpairs_dict, df_days_filled
