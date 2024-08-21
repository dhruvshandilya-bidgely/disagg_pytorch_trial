"""
Author - Sahana M
Date - 20/07/2021
This file contains all the functions to derive weather related features
"""


# Import python packages
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import seq_arr_idx


def get_reverse_seasonality_score(seq_arr, wh_config):
    """
    This function is used to get the reverse seasonality score which denotes whether the data is anti-water heater usage
    behaviour - higher usage in summer, lower usage in winters.
    Parameters:
        seq_arr             (np.ndarray)        : Contains info on each season detected
        wh_config           (dict)              : WH configurations dictionary
    Returns:
        reverse_seasonality_score (float)       : reverse seasonality score
    """

    summer_days = seq_arr[seq_arr[:, seq_arr_idx['s_label']] == 1, :]
    winter_days = seq_arr[seq_arr[:, seq_arr_idx['s_label']] == -1, :]

    # Only if there are both winter & summer days calculate the score

    if (len(summer_days) and len(winter_days)) and (np.sum(summer_days[:, seq_arr_idx['auc']]) > 0 and
                                                    np.sum(winter_days[:, seq_arr_idx['auc']]) > 0):

        summer_auc = (np.nanmean(summer_days[:, seq_arr_idx['auc']]) / np.nanmax(seq_arr[:, seq_arr_idx['auc']]))
        summer_dur = (np.nanmean(summer_days[:, seq_arr_idx['dur']]) / np.nanmax(seq_arr[:, seq_arr_idx['dur']]))
        summer_amp = (np.nanmean(summer_days[:, seq_arr_idx['amp']]) / np.nanmax(seq_arr[:, seq_arr_idx['amp']]))

        winter_auc = (np.nanmean(winter_days[:, seq_arr_idx['auc']]) / np.nanmax(seq_arr[:, seq_arr_idx['auc']]))
        winter_dur = (np.nanmean(winter_days[:, seq_arr_idx['dur']]) / np.nanmax(seq_arr[:, seq_arr_idx['dur']]))
        winter_amp = (np.nanmean(winter_days[:, seq_arr_idx['amp']]) / np.nanmax(seq_arr[:, seq_arr_idx['amp']]))

        auc_diff = winter_auc - summer_auc
        dur_diff = winter_dur - summer_dur
        amp_diff = winter_amp - summer_amp

        reverse_seasonality_score = max(((auc_diff + dur_diff + amp_diff) / 3), -1)

    else:
        reverse_seasonality_score = wh_config['default_rss']

    reverse_seasonality_score = (reverse_seasonality_score + 1) / 2

    return reverse_seasonality_score


def get_one_sided_seasonality_score(seq_arr, wh_config):
    """
    This function is used to get the one sided seasonality score i.e., high winter consumption in one winter season &
    low consumption in another winter season
    Parameters:
        seq_arr                     (np.ndarray)        : Contains info on each season detected
        wh_config                   (dict)              : WH configurations dictionary

    Returns:
        one_sided_seasonality_score (float)             : one sided seasonality score
    """

    winter_days = seq_arr[(seq_arr[:, seq_arr_idx['s_label']] == -1), :]

    # Only if there are more than 2 winters present then calculate the score

    if len(winter_days) > 1 and np.sum(winter_days[:, seq_arr_idx['auc']]) > 0:

        wtr_auc = (winter_days[:, seq_arr_idx['auc']] / np.nanmax(winter_days[:, seq_arr_idx['auc']]))
        wtr_dur = (winter_days[:, seq_arr_idx['dur']] / np.nanmax(winter_days[:, seq_arr_idx['dur']]))
        wtr_amp = (winter_days[:, seq_arr_idx['amp']] / np.nanmax(winter_days[:, seq_arr_idx['amp']]))

        diff_auc = max(wtr_auc) - min(wtr_auc)
        diff_dur = max(wtr_dur) - min(wtr_dur)
        diff_amp = max(wtr_amp) - min(wtr_amp)

        one_sided_seasonality_score = 1 - max(diff_auc, diff_dur, diff_amp)

    else:
        one_sided_seasonality_score = wh_config['default_oss']

    return one_sided_seasonality_score


def wtr_before_smr_dip(seq_arr, start_idx, end_idx, double_dip_scores, wh_config):
    """
    This function is used to calculate double dip score for winter-transition-summer sequences
    Parameters:
        seq_arr                     (np.ndarray)        : Contains info on each season detected
        start_idx                   (int)               : start index in seq arr
        end_idx                     (int)               : end index in seq arr
        double_dip_scores           (list)              : double dip scores
        wh_config                   (dict)              : WH configurations dictionary

    Returns:
        double_dip_scores           (list)              : double dip scores
    """

    double_dip_score = wh_config['default_dds']

    if seq_arr[(start_idx - 1), seq_arr_idx['s_label']] == -1 and seq_arr[(end_idx + 1), seq_arr_idx['s_label']] == 1:
        wtr_auc = seq_arr[(start_idx - 1), seq_arr_idx['auc']]
        wtr_dur = seq_arr[(start_idx - 1), seq_arr_idx['dur']]
        wtr_amp = seq_arr[(start_idx - 1), seq_arr_idx['amp']]

        smr_auc = seq_arr[(end_idx + 1), seq_arr_idx['auc']]
        smr_dur = seq_arr[(end_idx + 1), seq_arr_idx['dur']]
        smr_amp = seq_arr[(end_idx + 1), seq_arr_idx['amp']]
        if wtr_auc > 0:
            smr_auc = smr_auc / wtr_auc
            smr_dur = smr_dur / wtr_dur
            smr_amp = smr_amp / wtr_amp

            auc_diff = 1 - smr_auc
            dur_diff = 1 - smr_dur
            amp_diff = 1 - smr_amp

            double_dip_score = max(((auc_diff + dur_diff + amp_diff) / 3), -1)

        elif wtr_auc == 0 and smr_auc == 0:
            double_dip_score = wh_config['default_dds']
        elif wtr_auc == 0 and smr_auc > 0:
            double_dip_score = -1

        double_dip_scores.append(double_dip_score)

    return double_dip_scores


def smr_before_wtr_dip(seq_arr, start_idx, end_idx, double_dip_scores, wh_config):
    """
    This function is used to calculate double dip score for summer-transition-winter sequences
    Parameters:
        seq_arr                     (np.ndarray)        : Contains info on each season detected
        start_idx                   (int)               : start index in seq arr
        end_idx                     (int)               : end index in seq arr
        double_dip_scores           (list)              : double dip scores
        wh_config                   (dict)              : wh configuration dictionary

    Returns:
        double_dip_scores           (list)              : double dip scores
    """

    if seq_arr[(start_idx - 1), seq_arr_idx['s_label']] == 1 and seq_arr[(start_idx + 1), seq_arr_idx['s_label']] == -1:
        wtr_auc = seq_arr[(end_idx + 1), seq_arr_idx['auc']]
        wtr_dur = seq_arr[(end_idx + 1), seq_arr_idx['dur']]
        wtr_amp = seq_arr[(end_idx + 1), seq_arr_idx['amp']]

        smr_auc = seq_arr[(start_idx - 1), seq_arr_idx['auc']]
        smr_dur = seq_arr[(start_idx - 1), seq_arr_idx['dur']]
        smr_amp = seq_arr[(start_idx - 1), seq_arr_idx['amp']]

        if wtr_auc > 0:
            smr_auc = smr_auc / wtr_auc
            smr_dur = smr_dur / wtr_dur
            smr_amp = smr_amp / wtr_amp

            auc_diff = 1 - smr_auc
            dur_diff = 1 - smr_dur
            amp_diff = 1 - smr_amp

            double_dip_scores.append(max(((auc_diff + dur_diff + amp_diff) / 3), -1))
        elif wtr_auc == 0 and smr_auc == 0:
            double_dip_scores.append(wh_config['default_dds'])
        elif wtr_auc == 0 and smr_auc > 0:
            double_dip_scores.append(-1)

    return double_dip_scores


def get_start_end(seq_arr, i):
    """
    Get the start and end after a certain index
    Parameters:
        seq_arr             (np.ndarray)        : Contains info on each season detected
        i                   (int)               : From index to look into
    Returns:
        start_idx           (int)               : Start index
        end_idx             (int)               : End index
    """

    # Get the start and end index of the sequence from ith index array

    start_idx = i
    end_idx = i
    for j in range(i, len(seq_arr)):
        if np.isin(seq_arr[j, seq_arr_idx['s_label']], [-0.5, 0, 0.5]):
            end_idx = j
        else:
            break

    return start_idx, end_idx


def get_double_dip_score(seq_arr, wh_config):
    """
    This function is used to identify double dips in consumption in the data (transition moving towards summer / winter)
    Parameters:
        seq_arr                     (np.ndarray)        : Contains info on each season detected
        wh_config                   (dict)              : WH configurations dictionary

    Returns:
        double_dip_score            (float)             : Double dip score
    """

    double_dip_scores = []
    i = 0

    double_dip_score = wh_config['default_dds']

    # Identify if there are any transition seasons present

    if np.sum(np.isin(seq_arr[:, seq_arr_idx['s_label']], [-0.5, 0, 0.5])) > 0:
        while i < len(seq_arr):
            if np.isin(seq_arr[i, seq_arr_idx['s_label']], [-0.5, 0, 0.5]):

                start_idx, end_idx = get_start_end(seq_arr, i)

                if (start_idx - 1) >= 0 and (end_idx + 1) < len(seq_arr):

                    # Check if the dip has more consumption on the winter side
                    double_dip_scores = wtr_before_smr_dip(seq_arr, start_idx, end_idx, double_dip_scores, wh_config)

                    # Check if the dip has more consumption on the summer side
                    double_dip_scores = smr_before_wtr_dip(seq_arr, start_idx, end_idx, double_dip_scores, wh_config)

                else:
                    double_dip_scores.append(wh_config['default_dds'])

                i = end_idx+1
            else:
                i += 1

        if len(double_dip_scores):
            double_dip_score = np.mean(double_dip_scores)

    double_dip_score = (double_dip_score + 1) / 2

    return double_dip_score
