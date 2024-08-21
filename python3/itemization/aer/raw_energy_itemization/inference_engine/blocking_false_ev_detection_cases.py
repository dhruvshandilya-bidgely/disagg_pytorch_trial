

"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update ev consumption ranges using inference rules
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def eliminate_seasonal_ev_cases(item_input_object, vacation, valid_idx, samples_per_hour, season, ev_disagg,
                                ev_residual, logger):
    """
    remove ev detection (step 2) in cases where EV output aligns with particular season

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        vacation                  (np.ndarray)    : day wise vacation tag
        valid_idx                 (np.ndarray)    : ts level ev boxes detected in hybrid v2
        samples_per_hour          (int)           : samples in an hour
        season                    (np.ndarray)    : day wise season tag
        ev_disagg                 (np.ndarray)    : ev disagg data
        ev_residual               (np.ndarray)    : ts level ev estimation in hybrid v2
        logger                    (logger)        : logger object

    Returns:
        ev_residual               (np.ndarray)    : ts level ev estimation in hybrid v2
    """

    ev_config = get_inf_config(samples_per_hour).get('ev')

    ev_days = (np.sum(valid_idx, axis=1) > 0).astype(int)

    # checking whether user detected from hybrid is recent ev

    high_ev_usage_hours = ev_config.get('high_ev_usage_hours')

    recent_days_segment = max(60, int(len(ev_days) * 0.6))

    recent_days_segment_2 = min(90, recent_days_segment - (len(ev_days) - recent_days_segment) / 2)

    recency = ((ev_days[-recent_days_segment:].sum() / ev_days.sum()) > 0.9)

    high_cons_during_last_months = ((ev_days[-int(recent_days_segment_2):].sum() / ev_days[-recent_days_segment:].sum()) > 0.3)

    if ev_days[-recent_days_segment:].sum() > 0:
        recency = recency and high_cons_during_last_months

    season = season[np.logical_not(vacation)]

    min_season_days_required = 20

    season_data_present = (not np.all(season == 0)) and (not np.all(season < 0)) and (not np.all(season > 0))

    perform_seasonal_ev_check = ((not recency) and season_data_present) and np.sum(valid_idx) and (np.sum(ev_disagg) == 0)

    if perform_seasonal_ev_check:

        cons = np.zeros(3)
        cons[0] = np.sum(valid_idx[np.logical_not(vacation)][season > 0.5][:, high_ev_usage_hours]) / np.sum(season > 0.5)
        cons[1] = np.sum(valid_idx[np.logical_not(vacation)][season == 0][:, high_ev_usage_hours]) / np.sum(season == 0)
        cons[2] = np.sum(valid_idx[np.logical_not(vacation)][season < -0.5][:, high_ev_usage_hours]) / np.sum(season < -0.5)

        cons = np.nan_to_num(cons)

        seasonality_in_ev_output = (cons[0] == 0 and (np.sum(season > 0.5) > min_season_days_required)) or\
                                   (cons[2] == 0 and (np.sum(season < -0.5) > min_season_days_required))

        # blocked newly detected ev because either of winter or summer output is 0

        if seasonality_in_ev_output:
            valid_idx[:] = 0
            ev_residual[:] = 0
            logger.info('blocked newly detected ev because either of winter or summer output is 0 | ')

        ev_residual = blocking_seasonal_ev_cases(ev_residual, logger, cons, season)

    # blocking seasonal based on seasonality at day level

    if not perform_seasonal_ev_check:
        return ev_residual

    valid_idx = np.sum(valid_idx > 0, axis=1) > 0

    # blocked newly detected ev because either of winter or summer output is 0

    cons = np.zeros(3)
    cons[0] = np.sum(valid_idx[np.logical_not(vacation)][season > 0.5]) / np.sum(season > 0.5)
    cons[1] = np.sum(valid_idx[np.logical_not(vacation)][season == 0]) / np.sum(season == 0)
    cons[2] = np.sum(valid_idx[np.logical_not(vacation)][season < -0.5]) / np.sum(season < -0.5)

    cons = np.nan_to_num(cons)

    seasonality_in_ev_output = (cons[0:2].sum() == 0 and (np.sum(season > 0.5) > min_season_days_required)) or \
                               (cons[1:3].sum() == 0 and (np.sum(season < -0.5) > min_season_days_required))

    if seasonality_in_ev_output:
        valid_idx[:] = 0
        ev_residual[:] = 0
        logger.info('blocked newly detected ev because either of winter or summer output is 0 | ')

        ev_residual = blocking_seasonal_ev_cases(ev_residual, logger, cons, season)

    return ev_residual


def blocking_seasonal_ev_cases(ev_residual, logger, seasonal_cons, season):

    """
    remove ev detection (step 2) in cases where EV output aligns with particular season

    Parameters:
        ev_residual               (np.ndarray)    : ts level ev estimation in hybrid v2
        logger                    (logger)        : logger object
        seasonal_cons             (np.ndarray)    : season wise consumption
        season                    (np.ndarray)    : day wise season tag
    Returns:
        ev_residual               (np.ndarray)    : ts level ev estimation in hybrid v2

    """

    min_season_days_required = 20

    ev_config = get_inf_config(int(ev_residual.shape[1]/Cgbdisagg.HRS_IN_DAY)).get('ev')

    high_summer_cons = (np.sum(season > 0.5) > min_season_days_required) and \
                       (np.sum(season < -0.5) > min_season_days_required) and \
                       np.sum(seasonal_cons[0]) > ev_config.get('seasons_comparison_thres') * np.sum(seasonal_cons[1:])

    high_winter_cons = (np.sum(season > 0.5) > min_season_days_required) and \
                       (np.sum(season < -0.5) > min_season_days_required) and \
                       np.sum(seasonal_cons[2]) > ev_config.get('seasons_comparison_thres') * np.sum(seasonal_cons[:2])

    low_trans_cons = np.sum(season == 0) > 40 and np.sum(season > 0) > min_season_days_required and \
                     np.sum(season < 0) > min_season_days_required and seasonal_cons[1] < ((seasonal_cons[0] + seasonal_cons[2]) / 20)

    # blocked newly detected ev because tranistion output is less

    if low_trans_cons:
        ev_residual[:] = 0
        logger.info('blocked newly detected ev because tranistion output is less | ')

    # blocked newly detected ev because summer output is high

    elif high_summer_cons:
        ev_residual[:] = 0
        logger.info('blocked newly detected ev because summer output is high | ')

    # blocked newly detected ev because winter output is high

    elif high_winter_cons:
        ev_residual[:] = 0
        logger.info('blocked newly detected ev because winter output is high | ')

    return ev_residual


def block_user_with_feeble_ev_boxes(item_input_object, ev_days):

    """
    blocks ev consumption in cases where feeble ev boxes are detected

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        ev_days                   (np.ndarray)    : days where ev is added from hybrid v2
    Returns:
        ev_days                   (np.ndarray)    : days where ev is added from hybrid v2
    """

    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

    original_days = copy.deepcopy(ev_days)

    ev_days[ev_days > 0] = 1

    days_in_month = Cgbdisagg.DAYS_IN_MONTH

    # for each sequence of days, checking the amount of ev boxes present

    for i in range(0, len(ev_days), days_in_month):

        factor = 1 - (vacation[i:i + days_in_month]).sum()/days_in_month

        low_ev_boxes = np.sum(ev_days[i:i + days_in_month]) < 5 * factor

        if low_ev_boxes:
            ev_days[i:i + days_in_month] = 2

    ev_days_seq = find_seq(ev_days, np.zeros_like(ev_days), np.zeros_like(ev_days), overnight=0)

    ev_days_seq = ev_days_seq.astype(int)

    # if feeble ev consumption is present of continuous 5 months,
    # whole ev consumption is removed in that segment

    for i in range(len(ev_days_seq)):
        ev_days = block_inconsistent_ev_seq(ev_days, ev_days_seq, 5*days_in_month, i)

    ev_days[original_days == 0] = 0
    ev_days[ev_days == 2] = 0

    return ev_days


def block_inconsistent_ev_seq(ev_days, ev_days_seq, thres, seq_idx):

    """
    block ev consumption (added from hybrid v2) in cases where less ev boxes is present

    Parameters:
        ev_days                   (np.ndarray)    : days where ev is added from hybrid v2
        ev_days_seq               (np.ndarray)    : sequence of days where ev is added from hybrid v2
        thres                     (int)           : if number of disconnected ev months is less than threshold, current ev seq will be blocked
        seq_idx                   (int)           : index of current ev seq

    Returns:
        ev_days                   (np.ndarray)    : days where ev is added from hybrid v2
    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    if ev_days_seq[seq_idx, seq_label] == 2 and len(ev_days_seq) > 1:

        fill_inconsistent_ev_window = (seq_idx == 0 and (ev_days_seq[seq_idx + 1, seq_label] == 1 and ev_days_seq[seq_idx, seq_len] <= thres))

        fill_inconsistent_ev_window = fill_inconsistent_ev_window or \
                                      ((seq_idx == (len(ev_days_seq) - 1)) and
                                       (ev_days_seq[seq_idx - 1, seq_label] == 1 and ev_days_seq[seq_idx, seq_len] <= thres))

        fill_inconsistent_ev_window = fill_inconsistent_ev_window or \
                                      (seq_idx not in [0, len(ev_days_seq) - 1] and
                                       ((ev_days_seq[seq_idx + 1, seq_label] == 1) or
                                        (ev_days_seq[seq_idx - 1, seq_label] == 1)) and ev_days_seq[seq_idx, seq_len] <= thres)

        if fill_inconsistent_ev_window:
            ev_days[ev_days_seq[seq_idx, seq_start]: ev_days_seq[seq_idx, seq_end] + 1] = 1

    return ev_days


def block_ev_cons_in_bc_with_less_boxes(item_input_object, vacation, added_ev_cons, bc_list, unique_bc, ev_config, disagg_cons):

    """
    block ev consumption (added from hybrid v2) in cases where less ev boxes is present

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        vacation                  (np.ndarray)    : day wise vacation tag
        added_ev_cons             (np.ndarray)    : ts level ev estimation in hybrid v2
        bc_list                   (np.ndarray)    : day wise billing cycle value
        unique_bc                 (np.ndarray)    : list of unique billing cycles
        ev_config                 (dict)          : ev config
        disagg_cons               (np.ndarray)    : ev disagg output

    Returns:
        added_ev_cons             (np.ndarray)    : updated ts level ev estimation in hybrid v2
    """

    seq_label = seq_config.SEQ_LABEL

    if item_input_object.get("config").get('disagg_mode') == 'mtd':
        return added_ev_cons

    ev_cons_1d = ((added_ev_cons + disagg_cons) > 0).flatten()

    seq = find_seq(ev_cons_1d, np.zeros_like(ev_cons_1d), np.zeros_like(ev_cons_1d))

    if np.sum(seq[:, 0] > 0) < 8:
        added_ev_cons[:, :] = 0

    blocking_list = np.ones_like(unique_bc)

    # for each billing cycle, checking the amount of ev boxes present
    # and deciding whether sufficient EV boxes were present in the given billing cycle

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        ev_cons_1d = ((added_ev_cons + disagg_cons)[target_days] > 0).flatten()

        if np.sum(ev_cons_1d) == 0:
            blocking_list[i] = 0
            continue

        factor = 1 - (vacation[target_days]).sum()/np.sum(target_days)

        seq = find_seq(ev_cons_1d, np.zeros_like(ev_cons_1d), np.zeros_like(ev_cons_1d))

        low_ev_boxes = (np.sum(seq[:, 0] > 0) < ev_config.get('bc_min_box_count') * factor) and np.sum(disagg_cons[target_days]) == 0

        if low_ev_boxes:
            blocking_list[i] = 2

    # Either all billing cycles have low ev consumption or 0 ev consumption

    if (np.sum(blocking_list == 2) == 0) or np.all(blocking_list == 0):
        return added_ev_cons

    thres = 5

    ev_days_seq = find_seq(blocking_list, np.zeros_like(blocking_list), np.zeros_like(blocking_list), overnight=0)

    ev_days_seq = ev_days_seq.astype(int)

    # if feeble ev consumption is present of continuous 5 billing cycles,
    # whole ev consumption is removed in that segment

    for i in range(len(ev_days_seq)):
        blocking_list = block_inconsistent_ev_seq(blocking_list, ev_days_seq, thres, i)

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if blocking_list[i] == 2:
            added_ev_cons[target_days] = 0

        ev_cons_1d = ((added_ev_cons + disagg_cons)[target_days] > 0).flatten()

        seq = find_seq(ev_cons_1d, np.zeros_like(ev_cons_1d), np.zeros_like(ev_cons_1d))

        low_ev_boxes = (np.sum(seq[:, seq_label] > 0) < 2) and np.sum(disagg_cons[target_days]) == 0

        if low_ev_boxes:
            added_ev_cons[target_days] = 0

    return added_ev_cons
