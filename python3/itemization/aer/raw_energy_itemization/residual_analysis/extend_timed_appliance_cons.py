
"""
Author - Nisha Agarwal
Date - 17th Feb 2021
Detect timed appliance in residual data
"""

# Import python packages

import copy
import logging
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config


def extend_cons_for_detected_twh_sig(item_input_object, twh_cons, twh_cons_disagg, input_data, samples_per_hour):

    """
    extends twh consumption to avoid wh underestimation in wh tou region

    Parameters:
        twh_cons                (np.ndarray)    : twh residual output
        twh_cons_disagg         (np.ndarray)    : twh disagg output
        input_data              (np.ndarray)    : input data
        samples_per_hour        (int)           : samples in an hour

    Returns:
        twh_cons                (np.ndarray)    : updated twh residual output

    """

    config = get_residual_config(samples_per_hour).get('timed_app_det_config')

    min_twh_ts_level_cons_limit = config.get('min_twh_ts_level_cons_limit')
    twh_ts_level_cons_buffer = config.get('twh_ts_level_cons_buffer')
    min_twh_seq_len_for_extension = config.get('min_twh_seq_len_for_extension')
    min_twh_seq_frac_days_for_extension = config.get('min_twh_seq_frac_days_for_extension')
    perc_cap = config.get('perc_cap')
    min_twh_dur = config.get('min_twh_dur')
    min_twh_days_seq = config.get('min_twh_days_seq')
    max_non_timed_days = config.get('max_non_timed_days')

    if (np.sum(twh_cons) > 0):
        twh_residual = input_data - twh_cons
        twh_amp = np.percentile(twh_cons[twh_cons > 0], perc_cap)
        min_cons = max(min_twh_ts_level_cons_limit/samples_per_hour, twh_amp - twh_ts_level_cons_buffer/samples_per_hour)
        max_cons = twh_amp + twh_ts_level_cons_buffer/samples_per_hour

        # get the possible twh region based on amplitude values

        # checking the timestamps where additional TWH can be added based on consumption level and time of day

        posible_region = np.logical_and(twh_residual > min_cons, twh_residual < max_cons)

        posible_region[:, np.sum(twh_cons, axis=0) == 0] = 0

        timed_app_days = np.sum(posible_region+(twh_cons>0), axis=1) > 0

        thres = min(min_twh_days_seq, max(min_twh_seq_len_for_extension, min_twh_seq_frac_days_for_extension*len(input_data)))

        timed_app_days = remove_less_days_timed_signature(timed_app_days, max_non_timed_days, thres)

        posible_region[timed_app_days == 0] = 0

        length = np.sum((posible_region + (twh_cons>0)) > 0, axis=1)

        avg_duration = np.mean(length[length > 0])

        posible_region[np.sum((posible_region + (twh_cons>0)) > 0, axis=1) < avg_duration/2] = 0

        posible_region[twh_cons > 0] = 1

        posible_region = remove_low_duration_points(posible_region, min_twh_dur, twh_cons.shape)

        timed_app_days = np.sum(posible_region+(twh_cons>0), axis=1) > 0

        # post processing of derived twh region, based on the continuity

        thres = min(min_twh_days_seq, max(min_twh_seq_len_for_extension, min_twh_seq_frac_days_for_extension*len(input_data)))

        timed_app_days = remove_less_days_timed_signature(timed_app_days, max_non_timed_days, thres)

        posible_region[timed_app_days == 0] = 0

        posible_region = posible_region * twh_amp

        posible_region[twh_cons > 0] = 0

        twh_cons = twh_cons + posible_region

    return twh_cons


def extend_twh_cons(item_input_object, twh_cons, twh_cons_disagg, timed_sig, input_data, samples_per_hour):

    """
    extends twh consumption to avoid wh underestimation in wh tou region

    Parameters:
        item_input_object       (dict)          : dict containing all hybrid inputs
        twh_cons                (np.ndarray)    : twh residual output
        twh_cons_disagg         (np.ndarray)    : twh disagg output
        timed_sig               (np.ndarray)    : output of timed signature detection
        input_data              (np.ndarray)    : input data
        samples_per_hour        (int)           : samples in an hour

    Returns:
        twh_cons                (np.ndarray)    : updated twh residual output

    """
    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    config = get_residual_config(samples_per_hour).get('timed_app_det_config')

    min_twh_ts_level_cons_limit = config.get('min_twh_ts_level_cons_limit')
    twh_ts_level_cons_buffer = config.get('twh_ts_level_cons_buffer')

    min_twh_seq_len_for_disagg_extension = config.get('min_twh_seq_len_for_disagg_extension')
    min_twh_seq_frac_days_for_disagg_extension = config.get('min_twh_seq_frac_days_for_disagg_extension')

    twh_conf_level = config.get('twh_conf_level')
    min_timed_sig_days = config.get('min_timed_sig_days')
    consequetive_days_thres = config.get('consequetive_days_thres')
    perc_cap = config.get('perc_cap')

    total_samples = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    twh_det_conf = 1

    vali_twh_conf_present_flag = item_input_object.get("disagg_special_outputs") is not None and \
                                 item_input_object.get("disagg_special_outputs").get("timed_wh_confidence") is not None

    if vali_twh_conf_present_flag:
        twh_det_conf = copy.deepcopy(item_input_object.get("disagg_special_outputs").get("timed_wh_confidence"))

    if (np.sum(twh_cons) > 0) and twh_det_conf >= twh_conf_level[1]:
        twh_residual = input_data - twh_cons - timed_sig
        twh_amp = np.percentile(twh_cons[twh_cons > 0], perc_cap)
        min_cons = max(min_twh_ts_level_cons_limit/samples_per_hour, twh_amp - twh_ts_level_cons_buffer/samples_per_hour)
        max_cons = twh_amp + twh_ts_level_cons_buffer/samples_per_hour

        # get the possible twh region based on amplitude values

        posible_region = np.logical_and(twh_residual > min_cons, twh_residual < max_cons)

        twh_tou = np.sum(twh_cons > 0, axis=0) > 20

        timed_app_days_seq = find_seq(twh_tou, np.zeros_like(twh_tou), np.zeros_like(twh_tou), overnight=1)

        twh_tou = update_usage_seq_in_nearby_hours(timed_app_days_seq, twh_tou, samples_per_hour, total_samples)

        # filter possible twh region based on time of the day

        posible_region[:, twh_tou == 0] = 0

        timed_app_days = np.sum(posible_region + (twh_cons > 0), axis=1) > 0

        # post processing of derived twh region, based on the continuity

        thres = min(min_timed_sig_days, max(min_twh_seq_len_for_disagg_extension, min_twh_seq_frac_days_for_disagg_extension*len(input_data)))

        # remove less days timed signature sequence

        timed_app_days = remove_less_days_timed_signature(timed_app_days, 4, thres)

        posible_region[timed_app_days == 0] = 0

        length = np.sum((posible_region + (twh_cons > 0)) > 0, axis=1)

        avg_duration = np.mean(length[length > 0])

        posible_region[np.sum((posible_region + (twh_cons > 0)) > 0, axis=1) < avg_duration * 0.5] = 0

        # remove low duration points

        posible_region[twh_cons > 0] = 1

        posible_region = remove_low_duration_points(posible_region, max(samples_per_hour, 3), twh_cons.shape)

        posible_region[twh_cons > 0] = 1

        timed_app_days = np.sum(posible_region + (twh_cons > 0), axis=1) > 0

        # post processing of derived twh region, based on the continuity

        thres = min(min_timed_sig_days, max(min_twh_seq_len_for_disagg_extension, min_twh_seq_frac_days_for_disagg_extension*len(input_data)))

        timed_app_days = remove_less_days_timed_signature(timed_app_days, 4, thres)

        # remove inconsistent timed signature sequence

        timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days), overnight=0).astype(int)

        for i in range(1, len(timed_app_days_seq)-1):
            if timed_app_days_seq[i, seq_label] == 0 and \
                    timed_app_days_seq[i, seq_len] > consequetive_days_thres[0] and \
                    timed_app_days_seq[i+1, seq_len] < consequetive_days_thres[1]:
                timed_app_days[timed_app_days_seq[i+1, seq_start]:timed_app_days_seq[i+1, seq_end] + 1] = 0

        posible_region[timed_app_days == 0] = 0

        posible_region[twh_cons > 0] = 0

        posible_region = posible_region * twh_amp

        twh_cons = twh_cons + posible_region

    if (np.sum(twh_cons) > 0) and twh_det_conf >= twh_conf_level[0] and twh_det_conf < twh_conf_level[1]:
        twh_residual = input_data - twh_cons - timed_sig
        twh_amp = np.percentile(twh_cons[twh_cons > 0], perc_cap)
        min_cons = max(min_twh_ts_level_cons_limit/samples_per_hour, twh_amp - twh_ts_level_cons_buffer/samples_per_hour)
        max_cons = twh_amp + twh_ts_level_cons_buffer/samples_per_hour

        # get the possible twh region based on amplitude values

        posible_region = np.logical_and(twh_residual > min_cons, twh_residual < max_cons)

        twh_tou = np.sum(twh_cons > 0, axis=0) > 20

        # filter possible twh region based on time of the day

        posible_region[:, twh_tou == 0] = 0

        posible_region[twh_cons > 0] = 1

        timed_app_days = np.sum(posible_region + (twh_cons > 0), axis=1) > 0

        # post processing of derived twh region, based on the continuity

        thres = max(min_twh_seq_len_for_disagg_extension, 0.35*len(input_data))

        # remove less days timed signature sequence

        timed_app_days = remove_less_days_timed_signature(timed_app_days, 4, thres)

        posible_region[timed_app_days == 0] = 0

        length = np.sum((posible_region + (twh_cons > 0)) > 0, axis=1)

        avg_duration = np.mean(length[length > 0])

        posible_region[np.sum((posible_region + (twh_cons > 0)) > 0, axis=1) < avg_duration * 0.5] = 0

        posible_region[twh_cons > 0] = 1

        posible_region = remove_low_duration_points(posible_region, max(samples_per_hour*1, 3), twh_cons.shape)

        timed_app_days = np.sum(posible_region + (twh_cons > 0), axis=1) > 0

        # post processing of derived twh region, based on the continuity

        timed_app_days = remove_less_days_timed_signature(timed_app_days, 4, thres)

        timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days), overnight=0).astype(int)

        # remove inconsistent timed signature sequence

        for i in range(1, len(timed_app_days_seq)-1):
            if timed_app_days_seq[i, seq_label] == 0 and \
                    timed_app_days_seq[i, seq_len] > consequetive_days_thres[0] and \
                    timed_app_days_seq[i+1, seq_len] < consequetive_days_thres[1]:
                timed_app_days[timed_app_days_seq[i+1, seq_start]:timed_app_days_seq[i+1, seq_end] + 1] = 0

        posible_region[timed_app_days == 0] = 0

        posible_region[twh_cons > 0] = 0

        posible_region = posible_region * twh_amp

        twh_cons = twh_cons + posible_region

    twh_cons = extend_twh_cons_in_inc_run(item_input_object, twh_cons, twh_cons_disagg, timed_sig, input_data, samples_per_hour)

    return twh_cons


def extend_twh_cons_in_inc_run(item_input_object, twh_cons, twh_cons_disagg, timed_sig, input_data, samples_per_hour):

    """
    extends twh consumption to avoid wh underestimation in wh tou region

    Parameters:
        item_input_object       (dict)          : dict containing all hybrid inputs
        twh_cons                (np.ndarray)    : twh residual output
        twh_cons_disagg         (np.ndarray)    : twh disagg output
        timed_sig               (np.ndarray)    : output of timed signature detection
        input_data              (np.ndarray)    : input data
        samples_per_hour        (int)           : samples in an hour

    Returns:
        twh_cons                (np.ndarray)    : updated twh residual output

    """
    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    config = get_residual_config(samples_per_hour).get('timed_app_det_config')

    min_twh_ts_level_cons_limit = config.get('min_twh_ts_level_cons_limit')
    twh_ts_level_cons_buffer = config.get('twh_ts_level_cons_buffer')

    min_twh_seq_len_for_disagg_extension = config.get('min_twh_seq_len_for_disagg_extension')
    min_twh_seq_frac_days_for_disagg_extension = config.get('min_twh_seq_frac_days_for_disagg_extension')
    min_timed_sig_days = config.get('min_timed_sig_days')
    consequetive_days_thres = config.get('consequetive_days_thres')
    min_dur_for_inc_run_signature = config.get('min_dur_for_inc_run_signature')
    max_non_timed_days = config.get('max_non_timed_days')

    # fetching wh hsm data
    # this is added in order to handle cases of twh underestimation in incremental/mtd runs

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')
    wh_hsm_hld = 0
    wh_type = 0
    wh_tou = [0]
    wh_amp = 0

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'),
                                           'item_hld')
    # add twh if hsm wh type is timed

    if valid_hsm_flag:
        wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_hld', 0)

        if isinstance(wh_hsm, list):
            wh_hsm_hld = wh_hsm[0]
        else:
            wh_hsm_hld = wh_hsm

        wh_type = item_input_object.get("item_input_params").get('wh_hsm').get('item_type', 0)

        if isinstance(wh_type, list):
            wh_type = wh_type[0]

        wh_tou = item_input_object.get("item_input_params").get('wh_hsm').get('item_tou')

        if wh_tou is None:
            wh_tou = np.zeros(int(samples_per_hour*24))
        elif isinstance(wh_tou, list):
            wh_tou = np.array(wh_tou)

        wh_amp = item_input_object.get("item_input_params").get('wh_hsm').get('item_amp')

        if wh_amp is None:
            wh_amp = 0
        elif isinstance(wh_amp, list):
            wh_amp = wh_amp[0]

    if not ((item_input_object.get('config').get('disagg_mode') in ['incremental', 'mtd']) and (np.sum(twh_cons) == 0)
            and wh_hsm_hld > 0 and wh_type == 1 and wh_tou.sum() > 0 and wh_amp > 0):
        return twh_cons

    twh_residual = input_data - twh_cons
    min_cons = max(min_twh_ts_level_cons_limit/samples_per_hour, wh_amp/samples_per_hour - twh_ts_level_cons_buffer/samples_per_hour)
    max_cons = wh_amp/samples_per_hour + twh_ts_level_cons_buffer/samples_per_hour

    # get the possible twh region based on amplitude values

    posible_region = np.logical_and(twh_residual > min_cons, twh_residual < max_cons)

    posible_region[:, wh_tou == 0] = 0

    timed_app_days = np.sum(posible_region + (twh_cons > 0), axis=1) > 0

    # post processing of derived twh region, based on the continuity

    thres = min(min_timed_sig_days, max(min_twh_seq_len_for_disagg_extension, min_twh_seq_frac_days_for_disagg_extension*len(input_data)))

    timed_app_days = remove_less_days_timed_signature(timed_app_days, max_non_timed_days, thres)

    posible_region[timed_app_days == 0] = 0

    length = np.sum((posible_region + (twh_cons > 0)) > 0, axis=1)

    avg_duration = np.mean(length[length > 0])

    posible_region[np.sum((posible_region + (twh_cons > 0)) > 0, axis=1) < avg_duration * 0.5] = 0

    posible_region[twh_cons > 0] = 0

    posible_region = remove_low_duration_points(posible_region, min_dur_for_inc_run_signature, twh_cons.shape)

    timed_app_days = np.sum(posible_region + (twh_cons > 0), axis=1) > 0

    timed_app_days = remove_less_days_timed_signature(timed_app_days, max_non_timed_days, thres*0.5)

    timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days), overnight=0).astype(int)

    for i in range(1, len(timed_app_days_seq)-1):
        if timed_app_days_seq[i, seq_label] == 0 and \
                timed_app_days_seq[i, seq_len] > consequetive_days_thres[0] and \
                timed_app_days_seq[i+1, seq_len] < consequetive_days_thres[1]:
            timed_app_days[timed_app_days_seq[i+1, seq_start]:timed_app_days_seq[i+1, seq_end] + 1] = 0

    posible_region = posible_region * wh_amp/samples_per_hour

    posible_region[timed_app_days == 0] = 0

    twh_cons = twh_cons + posible_region

    return twh_cons


def extend_pp_cons(item_input_object, item_output_object, pp_cons, pp_cons_disagg, timed_sig, input_data, samples_per_hour):

    """
    extends pp consumption to avoid wh underestimation in pp tou region

    Parameters:
        item_input_object       (dict)          : dict containing all hybrid inputs
        item_output_object      (dict)          : dict containing all hybrid outputs
        pp_cons                 (np.ndarray)    : pp residual output
        pp_cons_disagg          (np.ndarray)    : pp disagg output
        timed_sig               (np.ndarray)    : output of timed signature detection
        input_data              (np.ndarray)    : input data
        samples_per_hour        (int)           : samples in an hour

    Returns:
        pp_cons                (np.ndarray)     : updated pp residual output

    """

    config = get_residual_config(samples_per_hour).get('timed_app_det_config')

    min_pp_ts_level_cons_limit = config.get('min_pp_ts_level_cons_limit')
    pp_ts_level_cons_buffer = config.get('pp_ts_level_cons_buffer')

    min_pp_seq_len_for_extension = config.get('min_pp_seq_len_for_extension')
    min_pp_seq_frac_days_for_extension = config.get('min_pp_seq_frac_days_for_extension')
    min_timed_seq = config.get('min_timed_seq')
    min_days_for_pp_cons_to_be_present = config.get('min_days_for_pp_cons_to_be_present')
    conf_thres_for_extension = config.get('conf_thres_for_extension')
    hours_thres_for_extension = config.get('hours_thres_for_extension')
    min_days_for_winter_pp_cons = config.get('min_days_for_winter_pp_cons')

    perc_cap = config.get('perc_cap')

    total_samples = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    # calculate flag that denotes whether PP consumption should be extended

    variable_speed_bool = 0

    if np.sum(pp_cons) > 0:
        values = pp_cons

        multi_mode_amp_bool = ((np.percentile(values[values > 0], 90) - np.percentile(values[values > 0], 25)) > 600 / samples_per_hour)

        variable_speed_bool = multi_mode_amp_bool

        if (item_input_object.get('created_hsm') is not None) and (item_input_object.get('created_hsm').get('pp') is not None):
            # determine if the pp is variable speed pp or 2 pp with different amplitudes

            variable_speed_bool = item_input_object.get('created_hsm').get('pp').get('attributes').get('run_type_code')[0] == 3
            variable_speed_bool = variable_speed_bool or multi_mode_amp_bool

    pp_det_conf = 1

    if item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None:
        pp_det_conf = (item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100)

    extend_pp_tou = np.sum(pp_cons > 0, axis=0) > (((np.sum(pp_cons > 0, axis=1) > 0).sum())*0.9)

    extend_pp = np.sum(extend_pp_tou) >= samples_per_hour
    extend_pp = extend_pp and (np.sum(extend_pp_tou) <= hours_thres_for_extension)
    extend_pp = extend_pp and (((np.sum(pp_cons > 0, axis=1) > 0).sum()) > 3*Cgbdisagg.DAYS_IN_MONTH)
    extend_pp = extend_pp and (pp_det_conf >= conf_thres_for_extension and variable_speed_bool ==0)
    extend_pp = extend_pp and (pp_cons[:min_days_for_pp_cons_to_be_present].sum() > 0)

    season = item_output_object.get("season")

    pp_only_in_summers = (extend_pp > 0) and (np.any(season < -0.5) and ((pp_cons[season <= -1].sum(axis=1)) > 0).sum() < min_days_for_winter_pp_cons)

    extend_pp = extend_pp * (not pp_only_in_summers) + 0.5 * (pp_only_in_summers)

    # starting extension of PP usage in nearby hours

    if not ((np.sum(pp_cons) > 0) and extend_pp > 0):
        pp_cons = extend_pp_cons_in_inc_run(item_input_object, item_output_object, pp_cons, pp_cons_disagg, timed_sig,
                                            input_data, samples_per_hour)

        return pp_cons

    pp_residual = input_data - pp_cons - timed_sig
    pp_amp = np.percentile(pp_cons[pp_cons > 0], perc_cap)
    min_cons = max(min_pp_ts_level_cons_limit/samples_per_hour, pp_amp - pp_ts_level_cons_buffer/samples_per_hour)

    # get the possible PP region based on amplitude values

    posible_region = np.logical_and(pp_residual > min_cons, pp_residual > min_cons)

    pp_tou = extend_pp_tou

    timed_app_days_seq = find_seq(pp_tou, np.zeros_like(pp_tou), np.zeros_like(pp_tou), overnight=1)

    pp_tou = update_usage_seq_in_nearby_hours(timed_app_days_seq, pp_tou, samples_per_hour, total_samples)

    # get the possible PP region based on time of days

    posible_region[:, pp_tou == 0] = 0

    posible_region[pp_cons.sum(axis=1) > 0] = 0

    # if extend pp value is 0.5, pp will only be kept in summer months

    if extend_pp == 0.5:
        posible_region[season <= -0.5] = 0

    timed_app_days = np.sum(posible_region + (pp_cons > 0), axis=1) > 0

    # post processing of derived PP  region, based on the continuity

    thres = min(min_timed_seq, max(min_pp_seq_len_for_extension, min_pp_seq_frac_days_for_extension*len(input_data)))

    timed_app_days = remove_less_days_timed_signature(timed_app_days, 6, thres)

    posible_region[timed_app_days == 0] = 0

    length = np.sum((posible_region + (pp_cons > 0)) > 0, axis=1)

    avg_duration = np.mean(length[length > 0])

    posible_region[np.sum((posible_region + (pp_cons > 0)) > 0, axis=1) < avg_duration * 0.7] = 0

    posible_region[pp_cons > 0] = 1

    posible_region = remove_low_duration_points(posible_region,  max(samples_per_hour, 3), pp_cons.shape)

    posible_region[pp_cons > 0] = 1

    if extend_pp == 0.5:
        posible_region[season <= -0.5] = 0

    timed_app_days = np.sum(posible_region + (pp_cons > 0), axis=1) > 0

    # post processing of derived twh region, based on the continuity

    thres = min(min_timed_seq, max(min_pp_seq_len_for_extension, min_pp_seq_frac_days_for_extension*len(input_data)))

    timed_app_days = remove_less_days_timed_signature(timed_app_days, 6, thres)

    posible_region[timed_app_days == 0] = 0

    posible_region[pp_cons > 0] = 0

    if extend_pp == 0.5:
        posible_region[season <= -0.5] = 0

    posible_region = posible_region * pp_amp

    posible_region[pp_cons.sum(axis=1) > 0] = 0

    pp_cons = pp_cons + posible_region

    pp_cons = extend_pp_cons_in_inc_run(item_input_object, item_output_object, pp_cons, pp_cons_disagg, timed_sig, input_data, samples_per_hour)

    return pp_cons


def extend_pp_cons_in_inc_run(item_input_object, item_output_object, pp_cons, pp_cons_disagg, timed_sig, input_data, samples_per_hour):

    """
    extends pp consumption to avoid wh underestimation in pp tou region

    Parameters:
        item_input_object       (dict)          : dict containing all hybrid inputs
        item_output_object      (dict)          : dict containing all hybrid outputs
        pp_cons                 (np.ndarray)    : pp residual output
        pp_cons_disagg          (np.ndarray)    : pp disagg output
        timed_sig               (np.ndarray)    : output of timed signature detection
        input_data              (np.ndarray)    : input data
        samples_per_hour        (int)           : samples in an hour

    Returns:
        pp_cons                (np.ndarray)     : updated pp residual output

    """

    config = get_residual_config(samples_per_hour).get('timed_app_det_config')

    season = item_output_object.get("season")

    min_pp_ts_level_cons_limit = config.get('min_pp_ts_level_cons_limit')
    pp_ts_level_cons_buffer = config.get('pp_ts_level_cons_buffer')

    min_pp_seq_len_for_extension = config.get('min_pp_seq_len_for_extension')
    min_pp_seq_frac_days_for_extension = config.get('min_pp_seq_frac_days_for_extension')
    min_timed_seq = config.get('min_timed_seq')

    total_samples = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    # fetching pp hsm data
    # this is added in order to handle cases of pp underestimation in incremental/mtd runs

    valid_pp_hsm = item_input_object.get("item_input_params").get('valid_pp_hsm')
    pp_hsm_hld = 0
    pp_tou = [0]
    pp_amp = 0
    extention_flag = 0

    # add pp if pp flag is true

    valid_hsm_flag = check_validity_of_hsm(valid_pp_hsm, item_input_object.get("item_input_params").get('pp_hsm'),
                                           'item_hld')

    if valid_hsm_flag:
        pp_hsm = item_input_object.get("item_input_params").get('pp_hsm').get('item_hld')

        if pp_hsm is None:
            pp_hsm_hld = 0
        elif isinstance(pp_hsm, list):
            pp_hsm_hld = pp_hsm[0]
        else:
            pp_hsm_hld = pp_hsm

        extention_flag = item_input_object.get("item_input_params").get('pp_hsm').get('item_extend_flag')

        if extention_flag is None:
            extention_flag = 0
        elif isinstance(extention_flag, list):
            extention_flag = extention_flag[0]

        pp_tou = item_input_object.get("item_input_params").get('pp_hsm').get('item_extend_tou')

        if pp_tou is None:
            pp_tou = np.zeros(int(samples_per_hour*24))
        elif isinstance(pp_tou, list):
            pp_tou = np.array(pp_tou)

        pp_amp = item_input_object.get("item_input_params").get('pp_hsm').get('item_amp')

        if pp_amp is None:
            pp_amp = 0
        elif isinstance(pp_amp, list):
            pp_amp = pp_amp[0]

    # adding PP in incremental run if hld is true and extension flag is true for pprevious run

    if not ((item_input_object.get('config').get('disagg_mode') in ['incremental', 'mtd']) and (np.sum(pp_cons) == 0)
            and pp_hsm_hld > 0 and pp_tou.sum() > 0 and pp_amp > 0 and extention_flag > 0):
        return pp_cons

    pp_residual = input_data - pp_cons - timed_sig
    min_cons = max(min_pp_ts_level_cons_limit/samples_per_hour, pp_amp/samples_per_hour - pp_ts_level_cons_buffer/samples_per_hour)

    # get the possible pp region based on amplitude values

    posible_region = np.logical_and(pp_residual > min_cons, pp_residual > min_cons)

    timed_app_days_seq = find_seq(pp_tou, np.zeros_like(pp_tou), np.zeros_like(pp_tou), overnight=1)

    pp_tou = update_usage_seq_in_nearby_hours(timed_app_days_seq, pp_tou, samples_per_hour, total_samples)

    posible_region[:, pp_tou == 0] = 0

    posible_region[pp_cons.sum(axis=1) > 0] = 0

    if extention_flag == 0.5:
        posible_region[season <= -0.5] = 0

    timed_app_days = np.sum(posible_region + (pp_cons > 0), axis=1) > 0

    # post processing of derived pp region, based on the continuity

    thres = min(min_timed_seq, max(min_pp_seq_len_for_extension, min_pp_seq_frac_days_for_extension*len(input_data)))

    timed_app_days = remove_less_days_timed_signature(timed_app_days, 6, thres)

    posible_region[timed_app_days == 0] = 0

    length = np.sum((posible_region + (pp_cons > 0)) > 0, axis=1)

    avg_duration = np.mean(length[length > 0])

    posible_region[np.sum((posible_region + (pp_cons > 0)) > 0, axis=1) < avg_duration * 0.7] = 0

    posible_region[pp_cons > 0] = 0

    posible_region = remove_low_duration_points(posible_region, min(5, max(samples_per_hour*1.5, 3)), pp_cons.shape)

    timed_app_days = np.sum(posible_region + (pp_cons > 0), axis=1) > 0

    timed_app_days = remove_less_days_timed_signature(timed_app_days, 6, thres)

    posible_region = posible_region * pp_amp/samples_per_hour

    if extention_flag == 0.5:
        posible_region[season <= -0.5] = 0

    posible_region[timed_app_days == 0] = 0

    posible_region[pp_cons.sum(axis=1) > 0] = 0

    pp_cons = pp_cons + posible_region

    return pp_cons


def remove_low_duration_points(posible_region, thres, shape):

    """
    remove low duration boxes from signature that will be added for appliance extension

    Parameters:
        posible_region          (np.ndarray)    : this array contains points that denotes region where PP/TWH will possibly be added
        thres                   (np.ndarray)    : min duration of boxes
        shape                   (np.ndarray)    : shape of original raw data
    Returns:
        posible_region          (np.ndarray)    : this array contains points that denotes region where PP/TWH will possibly be added

    """
    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    posible_region = posible_region.flatten()

    timed_app_days_seq = find_seq(posible_region, np.zeros_like(posible_region), np.zeros_like(posible_region),
                                  overnight=0)

    for i in range(len(timed_app_days_seq)):
        if timed_app_days_seq[i, seq_label] == 1 and timed_app_days_seq[i, seq_len] < thres:
            posible_region[timed_app_days_seq[i, seq_start]:timed_app_days_seq[i, seq_end] + 1] = 0

    posible_region = posible_region.reshape(shape)

    return posible_region


def remove_less_days_timed_signature(timed_app_days, max_non_timed_days, min_timed_days):

    """
    remove low duration days from signature that will be added for appliance extension

    Parameters:
        timed_app_days             (np.ndarray)    : this array contains days that denotes region where PP/TWH will possibly be added
        max_non_timed_days         (int)           : max count of non-timed sig days
        min_timed_days             (int)           : min count of timed sig days

    Returns:
        timed_app_days             (np.ndarray)    : this array contains days that denotes region where PP/TWH will possibly be added

    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days),
                                  overnight=0).astype(int)

    for i in range(len(timed_app_days_seq)):
        if timed_app_days_seq[i, seq_label] == 0 and timed_app_days_seq[i, seq_len] <= max_non_timed_days:
            timed_app_days[timed_app_days_seq[i, seq_start]:timed_app_days_seq[i, seq_end] + 1] = 1

    timed_app_days_seq = find_seq(timed_app_days, np.zeros_like(timed_app_days), np.zeros_like(timed_app_days),
                                  overnight=0).astype(int)

    for i in range(len(timed_app_days_seq)):
        if timed_app_days_seq[i, seq_label] == 1 and timed_app_days_seq[i, seq_len] < min_timed_days:
            timed_app_days[timed_app_days_seq[i, seq_start]:timed_app_days_seq[i, seq_end] + 1] = 0

    return timed_app_days


def update_usage_seq_in_nearby_hours(timed_app_days_seq, pp_tou, samples_per_hour, total_samples):

    """
    remove low duration days from signature that will be added for appliance extension

    Parameters:
        timed_app_days             (np.ndarray)    : this array contains days that denotes region where PP/TWH will possibly be added
        max_non_timed_days         (int)           : max count of non-timed sig days
        min_timed_days             (int)           : min count of timed sig days

    Returns:
        timed_app_days             (np.ndarray)    : this array contains days that denotes region where PP/TWH will possibly be added

    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    for i in range(len(timed_app_days_seq)):
        if timed_app_days_seq[i, seq_label] == 1:
            pp_tou[get_index_array(timed_app_days_seq[i, seq_start] - samples_per_hour, timed_app_days_seq[i, seq_start], total_samples)] = 1
            pp_tou[get_index_array(timed_app_days_seq[i, seq_end], timed_app_days_seq[i, seq_end] + samples_per_hour, total_samples)] = 1

    return pp_tou
