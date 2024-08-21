
"""
Author - Nisha Agarwal
Date - 17th Feb 2021
Detect timed appliance in residual data
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def allot_timed_app(item_input_object, inputs_required_for_timed_sig_allotment, original_tou, non_pp, logger):

    """
    allots timed signature to PP/TWH or heating category

    Parameters:
        item_input_object                       (dict)          : Dict containing all inputs
        inputs_required_for_timed_sig_allotment (list)          : list of parameters required for timed signnature allotment
        original_tou                            (np.ndarray)    : timed estimation tou
        non_pp                                  (int)           : not allot the timed sig to pp, if this value is true
        logger                                  (logger)        : logger object

    Returns:
        twh_timed_output                        (np.ndarray)    : output for TWH
        pp_timed_output                         (np.ndarray)    : output for PP
        heating_timed_output                    (np.ndarray)    : output for heating
    """

    config = get_residual_config().get('timed_app_det_config')

    min_days_required_for_pp_detection = config.get('min_days_required_for_pp_detection')

    timed_estimation = inputs_required_for_timed_sig_allotment[0]
    timed_estimation_wh = inputs_required_for_timed_sig_allotment[1]
    season_label = inputs_required_for_timed_sig_allotment[2]
    pp_cons = inputs_required_for_timed_sig_allotment[3]
    twh_cons = inputs_required_for_timed_sig_allotment[4]

    samples_per_hour = int(timed_estimation.shape[1] / Cgbdisagg.HRS_IN_DAY)

    bin_arr = [0, 0, 0, 0]

    # determine whether timed signature shud be alloted to pp based on hsm and pp disagg information

    bin_arr[0] = check_allotment_flag_based_on_disagg_output(timed_estimation, pp_cons, item_input_object, samples_per_hour, logger)

    twh_timed_output = np.zeros(pp_cons.shape)
    pp_timed_output = np.zeros(pp_cons.shape)
    heating_timed_output = np.zeros(pp_cons.shape)

    bin_arr = np.array(bin_arr).astype(int)

    # fetch pp app prof and HSM HLD info

    hld, app_prof = fetch_hsm_and_app_prof_info(item_input_object, logger)

    allot_to_pp = app_prof and (not np.any(pp_cons > 0))

    logger.info('PP allotment flag based on app profile and disagg consumption | %s', allot_to_pp)

    # determine whether timed signature shud be alloted to pp based on parameters of the detected signature

    pp_rejection_bool = prepare_pp_rejection_bool(item_input_object, timed_estimation, season_label,
                                                  pp_cons, original_tou, samples_per_hour, hld, logger)

    logger.info('PP allotment rejection flag | %s', pp_rejection_bool)

    allot_to_pp = allot_to_pp and (not pp_rejection_bool)

    if len(pp_cons) < min_days_required_for_pp_detection:
        return twh_timed_output, pp_timed_output, heating_timed_output

    # If pp flag is true, pp cons is updated with timed sig output

    if (allot_to_pp or bin_arr[0] > 0) and (not non_pp):
        pp_timed_output = timed_estimation

    pilot = item_input_object.get("config").get("pilot_id")

    twh_pilots = np.append(PilotConstants.TIMED_WH_PILOTS, PilotConstants.TIMED_WH_JAPAN_PILOTS)

    allot_to_wh = get_wh_allotment_bool(item_input_object, timed_estimation_wh, season_label, twh_cons, twh_pilots, logger)

    # If wh flag is true and it belongs to twh pilots, wh cons is updated with timed sig output

    if allot_to_wh and (pilot in twh_pilots):
        twh_timed_output = timed_estimation_wh

    return twh_timed_output, pp_timed_output, heating_timed_output


def get_wh_allotment_bool(item_input_object, timed_estimation, season_label, twh_cons, twh_pilots, logger):

    """
    prepare intermediate pp detection results in disagg module, which can further be used for timed signature detection

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        timed_estimation          (np.ndarray)    : final timed estimation
        season_label              (np.ndarray)    : season list
        twh_cons                  (np.ndarray)    : TWH disagg output
        twh_pilots                (np.ndarray)    : TWH pilot list
        logger                    (logger)        : logger object

    Returns:
        allot_to_wh               (bool)          : flag based on whether timed signature can be alloted to TWH category
    """

    pilot = item_input_object.get("config").get("pilot_id")

    samples_per_hour = int(len(timed_estimation[0]) / Cgbdisagg.HRS_IN_DAY)

    # fetch wh app profile

    config = get_inf_config().get("wh")
    amp_perc = 90

    app_profile =  get_app_prof_params(item_input_object)

    input_data = item_input_object.get('item_input_params').get('day_input_data')

    input_data = input_data - item_input_object.get('item_input_params').get('ao_cons')

    # fetch wh hsm params

    hld, hsm_tou, wh_type, amp = get_wh_hsm_params(item_input_object, timed_estimation, input_data, logger)

    # checking whether previous run WH output was non zero and timed

    allot_to_wh = (app_profile or (pilot in twh_pilots)) and hld and wh_type==1 and (pilot in twh_pilots)

    # prepare twh output in mtd mode, if twh was detected in historical run

    amp_buffer = config.get('twh_amp_buffer')

    japan_pilots = PilotConstants.TIMED_WH_JAPAN_PILOTS

    dur_thres = config.get('twh_min_thres') * (pilot not in japan_pilots) + config.get('twh_min_thres_japan') * (pilot in japan_pilots)

    pot_twh_points = np.logical_and(input_data > (max(dur_thres/samples_per_hour, amp-amp_buffer/samples_per_hour)),
                                    input_data < (amp+amp_buffer/samples_per_hour))

    # this checks whether previous run had twh detected, and whether current run output TOU overlaps with previous run output

    current_run_timed_sig_matching_with_previous_run = \
        (hld and wh_type==1 and hsm_tou.sum() > 0) and \
        np.sum(np.sum(pot_twh_points[:, hsm_tou.astype(int)], axis=1) > 2*samples_per_hour) > \
        0.5*len(input_data) and (pilot in twh_pilots)

    if current_run_timed_sig_matching_with_previous_run:
        timed_estimation = copy.deepcopy(input_data)
        timed_estimation[:, np.logical_not(hsm_tou).astype(bool)] = 0
        allot_to_wh = 1

        logger.info('TWH allotment flag based on TWH HLD of previous disagg run | %s', True)

        return allot_to_wh

    tou = timed_estimation.sum(axis=0) > 0
    tou = tou.astype(int)

    seq = find_seq(tou, np.zeros_like(tou), np.zeros_like(tou), overnight=1)

    dur_thres = (config.get('twh_dur_thres') * samples_per_hour) * (pilot not in PilotConstants.TIMED_WH_JAPAN_PILOTS) +\
                (config.get('twh_dur_thres_japan') * samples_per_hour) * (pilot in PilotConstants.TIMED_WH_JAPAN_PILOTS)

    # elimate twh addition if length of twh doesnot satisfy

    twh_amp = 0

    seq_label = seq_config.SEQ_LABEL
    seq_len = seq_config.SEQ_LEN

    day_time_hours = np.arange(4*samples_per_hour, 21*samples_per_hour+1)

    if np.sum(timed_estimation) > 0:
        twh_amp = np.percentile(timed_estimation[timed_estimation > 0], 95)

    if np.any(np.logical_and(seq[:, seq_len] > dur_thres, seq[:, seq_label] == 1)):
        allot_to_wh = 0

    logger.info('TWH allotment flag based on timed signature length | %s', allot_to_wh)

    # checking deviation between amplitudes of disagg TWH and hybrid timed signature

    twh_disagg_is_non_zero = np.sum(twh_cons) and np.sum(timed_estimation)

    if twh_disagg_is_non_zero:
        allot_to_wh = np.any(twh_cons > 0) and np.abs(np.percentile(twh_cons[twh_cons  > 0], amp_perc) - twh_amp) < \
                      amp_buffer / samples_per_hour

    logger.info('TWH allotment flag based on timed signature amplitude | %s', allot_to_wh)

    # This check to eliminate cases where user has high day time TWH usage during summers to avoid FPs

    if np.any(season_label >= 0):
        tou = timed_estimation[:, day_time_hours][season_label >= 0].sum(axis=0) > 0
        tou = tou.astype(int)

        seq = find_seq(tou, np.zeros_like(tou), np.zeros_like(tou),overnight=1)

        high_cons_during_day_time_in_summers = \
            np.sum(timed_estimation[:, day_time_hours][season_label >= 0]) and \
            (np.any(np.logical_and(seq[:, seq_len] > 3*samples_per_hour, seq[:, 0] == 1)))

        allot_to_wh = allot_to_wh * (not high_cons_during_day_time_in_summers) + 0 * (high_cons_during_day_time_in_summers)

    logger.info('TWH allotment flag based on seasonality | %s', allot_to_wh)

    allot_to_wh = determine_wh_allotment_flag(timed_estimation, twh_cons, twh_amp, twh_pilots, pilot, samples_per_hour, allot_to_wh)

    logger.info('TWH allotment flag based on segments in timed signature | %s', allot_to_wh)

    # elimate twh addition if wh app profile input is non electric or 0 wh count

    app_profile = item_input_object.get("app_profile").get('wh')

    if app_profile is None:
        return allot_to_wh

    allot_to_wh = update_wh_allotment_flag_based_on_app_profile(item_input_object, app_profile, allot_to_wh)

    logger.info('TWH allotment flag based on WH app profile | %s', allot_to_wh)

    return allot_to_wh


def update_wh_allotment_flag_based_on_app_profile(item_input_object, app_profile, allot_to_wh):

    """
    check wh presence based on app profile

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        allot_to_wh               (bool)          : flag based on whether timed signature can be alloted to TWH category

    Returns:
        allot_to_wh               (bool)          : flag based on whether timed signature can be alloted to TWH category
    """

    app_profile = app_profile.get("number", 0)

    if app_profile == 0:
        allot_to_wh = 0

    # removing TWH for storage WH type user

    wh_type = item_input_object.get("app_profile").get('wh').get("attributes", '')

    if wh_type is not None and "storagetank" in wh_type:
        allot_to_wh = 0

    # removing TWH for gas type users

    wh_type = item_input_object.get("app_profile").get('wh').get("type", '')

    if wh_type is not None and wh_type == 'GAS':
        allot_to_wh = 0

    return allot_to_wh


def determine_wh_allotment_flag(timed_estimation, twh_cons, twh_amp, twh_pilots, pilot, samples_per_hour, allot_to_wh):

    """
    Check whether timed signature can be alloted to TWH category

    Parameters:
        timed_estimation          (np.ndarray)    : final timed estimation
        twh_cons                  (np.ndarray)    : disagg TWH consumption
        twh_amp                   (int)           : TWH amplitude
        twh_pilots                (list)          : list of pilots for which TWH is enabled
        pilot                     (int)           : pilot id of the user
        samples_per_hour          (int)           : samples in an hour
        allot_to_wh               (bool)          : flag based on whether timed signature can be alloted to TWH category

    Returns:
        allot_to_wh               (bool)          : flag based on whether timed signature can be alloted to TWH category
    """

    tou = timed_estimation.sum(axis=1) > 0
    tou = tou.astype(int)

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    config = get_inf_config().get("wh")

    # elimate twh addition if continuous segment of TWH is absent

    seq = find_seq(tou, np.zeros_like(tou), np.zeros_like(tou), overnight=0)

    inconsistent_timed_sig_detected = \
        (np.any(np.logical_and(seq[:, seq_len] > 50, seq[:, seq_label] == 0)) and twh_cons.sum()==0) or \
        (np.sum(twh_cons) == 0 and (np.sum(timed_estimation > 0, axis=1) > 0).sum() < 0.5*len(timed_estimation))

    if inconsistent_timed_sig_detected:
        allot_to_wh = 0

    for i in range(len(seq)):
        if seq[i,  seq_len] < Cgbdisagg.DAYS_IN_MONTH and not seq[i, seq_label]:
            tou[seq[i, seq_start]:seq[i, seq_end] + 1] = 1

    seq = find_seq(tou, np.zeros_like(tou), np.zeros_like(tou), overnight=0)

    less_days_timed_sig_detected = np.any(np.logical_and(seq[:, seq_len] < 0.5*len(timed_estimation),
                                                         seq[:, seq_label] == 1)) and twh_cons.sum()==0

    if less_days_timed_sig_detected:
        allot_to_wh = 0

    tou = np.sum(timed_estimation, axis=0) > 0

    # elimate twh addition if amplitude of twh doesnot satisfy

    japan_pilots = PilotConstants.TIMED_WH_JAPAN_PILOTS

    thres = config.get('twh_min_thres') * (pilot not in japan_pilots) + \
            config.get('twh_min_thres_japan') * (pilot in japan_pilots)

    timed_sig_amp_is_out_of_bounds = \
        ((twh_cons[:, tou].sum() == 0) and (pilot in twh_pilots) and (np.sum(timed_estimation) > 0)) and \
            ((twh_amp < thres/samples_per_hour) or (twh_amp > config.get('twh_max_thres')/samples_per_hour))

    if timed_sig_amp_is_out_of_bounds:
        allot_to_wh = 0

    return allot_to_wh


def get_app_prof_params(item_input_object):

    """
    fetch app profile

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs

    Returns:
        app_profile               (dict)          : wh app profile
    """

    # fetch wh app profile info

    app_profile = item_input_object.get("app_profile").get('wh')

    if app_profile is not None:
        app_profile = app_profile.get("number", 0)
        type = item_input_object.get("app_profile").get('wh').get("attributes", '')

        if type is not None and "storagetank" in type:
            app_profile = 0

        type = item_input_object.get("app_profile").get('wh').get("type", '')

        if type is not None and type == 'GAS':
            app_profile = 0
    else:
        app_profile = 0

    return app_profile


def get_wh_hsm_params(item_input_object, timed_estimation, input_data, logger):

    """
    fetch WH hsm params

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        timed_estimation          (np.ndarray)    : final timed estimation
        input_data                (np.ndarray)    : input data
        logger                    (logger)        : logger object

    Returns:
        hld                       (int)           : hsm hld info
        tou                       (np.ndarray)    : hsm tou info
        type                      (int)           : hsm wh type info
        amp                       (int)           : hsmm wh amp info
    """

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')

    # initializing with default parameters
    hld = 1
    type = 1
    amp = 2500
    tou = np.zeros(len(timed_estimation[0]))

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'),
                                           'item_tou')

    if not valid_hsm_flag:
        return hld, tou, type, amp

    wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_tou')

    if wh_hsm is None:
        hld = 0
    else:
        tou = np.array(wh_hsm)

    wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_amp')

    if wh_hsm is None:
        type = '0'
    elif isinstance(wh_hsm, list):
        amp = wh_hsm[0]
    else:
        amp = wh_hsm

    wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_hld')

    if wh_hsm is None:
        amp = 0
    elif isinstance(wh_hsm, list):
        hld = wh_hsm[0]
    else:
        hld = wh_hsm

    wh_hsm = item_input_object.get("item_input_params").get('wh_hsm').get('item_type')

    if wh_hsm is None:
        tou = np.zeros(len(timed_estimation[0]))
    elif isinstance(wh_hsm, list):
        type = wh_hsm[0]
    else:
        type = wh_hsm

    if len(tou) == len(input_data[0]):
        tou = tou.astype(bool)
    else:
        tou = np.ones(len(input_data[0])).astype(bool)

    logger.info('WH amp in HSM | %s', amp)
    logger.info('WH type in HSM | %s', type)
    logger.info('WH hld in HSM | %s', hld)
    logger.info('WH tou in HSM | %s', tou)

    return hld, tou, type, amp


def prepare_pp_rejection_bool(item_input_object, timed_estimation, season_label, pp_cons, original_tou, samples_per_hour, hld, logger):

    """
    prepare intermediate pp detection results in disagg module, which can further be used for timed signature detection

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        timed_estimation          (np.ndarray)    : final timed estimation
        season_label              (np.ndarray)    : season list
        pp_cons                   (np.ndarray)    : pp disagg consumption
        original_tou              (np.ndarray)    : timed estimation tou
        samples_per_hour          (int)           : samples in an hour
        hld                       (int)           : hsm hld info

    Returns:
        item_output_object        (dict)          : Dict containing all outputs
    """

    config = get_residual_config().get('timed_app_det_config')

    max_pp_amp = config.get('max_pp_amp')
    min_pp_amp = config.get('min_pp_amp')
    min_timed_sig_len = config.get('min_timed_sig_len')
    len_to_check_on_min_timed_sig_len = config.get('len_to_check_on_min_timed_sig_len')
    max_pp_len = config.get('max_pp_len')
    min_pp_len = config.get('min_pp_len')

    # reject poolpump based on consumption level

    pp_reject_bool1 = np.sum(timed_estimation) and \
                      ((np.percentile(timed_estimation[timed_estimation > 0], 80) > max_pp_amp / samples_per_hour) or
                       (np.percentile(timed_estimation[timed_estimation > 0], 90) < min_pp_amp / samples_per_hour))

    logger.info('PP allotment rejection flag based on consumption level | %s', pp_reject_bool1)

    # reject poolpump based on length of signature

    pp_reject_bool2 = np.sum(timed_estimation) and np.sum(pp_cons) == 0 and \
                      (np.sum(timed_estimation.sum(axis=1) > 0) < min_timed_sig_len and
                       len(timed_estimation) > len_to_check_on_min_timed_sig_len) and \
                      item_input_object.get('config').get('disagg_mode') != 'mtd'

    duration = (np.sum(timed_estimation > 0, axis=1)[np.sum(timed_estimation > 0, axis=1) > 0]).mean()

    logger.info('PP allotment rejection flag based on length of signature | %s', pp_reject_bool2)

    # reject poolpump based on duration

    pp_reject_bool3 = np.sum(timed_estimation) and ((duration > max_pp_len * samples_per_hour) or (duration < min_pp_len * samples_per_hour))

    logger.info('PP allotment rejection flag based on duration | %s', pp_reject_bool3)

    # reject poolpump based on seasonality

    pp_reject_bool4 = ((not np.sum(pp_cons)) and not hld) or (np.sum(original_tou[season_label >= 0]) == 0)

    logger.info('PP allotment rejection flag based on seasonality | %s', pp_reject_bool4)

    final_bool = pp_reject_bool1 or pp_reject_bool2 or pp_reject_bool3 or pp_reject_bool4

    return final_bool


def fetch_hsm_and_app_prof_info(item_input_object, logger):

    """
    fetches pp hsm and app prof info

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs

    Returns:
        hld                       (int)           : pp hsm hld info
        app_prof                  (int)           : pp app count
    """

    valid_pp_hsm = item_input_object.get("item_input_params").get('valid_pp_hsm')
    hld = 1

    valid_hsm_flag = check_validity_of_hsm(valid_pp_hsm, item_input_object.get("item_input_params").get('pp_hsm'),
                                           'item_hld')

    if valid_hsm_flag:
        pp_hsm = item_input_object.get("item_input_params").get('pp_hsm').get('item_hld')

        if pp_hsm is None:
            hld = 0
        elif isinstance(pp_hsm, list):
            hld = pp_hsm[0]
        else:
            hld = pp_hsm

    app_prof = item_input_object.get("app_profile").get('pp')

    if app_prof is not None:
        app_prof = app_prof.get("number", 0)
    else:
        app_prof = 0

    logger.info('PP HLD in HSM | %s', hld)

    return hld, app_prof


def check_allotment_flag_based_on_disagg_output(timed_estimation, pp_cons, item_input_object, samples_per_hour, logger):

    """
    determine whether to allot timed appliance based on pp hsm

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        pp_cons                   (np.ndarray)    : disagg pp output
        item_output_object        (dict)          : Dict containing all outputs
        samples_per_hour          (int)           : samples in an hour

    Returns:
        flag                      (int)          : flag is true, if timed sigaturee is alloted to pp
    """

    disagg_hld = np.sum(pp_cons) > 0

    pp_amp = 0

    if np.any(pp_cons > 0):
        pp_amp = np.percentile(pp_cons[pp_cons > 0], 90)

    valid_pp_hsm = item_input_object.get("item_input_params").get('valid_pp_hsm')

    valid_hsm_flag = check_validity_of_hsm(valid_pp_hsm, item_input_object.get("item_input_params").get('pp_hsm'),
                                           'item_amp')

    if valid_hsm_flag:
        pp_hsm = item_input_object.get("item_input_params").get('pp_hsm').get('item_amp')

        if pp_hsm is None:
            pp_hsm_amp = 0
        elif isinstance(pp_hsm, list):
            pp_hsm_amp = pp_hsm[0]
        else:
            pp_hsm_amp = pp_hsm

        pp_hsm = item_input_object.get("item_input_params").get('pp_hsm').get('item_hld')

        if pp_hsm is None:
            hld = 0
        elif isinstance(pp_hsm, list):
            hld = pp_hsm[0]
        else:
            hld = pp_hsm

        if disagg_hld == 0:
            disagg_hld = hld
            pp_amp = pp_hsm_amp

    if disagg_hld and np.sum(timed_estimation):
        flag = np.any(pp_cons > 0) and np.abs(pp_amp -
                                              np.percentile(timed_estimation[timed_estimation > 0], 90)) < 500 / samples_per_hour

    else:
        flag = disagg_hld

    return flag
