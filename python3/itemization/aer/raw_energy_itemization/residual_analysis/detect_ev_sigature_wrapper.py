

"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.raw_energy_itemization.inference_engine.get_ev_inference import get_ev_residual_signature


def ev_signature_detection_wrapper(item_input_object, item_output_object, season, logger):

    """
    Prepare hybrid input object

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        season                    (np.ndarray): season list
        logger                    (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        residual_copy             (np.ndarray): residual data
    """

    # Fetch required data

    input_data = item_output_object.get("hybrid_input_data").get("input_data")
    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    residual_copy = item_output_object.get("hybrid_input_data").get("original_res")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")
    disagg_residual = item_output_object.get("hybrid_input_data").get("true_disagg_res")
    samples_per_hour = int(input_data.shape[2] /  Cgbdisagg.HRS_IN_DAY)

    ev_idx = np.where(np.array(appliance_list) == 'ev')[0][0] + 1

    # fetching ev app profile

    app_profile = item_input_object.get("app_profile").get('ev')

    if app_profile is not None:
        app_profile = app_profile.get("number", 0)
        logger.info('EV app profile present | ')
    else:
        app_profile = 0

    ev_disagg = output_data[ev_idx]

    # preparing consumption that can be added into ev residual to maintain consistency at billing cycle level

    non_ev_residual, disagg_residual = \
        prepare_cons_that_can_be_added_into_ev(item_output_object, output_data, appliance_list, disagg_residual, samples_per_hour)

    if app_profile or np.sum(ev_disagg) > 0:

        logger.info('Running EV residual detection for L2 | ')

        # try detecting ev l2/l1 boxes

        ev_disagg = output_data[ev_idx]

        ev_added_type, ev_residual, valid_idx = \
            get_ev_residual_boxes(item_input_object, item_output_object, non_ev_residual, ev_disagg, disagg_residual,
                                  samples_per_hour, logger)

        vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)

        # checking if seasonality needs to be evaluated based on previous runs info

        run_seasonal_check = check_ev_seasonal_check_flag(input_data, item_input_object)

        logger.info('EV seasonal check flag %s | ', run_seasonal_check)

        # if seasonality is found in ev output and previous run output was 0, the hybrid ev boxes are removed

        if run_seasonal_check:
            logger.debug("Running seasonality checks for EV detection | ")
            valid_idx, ev_residual = eliminate_seasonal_ev_cases(item_input_object, vacation, valid_idx,
                                                                 samples_per_hour, copy.deepcopy(season), ev_disagg, ev_residual, logger)

        if np.sum(ev_residual) > 0:
            amp = np.percentile(ev_residual[ev_residual > 0], 90)
            ev_residual = np.fmin(ev_residual, amp)
        else:
            amp = 0
            ev_residual[:, :] = 0

        if np.sum(ev_residual) == 0 and np.sum(ev_disagg) == 0:
            ev_added_type = ''

        # EV hybrid output is capped based on residual data

        logger.info('Amplitude of EV consumption added fromm residual data | %s', np.round(amp*samples_per_hour))

        ev_hybrid_cons = np.minimum(ev_residual, np.fmax(0, item_output_object.get('original_input_data')-non_ev_residual))
        ev_hybrid_cons[ev_hybrid_cons > 0] = amp

        ev_hybrid_cons = np.minimum(ev_hybrid_cons, input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX])

        item_output_object.update({
            "ev_residual": ev_hybrid_cons
        })

        # ev hsm is updated based on new ev output

        created_hsm = prepare_ev_hsm(item_input_object, ev_disagg, ev_hybrid_cons, samples_per_hour, ev_added_type, logger)

        item_output_object = update_ev_hsm(created_hsm, item_input_object, item_output_object)

        residual_copy = residual_copy - ev_residual

    ev_disagg = output_data[ev_idx]

    if np.sum(ev_disagg) > 0:
        ev_disagg = output_data[ev_idx]
        created_hsm = prepare_ev_hsm(item_input_object, ev_disagg, np.zeros_like(ev_disagg), samples_per_hour, '', logger)

        item_output_object = update_ev_hsm(created_hsm, item_input_object, item_output_object)

    return item_input_object, item_output_object, residual_copy


def get_ev_residual_boxes(item_input_object, item_output_object, non_ev_residual, ev_disagg, disagg_residual, samples_per_hour, logger):

    """
    pick ev l1/l2 boxes from residual

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid inputs
        non_ev_residual           (np.ndarray)    : amount of energy that cannot be added to ev
        ev_disagg                 (np.ndarray)    : ev disagg output
        ev_residual               (np.ndarray)    : ev signature in residual data
        disagg_residual           (np.ndarray)    : disagg residual
        samples_per_hour          (int)           : samples in an hour
        logger                    (logger)        : logger object
    Returns:
        ev_added_type             (int)           : ev charger type added from hybrid
        ev_residual               (np.ndarray)    : ts level ev estimation in hybrid v2
        valid_idx                 (np.ndarray)    : ts level ev boxes detected in hybrid v2

    """

    pilot = item_input_object.get("config").get("pilot_id")
    ev_config = get_inf_config().get('ev')
    ev_added_type = ''

    # try detecting ev l2 boxes

    valid_idx, ev_residual = \
        get_ev_residual_signature(non_ev_residual, item_input_object, item_output_object, ev_disagg,
                                  np.fmax(0, disagg_residual),
                                  item_output_object.get('original_input_data'), samples_per_hour, logger)

    ev_residual = np.minimum(ev_residual, np.fmax(0, item_output_object.get('original_input_data') - non_ev_residual))

    if np.sum(valid_idx) > 0 and np.sum(ev_disagg) == 0:
        ev_added_type = 'l2'

    if not np.sum(valid_idx):

        logger.info('Running EV residual detection for L1 | ')

        # if ev l2 not detected, try detecting ev l1 boxes

        valid_idx, ev_residual = \
            get_ev_residual_signature(non_ev_residual, item_input_object, item_output_object, ev_disagg,
                                      np.fmax(0, disagg_residual),
                                      item_output_object.get('original_input_data'), samples_per_hour,
                                      logger, l1_bool=True)

        ev_residual = np.minimum(ev_residual, np.fmax(0, disagg_residual))

        if np.sum(valid_idx) > 0 and np.sum(ev_disagg) == 0:
            ev_added_type = 'l1'

    ev_residual = np.minimum(ev_residual, np.fmax(0, item_output_object.get('original_input_data') - non_ev_residual))

    low_ev_detected_in_high_cons_pilot_users = \
        np.sum(valid_idx) and (pilot in ev_config.get('high_hvac_pilots')) and \
        np.percentile(ev_residual[ev_residual > 0], 70) < ev_config.get('high_hvac_pilots_amp_thres') / samples_per_hour

    if low_ev_detected_in_high_cons_pilot_users:
        valid_idx[:] = 0
        ev_residual[:] = 0
        logger.info('Detected ev residual but removed due to presence of high hvac pilot | ')

    return ev_added_type, ev_residual, valid_idx


def check_ev_seasonal_check_flag(input_data, item_input_object):

    """
    check whether EV seasonal checks should run based on run and hsm info

    Parameters:
        input_data                (np.ndarray)    : user input data
        item_input_object         (dict)          : Dict containing all hybrid inputs

    Returns:
        run_seasonal_check        (int)           : flag that represents whether EV seasonal checks should run
    """

    run_seasonal_check = 1

    valid_hsm_present = \
        item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_type') is not None

    if valid_hsm_present:
        ev_type = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(ev_type, list):
            ev_type = ev_type[0]

        l2_type_ev_detected = ev_type == 2 and len(input_data) > 90

        if l2_type_ev_detected:
            run_seasonal_check = 0

    if item_input_object.get("config").get('disagg_mode') == 'mtd':
        run_seasonal_check = 0

    return run_seasonal_check


def prepare_cons_that_can_be_added_into_ev(item_output_object, output_data, appliance_list, disagg_residual, samples_per_hour):

    """
    prepare consumption that can be added to ev residual

    Parameters:
        item_output_object         (dict)          : Dict containing all hybrid outputs
        output_data                (np.ndarray)    : disagg output for all appliances
        appliance_list             (list)          : list of all appliances
        disagg_residual            (np.ndarray)    : disagg residual data
        samples_per_hour           (int)           : samples in an hour

    Returns:
        non_ev_residual            (np.ndarray)    : consumption that wont be added to ev residual data -
                                                     as it is part of PP/WH/AO/REF/EV disagg estimates
        disagg_residual            (np.ndarray)    : disagg residual data
    """

    timed_output = copy.deepcopy(item_output_object.get("timed_app_dict").get("timed_output"))

    if timed_output is None:
        timed_output = np.zeros_like(output_data[0])

    # consumption belonging to disagg EV+PP+AO+REF+WH wont be added into hybrid ev output

    non_ev_residual = output_data[np.where(np.array(appliance_list) == 'pp')[0][0] + 1] + \
                  output_data[np.where(np.array(appliance_list) == 'wh')[0][0] + 1] + \
                  output_data[np.where(np.array(appliance_list) == 'ao')[0][0] + 1] + \
                  output_data[np.where(np.array(appliance_list) == 'ref')[0][0] + 1] + \
                  output_data[np.where(np.array(appliance_list) == 'ev')[0][0] + 1]

    # timed signature detected in hybrid wont be added into hybrid ev output

    if np.sum(timed_output) > 0 and (item_output_object.get("timed_app_dict").get("twh").sum() > 0):
        timed_output = np.maximum(timed_output, item_output_object.get("timed_app_dict").get("twh"))

    if np.sum(timed_output) > 0 and (item_output_object.get("timed_app_dict").get("pp").sum() > 0):
        timed_output = np.maximum(timed_output, item_output_object.get("timed_app_dict").get("pp"))

    remove_timed_app = (np.sum(timed_output) and (np.median(timed_output[timed_output > 0]) < 3000/samples_per_hour)) or \
                       ((item_output_object.get("timed_app_dict").get("pp").sum() > 0) or (item_output_object.get("timed_app_dict").get("twh").sum() > 0))

    if remove_timed_app:
        non_ev_residual = non_ev_residual + timed_output
        disagg_residual = disagg_residual - timed_output

    return non_ev_residual, disagg_residual


def prepare_ev_hsm(item_input_object, ev_disagg, ev_residual, samples_per_hour, ev_type, logger):

    """
    Update ev hsm

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        ev_disagg                 (np.ndarray)    : ev disagg output
        ev_residual               (np.ndarray)    : ev signature in residual data
        samples_per_hour          (int)           : samples in an hour
        logger                    (logger)        : logger object

    Returns:
        created_hsm               (dict)          : ev hsm updated with hybrid ev attributes
    """

    created_hsm = dict({
        'item_type': 0,
        'item_amp': 0
    })

    # update hsm if ev added from hybrid

    if np.sum(ev_disagg) == 0:
        created_hsm = prepare_hsm_for_hybrid_ev_users(created_hsm, item_input_object, ev_residual, samples_per_hour, ev_type, logger)

    # update hsm if ev added from true disagg

    else:
        created_hsm = prepare_hsm_for_disagg_ev_users(created_hsm, item_input_object, ev_disagg, samples_per_hour)

    logger.info('Prepared EV HSM after EV detection in residual data | %s', created_hsm)

    return created_hsm


def prepare_hsm_for_disagg_ev_users(created_hsm, item_input_object, ev_disagg, samples_per_hour):

    """
    Update ev hsm

    Parameters:
        created_hsm               (dict)          : ev hsm updated with hybrid ev attributes
        item_input_object         (dict)          : Dict containing all hybrid inputs
        ev_disagg                 (np.ndarray)    : ev disagg output
        samples_per_hour          (int)           : samples in an hour

    Returns:
        created_hsm               (dict)          : ev hsm updated with hybrid ev attributes
    """

    # cases where ev is present in disagg, hsm is prepared based on ev output

    disagg_ev_hsm = \
        (item_input_object.get('created_hsm') is not None) and \
        (item_input_object.get('created_hsm').get('ev') is not None) and \
        (item_input_object.get('created_hsm').get('ev').get('attributes') is not None)

    max_cons = np.median(ev_disagg[ev_disagg > 0]) * samples_per_hour

    # default charger type

    charger_type = 'l1'

    l2_charger_detected = np.median(ev_disagg[ev_disagg > 0]) > (2700 / samples_per_hour)

    if l2_charger_detected:
        charger_type = 'l2'

    # using true disagg info to check charger type if available in debug dict

    if disagg_ev_hsm and (
            item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude") is not None):
        max_cons = item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude") * 1.1

    if disagg_ev_hsm and (
            item_input_object.get('created_hsm').get('ev').get('attributes').get("charger_type") is not None):
        charger_type = item_input_object.get('created_hsm').get('ev').get('attributes').get("charger_type")

    created_hsm['item_type'] = charger_type
    created_hsm['item_amp'] = max_cons

    return created_hsm


def prepare_hsm_for_hybrid_ev_users(created_hsm, item_input_object, ev_residual, samples_per_hour, ev_type, logger):

    """
    Update ev hsm

    Parameters:
        created_hsm               (dict)          : ev hsm updated with hybrid ev attributes
        item_input_object         (dict)          : Dict containing all hybrid inputs
        ev_residual               (np.ndarray)    : ev signature in residual data
        samples_per_hour          (int)           : samples in an hour
        ev_type                   (int)           : charger type
        logger                    (logger)        : logger object

    Returns:
        created_hsm               (dict)          : ev hsm updated with hybrid ev attributes
    """

    # cases where ev is present in disagg, hsm is prepared based on hybrid ev output

    ev_hsm_type = 0

    if item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_type') is not None:
        ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        ev_hsm_type = ev_hsm

        if isinstance(ev_hsm, list):
            ev_hsm_type = ev_hsm[0]

    # inorder to avoid false ev addition incase actual ev already detected before
    if (ev_hsm_type in [1, 2, 3]) and np.sum(ev_residual) == 0:
        created_hsm['item_type'] = 3
        created_hsm['item_amp'] = 0

    elif np.sum(ev_residual) > 0 and ev_type != '':
        logger.info("Found ev consumption for charger type %s | ", ev_type)
        charger_type = 1 if ev_type == 'l1' else 2
        created_hsm['item_type'] = charger_type
        created_hsm['item_amp'] = np.median(ev_residual[ev_residual > 0]) * samples_per_hour

    return created_hsm


def eliminate_seasonal_ev_cases(item_input_object, vacation, valid_idx, samples_per_hour, season, ev_disagg, ev_residual, logger):

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

    if np.all(vacation):
        return np.zeros_like(valid_idx), np.zeros_like(ev_residual)

    ev_config = get_inf_config(samples_per_hour).get('ev')

    ev_days = (np.sum(valid_idx, axis=1) > 0).astype(int)

    high_ev_usage_hours = ev_config.get('high_ev_usage_hours')

    # comparison of recent days segment and complete ev days to check whether the EV was detected recently
    # If yes, seasonality checks are not performed on detected boxes

    days_segment_2 = max(60, int(len(ev_days)*0.6))
    days_segment_1 = min(90, days_segment_2 - (len(ev_days) - days_segment_2) / 2)

    recency = ((ev_days[-days_segment_2:].sum() / ev_days.sum()) > 0.9)

    consumption_present_in_recent_days = ev_days[-days_segment_2:].sum() > 0

    sig_cons_present_in_last_months_of_target_wind = ((ev_days[-int(days_segment_1):].sum() / ev_days[-days_segment_2:].sum()) > 0.3)

    if consumption_present_in_recent_days:
        recency = recency and sig_cons_present_in_last_months_of_target_wind

    logger.info('Recency flag of EV addition | %s', recency)

    min_season_days_required = 20

    season = season[np.logical_not(vacation)]

    season_days_present = \
        not recency and (not np.all(season == 0) and (not np.all(season < 0)) and
                         (not np.all(season > 0))) and np.sum(valid_idx) and (np.sum(ev_disagg) == 0)

    if not season_days_present:
        return valid_idx, ev_residual

    cons = np.zeros(3)
    cons[0] = np.sum(valid_idx[np.logical_not(vacation)][season > 0][:, high_ev_usage_hours]) / np.sum(season > 0)
    cons[1] = np.sum(valid_idx[np.logical_not(vacation)][season == 0][:, high_ev_usage_hours]) / np.sum(season == 0)
    cons[2] = np.sum(valid_idx[np.logical_not(vacation)][season < 0][:, high_ev_usage_hours]) / np.sum(season < 0)

    cons = np.nan_to_num(cons)

    zero_cons_in_peak_season = \
        (cons[0] == 0 and (np.sum(season > 0) > min_season_days_required)) or \
        (cons[2] == 0 and (np.sum(season < 0) > min_season_days_required))

    low_cons_in_transition = \
        np.sum(season == 0) > Cgbdisagg.DAYS_IN_MONTH and np.sum(season > 0) > min_season_days_required and\
        np.sum(season < 0) > min_season_days_required and (cons[1] < ((cons[0] + cons[2]) / 20))

    high_cons_frac_in_summers = \
        (np.sum(season > 0) > min_season_days_required) and \
        np.sum(cons[0]) > ev_config.get('seasons_comparison_thres') * np.sum(cons[1:])

    high_cons_frac_in_winters = \
        (np.sum(season < 0) > min_season_days_required) and \
        np.sum(cons[2]) > ev_config.get('seasons_comparison_thres') * np.sum(cons[:2])

    block_ev_detection = zero_cons_in_peak_season + low_cons_in_transition + high_cons_frac_in_summers + high_cons_frac_in_winters

    logger.info('flag based on newly detected ev because either of winter or summer output is 0 | %s ', zero_cons_in_peak_season)
    logger.info('flag based on newly detected ev because summer output is high | %s', high_cons_frac_in_summers)
    logger.info('flag based on newly detected ev because winter output is high | %s', high_cons_frac_in_winters)
    logger.info('flag based on newly detected ev because tranistion output is less | %s', low_cons_in_transition)

    # blocked newly detected ev because either of winter or summer output is 0

    valid_idx = valid_idx * (1 - block_ev_detection)
    ev_residual = ev_residual * (1 - block_ev_detection)

    return valid_idx, ev_residual


def update_ev_hsm(created_hsm, item_input_object, item_output_object):

    """
    Update ev hsm

    Parameters:
        created_hsm               (dict)          : created EV hsm
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    if post_hsm_flag and (item_output_object.get('created_hsm').get('ev') is None):
        item_output_object['created_hsm']['ev'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    # update ev hsm with ev postprocessing attributes

    if post_hsm_flag and (item_output_object.get('created_hsm').get('ev') is not None) and \
            (item_output_object.get('created_hsm').get('ev').get('attributes') is not None):
        item_output_object['created_hsm']['ev']['attributes'].update(created_hsm)

    return item_output_object
