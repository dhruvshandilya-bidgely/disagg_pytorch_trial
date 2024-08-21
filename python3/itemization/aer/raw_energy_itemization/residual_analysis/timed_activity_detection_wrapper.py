
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.raw_energy_itemization.residual_analysis.detect_timed_sig import detect_timed_appliance


def timed_activity_detection_wrapper(item_input_object, item_output_object, logger, logger_pass):

    """
    wrapper function for timed signature detection

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object
        logger_pass               (dict)      : Contains base logger and logging dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    input_data = item_output_object.get("hybrid_input_data").get("input_data")
    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    residual_copy = item_output_object.get("hybrid_input_data").get("original_res")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    ref_index = np.where(np.array(appliance_list) == 'ref')[0][0] + 1
    ao_index = np.where(np.array(appliance_list) == 'ao')[0][0] + 1

    ao_cons = output_data[ao_index]
    ref_cons = output_data[ref_index]

    # Run timed signature detection

    final_residual, twh_cons, pp_cons, heating_cons, timed_output = \
        detect_timed_appliance(input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :] - ao_cons - ref_cons,
                               item_input_object, item_output_object, logger_pass)

    timed_app_dict = dict({
        "twh": twh_cons,
        "pp": pp_cons,
        "heating": heating_cons,
        "timed_output": timed_output
    })

    logger.debug('Calculated timed signature | ')

    # update PP HSM attributes for new detected appliances

    item_input_object, item_output_object = update_pp_hsm(item_input_object, item_output_object, pp_cons, logger)

    # update TWH HSM attributes for new detected appliances

    item_input_object, item_output_object = update_wh_hsm(item_input_object, item_output_object, twh_cons, logger)

    logger.debug('updated hsm with timed signature attributes | ')

    residual_copy = residual_copy - (pp_cons + twh_cons + heating_cons)

    timed_sig_detected = np.any(twh_cons > 0) or np.any(pp_cons > 0) or np.any(heating_cons > 0)

    if timed_sig_detected:
        item_output_object["residual_detection"][0] = 1

    # updating timed wh flag if TWH is detected in hybrid v2

    if np.sum(twh_cons) > 0:
        item_input_object["item_input_params"]["timed_wh_user"] = 1

    return timed_app_dict, residual_copy, item_input_object, item_output_object


def update_pp_hsm(item_input_object, item_output_object, pp_cons, logger):

    """
    update existing pp hsm

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        pp_cons                   (np.ndarray): pp output
        logger                    (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    created_hsm = dict({
        'item_tou': np.zeros(len(pp_cons[0])),
        'item_hld': 0,
        'item_conf': 0,
        'item_amp': 0,
        'final_item_amp': 0,
    })

    perc_cap = 90

    samples = int(pp_cons.shape[1]/Cgbdisagg.HRS_IN_DAY)

    # preparing values to be posted in PP HSM if PP is estimated from hybrid timed signature detection

    if np.sum(pp_cons > 0) > 0:
        created_hsm['item_tou'] = np.sum(pp_cons, axis=0) > 0
        created_hsm['item_amp'] = np.percentile(pp_cons[pp_cons > 0], perc_cap) * samples
        created_hsm['final_item_amp'] = np.percentile(pp_cons[pp_cons > 0], perc_cap) * samples
        created_hsm['item_conf'] = 1
        created_hsm['item_hld'] = 1

    # preparing values to be posted in PP HSM if PP is not detected

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    pp_absent_flag = item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
                     item_input_object.get("item_input_params").get('pp_hsm') is not None and \
                     item_input_object.get("item_input_params").get('pp_hsm').get('item_amp') is not None and (np.sum(pp_cons) == 0)

    if pp_absent_flag:
        created_hsm['item_amp'] = 0
        created_hsm['final_item_amp'] = 0
        created_hsm['item_tou'] = 0
        created_hsm['item_conf'] = 0
        created_hsm['item_hld'] = 0

    pp_hsm_key_present = post_hsm_flag and item_output_object.get('created_hsm').get('pp') is None

    # updating PP HSM with new hybrid parameters

    if pp_hsm_key_present:
        item_output_object['created_hsm']['pp'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    pp_hsm_attributes_present = post_hsm_flag and  (item_output_object.get('created_hsm') is not None) and \
                                (item_output_object.get('created_hsm').get('pp') is not None) and \
                                (item_output_object.get('created_hsm').get('pp').get('attributes') is not None)

    if pp_hsm_attributes_present:
        item_output_object['created_hsm']['pp']['attributes'].update(created_hsm)
        logger.debug("Updating PP HSM in Hybrid module | ")

    return item_input_object, item_output_object


def update_wh_hsm(item_input_object, item_output_object, twh_cons, logger):

    """
    update existing pp hsm

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        twh_cons                  (np.ndarray): TWH disagg output
        logger                    (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    created_hsm = dict({
        'item_tou': np.zeros(len(twh_cons[0])),
        'item_hld': 0,
        'item_type': 0,
        'item_amp': 0
    })

    config = get_inf_config().get("wh")
    min_days_required_for_hsm_posting = config.get('min_days_required_for_hsm_posting')

    # preparing values to be posted in TWH HSM

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    if np.sum(twh_cons) > 0:
        created_hsm['item_tou'] = np.sum(twh_cons > 0, axis=0) > min_days_required_for_hsm_posting
        created_hsm['item_amp'] = np.median(twh_cons[twh_cons > 0])
        created_hsm['item_hld'] = 1
        created_hsm['item_type'] = 1
        created_hsm['item_conf'] = 1

        # updating TWH HSM with new hybrid parameters

        twh_hsm_present = post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is None)

        if twh_hsm_present:
            item_output_object['created_hsm']['wh'] = {
                'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
                'attributes': dict()
            }

        twh_hsm_attributes_present = \
            post_hsm_flag and (item_output_object.get('created_hsm').get('wh') is not None) and \
            (item_output_object.get('created_hsm').get('wh').get('attributes') is not None)

        if twh_hsm_attributes_present:
            item_output_object['created_hsm']['wh']['attributes'].update(created_hsm)
            logger.debug("Updating TWH HSM in Hybrid module | ")

    return item_input_object, item_output_object
