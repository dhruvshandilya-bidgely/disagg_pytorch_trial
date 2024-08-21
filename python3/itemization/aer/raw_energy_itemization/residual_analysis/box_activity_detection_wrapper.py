
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Master file for itemization pipeline
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.residual_analysis.config.get_residual_config import get_residual_config

from python3.itemization.aer.raw_energy_itemization.residual_analysis.detect_box_type_cons import box_detection


def box_activity_detection_wrapper(item_input_object, item_output_object, logger):

    """
    Prepare hybrid input object

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object
        logger_pass               (dict)      : Contains base logger and logging dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    pilot = item_input_object.get("config").get("pilot_id")
    disagg_residual = item_output_object.get("hybrid_input_data").get("true_disagg_res")

    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]
    samples_per_hour = int(len(original_input_data[0]) / Cgbdisagg.HRS_IN_DAY)

    ev_app_profile = item_input_object.get("app_profile").get('ev')

    if ev_app_profile is not None:
        ev_app_profile = ev_app_profile.get("number", 0)
    else:
        ev_app_profile = 0

    pp_app_profile = item_input_object.get("app_profile").get('pp')

    if pp_app_profile is not None:
        pp_app_profile = pp_app_profile.get("number", 0)
    else:
        pp_app_profile = 0

    config = get_residual_config(samples_per_hour).get('box_detection_config')

    # If ev or pp app profile is present, we detect activity of duration more than 4 hours

    if (pp_app_profile or ev_app_profile):
        max_len = config.get('max_box_len')[1]
    else:
        max_len = config.get('max_box_len')[0]

    min_box_amp = config.get('min_box_amp')
    max_box_amp = config.get('max_box_amp')
    min_box_len = config.get('min_box_len')
    min_box_len_for_wh_boxes = config.get('min_box_len_for_wh_boxes')
    min_len_for_using_disagg_residual = config.get('min_len_for_using_disagg_residual')

    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")
    wh_idx = np.where(np.array(appliance_list) == 'wh')[0][0] + 1

    # if days count is less than 180, pick activity from original input data
    # else pick activity from residual data

    if len(original_input_data) > min_len_for_using_disagg_residual:
        box_label, box_cons, box_seq = \
            box_detection(pilot, original_input_data, np.fmax(0, disagg_residual), disagg_residual < 0,
                          min_amp=min_box_amp, max_amp=max_box_amp,
                          min_len=min_box_len, max_len=max_len)

    else:
        box_label, box_cons, box_seq = \
            box_detection(pilot, original_input_data, np.fmax(0, original_input_data), original_input_data < 0,
                          min_amp=min_box_amp, max_amp=max_box_amp,
                          min_len=min_box_len, max_len=max_len)

    pilot = item_input_object.get("config").get("pilot_id")
    disagg_residual = item_output_object.get("hybrid_input_data").get("true_disagg_res")

    if len(original_input_data) > min_len_for_using_disagg_residual and np.sum(output_data[wh_idx]) > 0:

        box_label_wh, box_cons_wh, box_seq_wh = \
            box_detection(pilot, original_input_data, np.fmax(0, disagg_residual), disagg_residual < 0,
                          min_amp=min_box_amp, max_amp=max_box_amp,
                          min_len=min_box_len, max_len=min_box_len_for_wh_boxes, detect_wh=1)

    else:
        box_label_wh, box_cons_wh, box_seq_wh = \
            box_detection(pilot, original_input_data, np.fmax(0, original_input_data), original_input_data < 0,
                          min_amp=min_box_amp, max_amp=max_box_amp,
                          min_len=min_box_len, max_len=min_box_len_for_wh_boxes, detect_wh=1)

    return box_label, box_cons, box_seq, box_label_wh, box_cons_wh, box_seq_wh
