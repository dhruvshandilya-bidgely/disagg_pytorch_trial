

"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Wrapper for seasonal signature detection
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.raw_energy_itemization.residual_analysis.detect_seasonal_sig import detect_hvac_appliances


def seasonal_sig_detection_wrapper(item_input_object, item_output_object,  weather_analytics, logger, logger_pass):

    """
    Wrapper for seasonal signature detection

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        weather_analytics         (dict)      : dict containing weather information
        logger                    (logger)    : logger object
        logger_pass               (dict)      : Contains base logger and logging dictionary

    Returns:
        detected_cool             (np.ndarray)  : detected cooling signature
        detected_heat             (np.ndarray)  : detected heating signature
        detected_wh               (np.ndarray)  : detected wh signature
        residual_data             (np.ndarray)  : updated disagg residual data
    """

    # Fetch required data

    sampling_rate = item_input_object.get("config").get("sampling_rate")
    input_data = item_output_object.get("hybrid_input_data").get("input_data")
    output_data = item_output_object.get("hybrid_input_data").get("output_data")
    residual_data = item_output_object.get("hybrid_input_data").get("original_res")
    appliance_list = item_output_object.get("hybrid_input_data").get("appliance_list")

    ref_index = np.where(np.array(appliance_list) == 'ref')[0][0] + 1
    ao_index = np.where(np.array(appliance_list) == 'ao')[0][0] + 1

    # removing baseload consumption before detecting seasonal signature

    input_without_baseload = input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :] - output_data[ao_index, :, :] - output_data[ref_index, :, :]

    # detecting and estimating seasonal signature that can be alloted to either cooling/heating/SWH

    detected_cool, detected_heat, detected_wh = \
        detect_hvac_appliances(item_input_object, item_output_object, sampling_rate, np.fmax(residual_data, 0),
                               appliance_list, output_data, weather_analytics, input_without_baseload, logger)

    if np.sum(detected_cool):
        logger.info("Found cooling signature in residual data | ")
    if np.sum(detected_heat):
        logger.info("Found heating signature in residual data | ")
    if np.sum(detected_wh):
        logger.info("Found seasonal WH signature in residual data | ")

    item_output_object["residual_detection"][1] = int(np.any(detected_cool > 0) or np.any(detected_heat > 0))

    hvac_dict = dict({
        "heating": detected_heat,
        "cooling": detected_cool,
        "wh": detected_wh
    })

    item_output_object.update({
        "hvac_dict": hvac_dict
    })

    residual_data = residual_data - detected_heat
    residual_data = residual_data - detected_cool

    return detected_cool, detected_heat, detected_wh, residual_data
