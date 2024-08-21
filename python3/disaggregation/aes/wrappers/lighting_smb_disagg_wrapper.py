"""
Author: Neelabh Goyal
Date: 14 June 2023
Call the lighting disaggregation module and get results
"""

# Import python packages

import copy
import timeit
import logging
import numpy as np

# Import from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.lighting.get_hourglass_params import get_hourglass_params
from python3.disaggregation.aes.lighting.detect_hourglass_pattern import get_hourglass_pattern
from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import prepare_input_df
from python3.disaggregation.aes.lighting.hourglass_lighting_estimation import external_lighting_estimation


def lighting_smb_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Wrapper for Lighting Disagg
    Parameters:
        disagg_input_object  (dict): Dictionary with pipeline level input data
        disagg_output_object (dict): Dictionary containing pipeline level output data

    Returns:
        disagg_input_object  (dict): Dictionary with pipeline level input data
        disagg_output_object (dict): Dictionary containing pipeline level output data
    """

    # Initiate logger for the ao module
    logger_li_base = disagg_input_object.get('logger').getChild('lighting_smb_disagg_wrapper')
    logger_li = logging.LoggerAdapter(logger_li_base, disagg_input_object.get('logging_dict'))
    logger_li_pass = {"logger": logger_li_base, "logging_dict": disagg_input_object.get("logging_dict")}

    params = get_hourglass_params()

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    input_df = prepare_input_df(input_data, disagg_input_object)
    disagg_input_object['input_df'] = input_df

    start = timeit.default_timer()
    hourglass_dict, hourglass_data = get_hourglass_pattern(input_df, params, logger_li_pass)
    hourglass_dict['sampling'] = int(Cgbdisagg.SEC_IN_HOUR / disagg_input_object.get('config').get('sampling_rate'))
    disagg_output_object['hourglass_data'] = hourglass_data
    end_time = timeit.default_timer()
    logger_li.info(' Hourglass pattern for external light detected | {}'.format(hourglass_dict["Bool"]))
    logger_li.info(' Overall Hourglass Pattern detection took | {} s'.format(round(end_time - start, 3)))

    if hourglass_dict['Bool']:
        logger_li.info(' Starting with External Light estimation |')
        start = timeit.default_timer()

        base_load = disagg_output_object['ao_seasonality']['epoch_baseload']
        ao_hvac = disagg_output_object['ao_seasonality']['epoch_cooling'] + \
                  disagg_output_object['ao_seasonality']['epoch_heating']

        input_df['raw-ao'] = input_df['consumption'] - base_load
        logger_li.info(' Removed base load before external lighting estimation | ')

        external_light = external_lighting_estimation(input_df, hourglass_dict, ao_hvac, params, logger_li_pass)
        external_light = np.nan_to_num(external_light).reshape(-1, 1)[:, 0]

        epoch_df = input_df.pivot_table(index='date', columns=['time'], values='epoch', aggfunc=np.min)
        epochs = epoch_df.values.flatten()
        external_light = external_light.flatten()
        _, idx_mem_1, idx_mem_2 = np.intersect1d(epochs, input_df['epoch'], return_indices=True)
        write_idx = disagg_output_object.get('output_write_idx_map').get('li_smb')
        disagg_output_object['epoch_estimate'][idx_mem_2, write_idx] = external_light[idx_mem_1]

        end_time = timeit.default_timer()
        logger_li.info(' Overall external light Estimation Took | {} seconds'.format(round(end_time - start, 3)))

    return disagg_input_object, disagg_output_object
