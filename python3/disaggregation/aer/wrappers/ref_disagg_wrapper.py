"""
Author - Pratap Dangeti
Date - 30/OCT/18
Call the refrigerator disaggregation module and get results
"""

# Import python packages

import copy
import logging
import numpy as np
from scipy import stats
from datetime import datetime

# Import functions from within the project

from python3.utils.write_estimate import write_estimate

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.refrigerator.get_ref_estimation import get_ref_estimation
from python3.disaggregation.aer.refrigerator.functions.get_ref_user_profile import get_ref_profile
from python3.disaggregation.aer.refrigerator.functions.initialize_gbref_params import initialize_gbref_params

from python3.utils.validate_output_schema import validate_appliance_profile_schema_for_billcycle


def check_ref_hsm(hsm_in, disagg_input_object):

    """
    This function checks the validity of ref hsm
    Parameters:
        hsm_in                  (dict)              : Ref hsm
        disagg_input_object     (dict)              : Dictionary containing all inputs
    Returns:
        hsm_fail                (bool)              : flag that represents whether hsm checks conditions failed
    """

    global_config = disagg_input_object.get('config')

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               ((global_config.get("disagg_mode") == "incremental") or (global_config.get("disagg_mode") == "mtd"))

    if (hsm_in is not None) and \
        (hsm_in.get('attributes') is not None) and \
        (hsm_in.get('attributes').get('Ref_Energy_Per_DataPoint') is None) and \
               ((global_config.get("disagg_mode") == "incremental") or (global_config.get("disagg_mode") == "mtd")):
        hsm_fail = True

    return hsm_fail


def ref_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the lighting module

    logger_ref_base = disagg_input_object.get("logger").getChild("ref_disagg_wrapper")
    logger_pass = {"logger"      : logger_ref_base,
                   "logging_dict": disagg_input_object.get("logging_dict")}
    logger_ref = logging.LoggerAdapter(logger_ref_base, disagg_input_object.get("logging_dict"))

    t_ref_start = datetime.now()

    # Initialise arguments to give to the disagg code
    error_list = []

    global_config = disagg_input_object.get('config')
    if global_config is None:
        error_list.append('Key Error: config does not exist')

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))

    if input_data is None:
        error_list.append('Key Error: input data does not exist')

    is_nan_cons = disagg_input_object['data_quality_metrics']['is_nan_cons']
    input_data[is_nan_cons, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.nan

    sampling_rate = disagg_input_object.get('config').get('sampling_rate')

    ref_config = initialize_gbref_params(sampling_rate)
    # check here for dict format
    ref_config['UUID'] = global_config.get('uuid')

    exit_status = {
        'exit_code' : 1,
        'error_list': [],
    }

    # noinspection PyBroadException
    try:
        hsm_dic = disagg_input_object.get("appliances_hsm")
        hsm_in = hsm_dic.get("ref")
    except KeyError:
        hsm_in = None

    hsm_fail = check_ref_hsm(hsm_in, disagg_input_object)

    # Run ref disagg with different inputs as per requirement as for different modes
    # Code flow can split here to accommodate separate run options for prod and custom

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom') and not hsm_fail:
        if global_config.get('disagg_mode') == 'historical':

            # Flags related to HSM
            # bypass_hsm: Whether to skip using HSM or not
            # make_hsm: Whether to push new HSM or not

            bypass_hsm = True
            make_hsm = True
            hsm_in = {}

            ref_detection = get_ref_estimation(input_data, ref_config, make_hsm, hsm_in, bypass_hsm, logger_pass)
            disagg_output_object['created_hsm']['ref'] = ref_detection.get('hsm')

        elif global_config.get('disagg_mode') == 'incremental':

            # Flags related to HSM
            # bypass_hsm: Whether to skip using HSM or not
            # make_hsm: Whether to push new HSM or not

            bypass_hsm = False
            make_hsm = True

            ref_detection = get_ref_estimation(input_data, ref_config, make_hsm, hsm_in, bypass_hsm, logger_pass)
            disagg_output_object['created_hsm']['ref'] = ref_detection.get('hsm')

        elif global_config.get('disagg_mode') == 'mtd':

            # Flags related to HSM
            # bypass_hsm: Whether to skip using HSM or not
            # make_hsm: Whether to push new HSM or not

            bypass_hsm = False
            make_hsm = False

            ref_detection = get_ref_estimation(input_data, ref_config, make_hsm, hsm_in, bypass_hsm, logger_pass)

        else:
            ref_detection = None

            logger_ref.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    if not hsm_fail:

        # Monthly Ref output
        monthly_ref = ref_detection['monthRef']
        ref_epoch = ref_detection['refHourlyOutput'][:, 1:]

        # Code to write results to disagg output object
        # double check the below one during final stage

        ref_out_idx = disagg_output_object.get('output_write_idx_map').get('ref')

        # Write billing cycle estimate

        disagg_output_object = write_estimate(disagg_output_object, monthly_ref, 1, ref_out_idx, 'bill_cycle')

        # Write timestamp level estimates

        disagg_output_object = write_estimate(disagg_output_object, ref_epoch, 1, ref_out_idx, 'epoch')
    else:
        ref_detection = None
        logger_ref.warning("Ref did not run because %s mode needs HSM and it is missing |",
                           global_config.get("disagg_mode"))

    t_ref_end = datetime.now()

    logger_ref.info('Ref Estimation took | %.3f s ', get_time_diff(t_ref_start, t_ref_end))

    # Add user attributes to the user profile object

    disagg_mode = global_config.get('disagg_mode')

    if not (disagg_mode == 'mtd'):
        disagg_output_object = get_ref_profile(disagg_input_object, disagg_output_object, logger_pass,
                                               ref_detection)

    disagg_metrics_dict = {
        'time'       : get_time_diff(t_ref_start, t_ref_end),
        'confidence' : 1.0,
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['ref'] = disagg_metrics_dict

    return disagg_input_object, disagg_output_object
