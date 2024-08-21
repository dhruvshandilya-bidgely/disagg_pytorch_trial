"""
Author - Mayank Sharan
Date - 23rd Jan 2019
Call the pool pump disaggregation module and get results
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.write_estimate import write_estimate
from python3.config.pilot_constants import PilotConstants
from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.poolpump.pp_disagg import pp_disagg
from python3.disaggregation.aer.poolpump.mtd_pp_disagg import mtd_pp_disagg
from python3.disaggregation.aer.poolpump.init_pp_params import init_pp_params

from python3.disaggregation.aer.poolpump.functions.get_poolpump_user_profile import get_poolpump_user_profile


def run_pp_bool(input_data, pp_config, global_config, logger_pp):

    """
    Parameters:
        input_data          (np.ndarray)        : 21 column matrix
        pp_config           (dict)              : Configuration parameters for PP
        global_config       (dict)              : Global configuration parameters
        logger_pp           (logger)            : Logger

    Returns:
        run_pp              (bool)              : Boolean indicating whether PP should be run or not
    """

    run_pp = True

    num_days = len(np.unique(input_data[:, Cgbdisagg.INPUT_DAY_IDX]))

    # Do not run PP in historical or incremental mode if data is available for less than min_days_to_run_pp

    if num_days < pp_config.get('min_days_to_run_pp') and (global_config.get('disagg_mode') == 'historical' or
                                                           global_config.get('disagg_mode') == 'incremental'):
        logger_pp.info('Not running pool pump since data less than {} days |'.format(pp_config.
                                                                                     get('min_days_to_run_pp')))
        run_pp = False

    return run_pp


def run_mtd_mode(hsm_in, is_python_hsm, pp_config, global_config, logger_pp):

    """Check if MTD mode should be run or not"""

    mtd_bool = True

    if global_config.get('disagg_mode') == 'mtd' and not is_python_hsm:
        logger_pp.info('Not running MTD mode since valid HSM not found | ')
        mtd_bool = False

    elif global_config.get('disagg_mode') == 'mtd' and is_python_hsm and \
            hsm_in.get('attributes').get('num_of_runs')[0] == 0:
        logger_pp.info('Not running MTD mode since number of runs is 0 in HSM | ')
        mtd_bool = False

    elif is_python_hsm and hsm_in.get('attributes').get('num_samples_per_hr')[0] != \
            int(Cgbdisagg.SEC_IN_HOUR / pp_config['sampling_rate']):
        logger_pp.info('Not running MTD mode since mismatch in sampling rates of HSM and MTD | ')
        mtd_bool = False

    return mtd_bool


def extract_pp_hsm(disagg_input_object):

    """Utility to extract pool pump hsm from disagg input object and confirm if it is python code"""

    hsm_dic = disagg_input_object.get('appliances_hsm')
    hsm_in = hsm_dic.get('pp')

    is_python_hsm = False

    if hsm_in is not None and hsm_in.get('attributes') is not None and hsm_in.get('attributes').get(
            'run_type_code') is not None:
        is_python_hsm = True

    return hsm_in, is_python_hsm


def check_pp_exists(disagg_input_object, global_config):
    """
    Function to check whether to write poolpump output to disagg output object
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        global_config       (dict)              : Global configurations
    Returns:
        pp_exists           (bool)              : Pool pump output write status
    """

    pp_exists = ((disagg_input_object.get('app_profile').get('pp') is not None and
                  disagg_input_object.get('app_profile').get('pp').get('number') > 0) or
                 global_config.get('pilot_id') in PilotConstants.PILOTS_TO_RUN_PP_FOR_ALL_USERS)

    return pp_exists


def write_pp_profile(hsm, pp_epoch, pp_confidence, disagg_input_object, disagg_output_object, logger_pass, logger_pp):

    """Writing pool pump user profile if disagg mode is not mtd"""

    if hsm is not None:
        pp_info = dict()
        pp_info['pp_epoch'] = pp_epoch
        pp_info['confidence_value'] = pp_confidence
        pp_info['run_type_code'] = hsm['attributes']['run_type_code'][0]
        pp_info['pp_runs'] = hsm['attributes']['num_of_runs'][0]

        schedules = hsm['attributes']['schedules']
        schedule_rows, schedule_cols = hsm['attributes']['num_schedules'][0], \
                                       hsm['attributes']['num_schedule_params'][
                                           0]

        if hsm.get('attributes').get('run_type_code')[0] != 0:
            pp_info['schedules'] = np.reshape(schedules, newshape=(schedule_rows, schedule_cols))
            logger_pp.info("Pool pump schedule matrix : | {}".format(str(pp_info['schedules']).replace('\n', '; ')))
        else:
            pp_info['schedules'] = np.zeros(0)
            logger_pp.info("Pool pump schedule matrix is Null since detection is 0 | ")

        disagg_output_object = get_poolpump_user_profile(disagg_input_object, disagg_output_object, pp_info,
                                                         logger_pass)
    else:
        logger_pp.info('Poolpump Profile Fill skipped, HSM not available')

    return disagg_output_object


def pool_pump_disagg_wrapper(disagg_input_object, disagg_output_object):

    """
    Parameters:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs

    Returns:
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the pool_pump module

    logger_pp_base = disagg_input_object.get('logger').getChild('pool_pump_disagg_wrapper')
    logger_pp = logging.LoggerAdapter(logger_pp_base, disagg_input_object.get('logging_dict'))

    logger_pass = {
        'logger_base' : logger_pp_base,
        'logging_dict': disagg_input_object.get('logging_dict'),
    }

    t_pp_start = datetime.now()

    # Initialise arguments to give to the disagg code

    global_config = disagg_input_object.get('config')
    input_data = copy.deepcopy(disagg_input_object.get('input_data'))

    pp_config = init_pp_params()
    pp_config['uuid'] = global_config.get('uuid')
    pp_config['sampling_rate'] = global_config.get('sampling_rate')
    pp_config['pilot_id'] = global_config.get('pilot_id')

    # Initialize outputs so that we don't get errors

    monthly_pp = np.array([])
    pp_epoch = np.array([])

    exit_status = {
        'exit_code' : 1,
        'error_list': [],
    }

    hsm_in, is_python_hsm = extract_pp_hsm(disagg_input_object)

    # Check if PP should be run

    run_pp = run_pp_bool(input_data, pp_config, global_config, logger_pp)
    run_mtd = run_mtd_mode(hsm_in, is_python_hsm, pp_config, global_config, logger_pp)

    # TODO(Mayank): remove this and add everything in new appliance profile
    user_profile_object = {}

    # Run pool pump disagg with different inputs as per requirement as for different modes

    # Code flow can split here to accommodate separate run options for prod and custom

    if global_config.get('disagg_mode') == 'historical' and run_pp:

        logger_pp.info('Running historical mode |')

        # Call the pp disagg module

        monthly_pp, ts, pp_cons, hsm, user_profile_object = \
            pp_disagg(input_data, user_profile_object, pp_config, logger_pass)

        disagg_output_object['created_hsm']['pp'] = hsm

        ts_1d = np.reshape(ts, newshape=(len(ts) * len(ts[0]),))

        pp_1d = pp_cons
        pp_1d = np.reshape(pp_1d, newshape=(len(pp_1d) * len(pp_1d[0]),))

        pp_epoch = np.c_[ts_1d, pp_1d]

    elif global_config.get('disagg_mode') == 'incremental' and run_pp:

        logger_pp.info('Running incremental mode |')

        # Call the pp disagg module

        monthly_pp, ts, pp_cons, hsm, user_profile_object = \
            pp_disagg(input_data, user_profile_object, pp_config, logger_pass)

        disagg_output_object['created_hsm']['pp'] = hsm

        ts_1d = np.reshape(ts, newshape=(len(ts) * len(ts[0]),))

        pp_1d = pp_cons
        pp_1d = np.reshape(pp_1d, newshape=(len(pp_1d) * len(pp_1d[0]),))

        pp_epoch = np.c_[ts_1d, pp_1d]

    elif global_config.get('disagg_mode') == 'mtd' and run_mtd:

        logger_pp.info('Running mtd mode |')

        monthly_pp, ts, pp_cons, hsm, user_profile_object = \
            mtd_pp_disagg(input_data, user_profile_object, pp_config, hsm_in, logger_pass)

        ts_1d = np.reshape(ts, newshape=(len(ts) * len(ts[0]),))

        pp_1d = pp_cons
        pp_1d = np.reshape(pp_1d, newshape=(len(pp_1d) * len(pp_1d[0]),))

        pp_epoch = np.c_[ts_1d, pp_1d]

    elif not (global_config.get('disagg_mode') in ['historical', 'incremental', 'mtd']):
        logger_pp.error('Unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    else:
        hsm = None

    pp_exists = check_pp_exists(disagg_input_object, global_config)

    if len(monthly_pp) > 0 and len(pp_epoch) > 0 and pp_exists:
        # Code to write results to disagg output object

        pp_out_idx = disagg_output_object.get('output_write_idx_map').get('pp')
        pp_read_idx = 1

        # Writing the monthly output to log

        monthly_output_log = [(datetime.utcfromtimestamp(monthly_pp[i, 0]).strftime('%b-%Y'),
                               monthly_pp[i, 1]) for i in range(monthly_pp.shape[0])]

        logger_pp.info("The monthly pool pump consumption (in Wh) is : | %s",
                       str(monthly_output_log).replace('\n', ' '))

        # Write billing cycle estimate

        disagg_output_object = write_estimate(disagg_output_object, monthly_pp, pp_read_idx,
                                              pp_out_idx, 'bill_cycle')

        # Write timestamp level estimates

        disagg_output_object = write_estimate(disagg_output_object, pp_epoch, pp_read_idx, pp_out_idx,
                                              'epoch')

    t_pp_end = datetime.now()

    logger_pp.info('Pool Pump Estimation took | %.3f s ', get_time_diff(t_pp_start, t_pp_end))

    # Write exit status time taken etc.

    disagg_metrics_dict = {
        'time'       : get_time_diff(t_pp_start, t_pp_end),
        'confidence' : 1.0,
        'exit_status': exit_status,
    }

    disagg_output_object['disagg_metrics']['pp'] = disagg_metrics_dict

    # Writing pool pump output to special_outputs to be used in Australia_pilots or otherwise

    pp_consumption = np.c_[input_data[:, Cgbdisagg.INPUT_EPOCH_IDX],
                           np.zeros_like(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])]
    pp_confidence = 0
    pp_hybrid_confidence = 0

    if len(monthly_pp) > 0 and len(pp_epoch) > 0:
        vals, idx1, idx2 = np.intersect1d(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], pp_epoch[:, 0], return_indices=True)
        pp_consumption = pp_epoch[idx2]
        pp_confidence = hsm.get('attributes').get('confidence')
        pp_hybrid_confidence = hsm.get('attributes').get('hybrid_confidence')

        if isinstance(pp_confidence, list):
            pp_confidence = pp_confidence[0]

        if isinstance(pp_hybrid_confidence, list):
            pp_hybrid_confidence = pp_hybrid_confidence[0]

    disagg_output_object['special_outputs']['pp_consumption'] = pp_consumption
    disagg_output_object['special_outputs']['pp_confidence'] = pp_confidence
    disagg_output_object['special_outputs']['pp_hybrid_confidence'] = pp_hybrid_confidence

    # Writing pool pump user profile if disagg mode is not mtd

    disagg_mode = global_config.get('disagg_mode')

    if not (disagg_mode == 'mtd'):

        disagg_output_object = \
            write_pp_profile(hsm, pp_epoch, pp_confidence, disagg_input_object, disagg_output_object, logger_pass, logger_pp)

    return disagg_input_object, disagg_output_object
