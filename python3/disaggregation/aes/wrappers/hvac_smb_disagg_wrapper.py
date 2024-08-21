"""
Author - Abhinav Srivastava
Date - 22nd Oct 2018
Call the hvac disaggregation module and get results
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.write_estimate import write_estimate
from python3.utils.time.get_time_diff import get_time_diff
from python3.disaggregation.aes.work_hours.smb_work_hours import get_smb_components

from python3.disaggregation.aer.hvac.hvac_utility import read_hsm_in
from python3.disaggregation.aes.hvac.plot_utils_smb_hvac import bar_appmap_baseline
from python3.disaggregation.aer.hvac.hvac_utility import write_ao_od_hvac_at_epoch
from python3.disaggregation.aer.hvac.hvac_utility import write_analytics_month_and_epoch_results

from python3.disaggregation.aes.hvac.smb_hvac import smb_hvac
from python3.disaggregation.aes.hvac.get_residue_from_hvac import get_residues
from python3.disaggregation.aes.hvac.populate_lifestyle import populate_smb_lifestyle
from python3.disaggregation.aes.hvac.fill_smb_userprofile import fill_user_profile_smb
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import hvac_static_params
from python3.disaggregation.aes.hvac.postprocess_hvac_smb import postprocess_results_smb


def hvac_control_centre(disagg_input_object):
    """
       Function to control the dumping of epoch and month level estimates of HVAC

       Parameters:

           disagg_input_object (dict) : Dictionary containing all the inputs

       Returns:
            None
       """

    # Initialize switch key in disagg input object

    static_params = hvac_static_params()

    config = disagg_input_object.get('config')

    # Initialise the switch key if not already created in the disagg_input_object
    if disagg_input_object.get('switch', None) is None:
        disagg_input_object['switch'] = {}

    if not config.get('generate_plots') is None:
        disagg_input_object['switch']['plot_level'] = len(config.get('generate_plots'))

    else:
        disagg_input_object['switch']['plot_level'] = 0

    hour_aggregate_level = static_params['hour_aggregate_level']

    if disagg_input_object['config']['pilot_id'] == static_params['hour_aggregate_level']:
        hour_aggregate_level = static_params['fpl_id']

    # SMB : added for smb control
    if config.get('user_type').lower() == 'smb':
        if disagg_input_object.get('switch').get('smb', None) is None:
            disagg_input_object['switch']['smb'] = {}

        disagg_input_object['switch']['smb']['flag_vector'] = np.ones(disagg_input_object['input_data'].shape[0])

    disagg_input_object['switch']['hvac'] = {'metrics': False,
                                             'dump_metrics': False,

                                             'epoch_estimate': False,
                                             'month_estimate': False,

                                             'epoch_od_ao_hvac_dump': False,
                                             'hour_aggregate_level': hour_aggregate_level
                                             }

    disagg_input_object['switch']['residue'] = False


def run_smb_hvac(global_config, disagg_input_object, disagg_output_object, logger_pass, logger_hvac, hvac_exit_status):
    """
    Parameters:
        global_config (dict)                    : Dictionary containing user level config info
        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
        logger_pass(logging object)             : Writes logs during code flow
        logger_hvac(logging object)             : Writes logs during code flow
        hvac_exit_status (dict)                 : Dictionary containing hvac exit status code

    Returns:
        month_ao_hvac_true (np.ndarray)         : Array containing month level ao and hvac consumption
        epoch_ao_hvac_true (np.ndarray)         : Array containing epoch level ao and hvac consumption
        debug (dict)                            : Dictionary containing hvac debug information
        hsm_update(dict)                        : Dictionary containing hsm update information
        hvac_exit_status (dict)                 : Dictionary containing hvac exit status code
    """

    epoch_ao_hvac_true = np.array([])
    month_ao_hvac_true = np.array([])
    hsm_update = []
    debug = {}

    if global_config.get('disagg_mode') == 'historical':
        logger_hvac.info("Running historical mode : |")

        month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status = \
            smb_hvac(disagg_input_object, disagg_output_object, logger_pass, hvac_exit_status)
        disagg_output_object['created_hsm']['hvac'] = hsm_update
        disagg_output_object['hvac_debug'] = debug

    elif global_config.get('disagg_mode') == 'incremental':
        logger_hvac.info("Running incremental mode : |")

        month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status = \
            smb_hvac(disagg_input_object, disagg_output_object, logger_pass, hvac_exit_status)
        disagg_output_object['created_hsm']['hvac'] = hsm_update
        disagg_output_object['hvac_debug'] = debug

    elif global_config.get('disagg_mode') == 'mtd':
        logger_hvac.info("Running mtd mode : |")

        month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status = \
            smb_hvac(disagg_input_object, disagg_output_object, logger_pass, hvac_exit_status)
        hsm_update = []

    else:
        logger_hvac.error('unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    return month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status


def hvac_smb_disagg_wrapper(disagg_input_object, disagg_output_object):
    """
    Parameters:

        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    Returns:

        disagg_input_object (dict)              : Dictionary containing all inputs
        disagg_output_object(dict)              : Dictionary containing all outputs
    """

    # Initiate logger for the hvac module
    logger_hvac_base = disagg_input_object.get('logger').getChild('hvac_disagg_wrapper')
    logger_pass = {"logger": logger_hvac_base, "logging_dict": disagg_input_object.get("logging_dict")}
    logger_hvac = logging.LoggerAdapter(logger_hvac_base, disagg_input_object.get('logging_dict'))

    hvac_control_centre(disagg_input_object)

    t_hvac_start = datetime.now()
    error_list = []

    global_config = disagg_input_object.get('config')

    if global_config is None:
        error_list.append('Key Error: config does not exist')

    hvac_exit_status = {
        'exit_code': -1,
        'error_list': [],
    }

    hsm_in = read_hsm_in(disagg_input_object, logger_hvac)

    hsm_update = []

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (global_config.get("disagg_mode") == "mtd")

    # Initializing monthly and epoch level hvac consumption arrays as failsafe
    epoch_ao_hvac_true = np.array([])
    month_ao_hvac_true = np.array([])

    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom') and (not hsm_fail):
        logger_hvac.info('user is smb. hvac run mode is | %s', global_config.get('run_mode'))

        month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status = run_smb_hvac(global_config,
                                                                                                   disagg_input_object,
                                                                                                   disagg_output_object,
                                                                                                   logger_pass,
                                                                                                   logger_hvac,
                                                                                                   hvac_exit_status)

    column_index = {'ao': 1, 'ac': 2, 'sh': 3}

    # THIS IS REDUNDANT

    if len(hsm_update) > 0:
        disagg_output_object['created_hsm']['hvac'] = hsm_update

    generate_plot = ('hvac' in disagg_input_object['config']['generate_plots']) or \
                    ('all' in disagg_input_object['config']['generate_plots'])

    disagg_mode = global_config.get('disagg_mode')

    if not hsm_fail:

        # Code to write results to disagg output object
        ao_out_idx = disagg_output_object.get('output_write_idx_map').get('ao_smb')
        hvac_out_idx = disagg_output_object.get('output_write_idx_map').get('hvac_smb')
        ac_out_idx = hvac_out_idx[0]
        sh_out_idx = hvac_out_idx[1]

        # =========================================================================================================== #
        month_ao_hvac_res_net = get_residues(disagg_input_object, global_config, disagg_output_object,
                                             month_ao_hvac_true, column_index, logger_pass)
        # =========================================================================================================== #
        bar_appmap_baseline(generate_plot, disagg_input_object, disagg_output_object, month_ao_hvac_res_net,
                            column_index, epoch_ao_hvac_true)
        # =========================================================================================================== #

        logger_hvac.info('User is SMB. Getting appliance results | ')
        month_ao_hvac_res_net, epoch_ao_hvac_true = postprocess_results_smb(global_config, month_ao_hvac_res_net,
                                                                            epoch_ao_hvac_true, disagg_input_object,
                                                                            disagg_output_object, column_index,
                                                                            logger_pass)

        month_ao_hvac_res_net, epoch_ao_hvac_true = get_smb_components(global_config, month_ao_hvac_res_net,
                                                                       epoch_ao_hvac_true, disagg_input_object,
                                                                       disagg_output_object, column_index, logger_pass)

        if disagg_mode != 'mtd':
            populate_smb_lifestyle(disagg_input_object, disagg_output_object)

        # write_smb_analytics(disagg_input_object, disagg_output_object, disagg_output_object['analytics']['values'])

        column_index['x-ao'] = 4
        column_index['ac_open'] = 5
        column_index['ac_close'] = 6
        column_index['sh_open'] = 7
        column_index['sh_close'] = 8
        column_index['op'] = 9

        month_ao_hvac_res_net_backup = copy.deepcopy(month_ao_hvac_res_net)
        month_ao_hvac_true = copy.deepcopy(month_ao_hvac_res_net)

        # deleting 4th and 5th columns (axis=1) to get rid of res-net
        month_ao_hvac_true = np.delete(month_ao_hvac_true, [4, 5], 1)
        scale_arr = month_ao_hvac_true[:, 1:]
        scaled_arr = scale_arr * Cgbdisagg.WH_IN_1_KWH
        month_ao_hvac_true[:, 1:] = scaled_arr

        if disagg_mode != 'mtd':
            disagg_output_object['hvac_debug']['write'] = {}
            disagg_output_object['hvac_debug']['write']['column_dentify'] = column_index
            disagg_output_object['hvac_debug']['write']['month_ao_hvac_true'] = month_ao_hvac_true
            disagg_output_object['hvac_debug']['write']['month_ao_hvac_res_net'] = month_ao_hvac_res_net_backup
            disagg_output_object['hvac_debug']['write']['epoch_ao_hvac_true'] = epoch_ao_hvac_true

        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['ao'],
                                              ao_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['ac'],
                                              ac_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['sh'],
                                              sh_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['ao'],
                                              ao_out_idx, 'bill_cycle')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['ac'],
                                              ac_out_idx, 'bill_cycle')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['sh'],
                                              sh_out_idx, 'bill_cycle')

        x_ao_out_idx = disagg_output_object.get('output_write_idx_map').get('x-ao')
        ac_open_out_idx = disagg_output_object.get('output_write_idx_map').get('ac_open')
        ac_close_out_idx = disagg_output_object.get('output_write_idx_map').get('ac_close')
        sh_open_out_idx = disagg_output_object.get('output_write_idx_map').get('sh_open')
        sh_close_out_idx = disagg_output_object.get('output_write_idx_map').get('sh_close')
        op_out_idx = disagg_output_object.get('output_write_idx_map').get('op')

        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['x-ao'],
                                              x_ao_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['ac_open'],
                                              ac_open_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['ac_close'],
                                              ac_close_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['sh_open'],
                                              sh_open_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['sh_close'],
                                              sh_close_out_idx, 'epoch')
        disagg_output_object = write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['op'],
                                              op_out_idx, 'epoch')

        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['x-ao'],
                                              x_ao_out_idx, 'bill_cycle')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['ac_open'],
                                              ac_open_out_idx, 'bill_cycle')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['ac_close'],
                                              ac_close_out_idx, 'bill_cycle')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['sh_open'],
                                              sh_open_out_idx, 'bill_cycle')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['sh_close'],
                                              sh_close_out_idx, 'bill_cycle')
        disagg_output_object = write_estimate(disagg_output_object, month_ao_hvac_true, column_index['op'],
                                              op_out_idx, 'bill_cycle')

        # Writing the monthly output to log

        monthly_output_written = disagg_output_object['bill_cycle_estimate']
        all_bill_cycles = monthly_output_written[:, 0]
        out_bill_cycles = disagg_input_object['out_bill_cycles'][:, 0]
        monthly_output_written = monthly_output_written[np.isin(all_bill_cycles, out_bill_cycles), :]

        logger_hvac.info(">> SMB : Month list = {} ".format(monthly_output_written[:, 0].astype(int).tolist()))

        monthly_output_log = [((int(monthly_output_written[i, 0])),
                               monthly_output_written[i, ao_out_idx]) for i in
                              range(monthly_output_written.shape[0])]
        logger_hvac.info(">> SMB : The monthly always on consumption (in Wh) is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [((int(monthly_output_written[i, 0])),
                               monthly_output_written[i, x_ao_out_idx]) for i in
                              range(monthly_output_written.shape[0])]
        logger_hvac.info(">> SMB : The monthly extra always on consumption (in Wh) is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [((int(monthly_output_written[i, 0])),
                               monthly_output_written[i, ac_open_out_idx]) for i in
                              range(monthly_output_written.shape[0])]
        logger_hvac.info(">> SMB : The monthly cooling consumption (in Wh) in open hour is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [((int(monthly_output_written[i, 0])),
                               monthly_output_written[i, ac_close_out_idx]) for i in
                              range(monthly_output_written.shape[0])]
        logger_hvac.info(">> SMB : The monthly cooling consumption (in Wh) in close hour is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [((int(monthly_output_written[i, 0])),
                               monthly_output_written[i, sh_open_out_idx]) for i in
                              range(monthly_output_written.shape[0])]
        logger_hvac.info(">> SMB : The monthly heating consumption (in Wh) in open hour is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [((int(monthly_output_written[i, 0])),
                               monthly_output_written[i, sh_close_out_idx]) for i in
                              range(monthly_output_written.shape[0])]
        logger_hvac.info(">> SMB : The monthly heating consumption (in Wh) in close hour is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [((int(monthly_output_written[i, 0])),
                               monthly_output_written[i, op_out_idx]) for i in
                              range(monthly_output_written.shape[0])]
        logger_hvac.info(">> SMB : The monthly operational consumption (in Wh) is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        if not (disagg_mode == 'mtd'):
            disagg_output_object = fill_user_profile_smb(disagg_input_object, disagg_output_object, logger_pass)
        # =========================================================================================================== #

    else:
        month_ao_hvac_res_net = np.array([])
        epoch_ao_hvac_true_backup = []
        month_ao_hvac_res_net_backup = []
        logger_hvac.warning("HVAC did not run because %s mode needs HSM and it is missing |",
                            global_config.get("disagg_mode"))

    epoch_ao_hvac_true_backup = []

    t_hvac_end = datetime.now()
    logger_hvac.info('HVAC Estimation took | %.3f s ', get_time_diff(t_hvac_start, t_hvac_end))

    # ======================================== Dump AO-OD HVAC Results Separately =================================== #
    write_ao_od_hvac_at_epoch(disagg_input_object, disagg_output_object, epoch_ao_hvac_true_backup)
    # ================================================ Dump HVAC Results ============================================ #
    month_ao_hvac_res_net = month_ao_hvac_res_net_backup
    write_analytics_month_and_epoch_results(disagg_input_object, disagg_output_object, month_ao_hvac_res_net)
    # =============================================================================================================== #

    hvac_exit_status['exit_code'] = int(not bool(hvac_exit_status['error_list']))

    # Write exit status time taken etc
    hvac_metrics = {
        'time': get_time_diff(t_hvac_start, t_hvac_end),
        'confidence': 1.0,
        'exit_status': hvac_exit_status,
    }

    disagg_output_object['disagg_metrics']['hvac'] = hvac_metrics

    return disagg_input_object, disagg_output_object
