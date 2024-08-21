"""
Author - Abhinav Srivastava
Date - 22nd Oct 2018
Call the hvac disaggregation module and get results
"""

# Import python packages

import copy
import scipy
import logging
import numpy as np
import os
import pickle
import pandas as pd
from datetime import datetime

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.write_estimate import write_estimate
from python3.utils.time.get_time_diff import get_time_diff

from python3.disaggregation.aer.hvac.plot_monthly_bar import plot_monthly_bar
from python3.disaggregation.aer.hvac.plot_appmap import generate_appliance_heatmap_new

from python3.disaggregation.aer.hvac.hvac_utility import read_hsm_in
from python3.disaggregation.aer.hvac.hvac_utility import write_ao_od_hvac_at_epoch
from python3.disaggregation.aer.hvac.hvac_utility import write_analytics_month_and_epoch_results

from python3.disaggregation.aer.hvac.hourly_hvac import hvac_module
from python3.disaggregation.aer.hvac.hvac_residue import get_residues
from python3.disaggregation.aer.hvac.postprocess_hvac import postprocess_hvac
from python3.disaggregation.aer.hvac.postprocess_hvac_mtd import postprocess_hvac_mtd
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.disaggregation.aer.hvac.hvac_user_profile import fill_and_validate_user_profile


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

    # plot_level = 1 -> only data files
    # plot_level = 3 -> data files + 4 plots (regression chart, detection histogram, heatmap and monthly bar plot)
    disagg_input_object['switch'] = {}
    disagg_input_object['switch']['plot_level'] = 0

    if disagg_input_object['switch']['plot_level'] > 0:
        plot_list = disagg_input_object['config']['generate_plots']
        plot_list.append('hvac')
        disagg_input_object['config']['generate_plots'] = plot_list

    hour_aggregate_level_ac = static_params['hour_aggregate_level_ac']
    hour_aggregate_level_sh = static_params['hour_aggregate_level_sh']

    if disagg_input_object['config']['pilot_id'] == static_params['hour_aggregate_level_sh']:
        hour_aggregate_level_sh = static_params['fpl_id']
        hour_aggregate_level_ac = static_params['fpl_id']

    disagg_input_object['switch']['hvac'] = {'metrics': True,
                                             'dump_metrics': False,

                                             'epoch_estimate': False,
                                             'month_estimate': False,

                                             'epoch_od_ao_hvac_dump': False,
                                             'hour_aggregate_level_ac': hour_aggregate_level_ac,
                                             'hour_aggregate_level_sh': hour_aggregate_level_sh
                                             }

    disagg_input_object['switch']['residue'] = False


def postprocess_results(global_config, month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_input_object,
                        disagg_output_object, column_index, hvac_tn, logger_hvac):
    """
    Function to postprocess hvac results in case of over/under estimation for Historical and MTD Mode separately

    Parameters:
        global_config               (dict)          : Dictionary containing user level global config parameters
        month_ao_hvac_res_net       (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        epoch_ao_hvac_true          (np.ndarray)    : Array containing | epoch-ao-ac-sh energies
        disagg_input_object         (dict)          : Dictionary containing all input attributes
        disagg_output_object        (dict)          : Dictionary containing all output attributes
        column_index                (dict)          : Dictionary containing column identifier indices of ao-ac-sh
        hvac_tn                     (dict)          : Dictioonary representing boolean to flag non-HVAC detected user
        logger_hvac                 (logger)        : Logger to log stuff in code

    Returns:

        month_ao_hvac_res_net       (np.ndarray)     : Array containing | month-ao-ac-sh-residue-net energies (Processed)
        epoch_ao_hvac_true          (np.ndarray)     : Array containing | epoch-ao-ac-sh energies (Processed)
        disagg_output_object        (dict)           : Dictionary containing all output attributes
    """

    if global_config.get('disagg_mode') != 'mtd':
        month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_output_object = postprocess_hvac(month_ao_hvac_res_net,
                                                                                           epoch_ao_hvac_true,
                                                                                           disagg_input_object,
                                                                                           disagg_output_object,
                                                                                           column_index, hvac_tn,
                                                                                           logger_hvac)
    else:
        month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_output_object = postprocess_hvac_mtd(month_ao_hvac_res_net,
                                                                                               epoch_ao_hvac_true,
                                                                                               disagg_input_object,
                                                                                               disagg_output_object,
                                                                                               column_index)

    return month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_output_object


def run_default_hvac(global_config, disagg_input_object, disagg_output_object, logger_pass, logger_hvac,
                     hvac_exit_status):
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
            hvac_module(disagg_input_object, disagg_output_object, logger_pass, hvac_exit_status)
        disagg_output_object['created_hsm']['hvac'] = hsm_update
        disagg_output_object['hvac_debug'] = debug

    elif global_config.get('disagg_mode') == 'incremental':
        logger_hvac.info("Running incremental mode : |")

        month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status = \
            hvac_module(disagg_input_object, disagg_output_object, logger_pass, hvac_exit_status)
        disagg_output_object['created_hsm']['hvac'] = hsm_update
        disagg_output_object['hvac_debug'] = debug

    elif global_config.get('disagg_mode') == 'mtd':
        logger_hvac.info("Running mtd mode : |")

        month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status = \
            hvac_module(disagg_input_object, disagg_output_object, logger_pass, hvac_exit_status)
        hsm_update = []

    else:
        logger_hvac.error('unrecognized disagg mode %s |', global_config.get('disagg_mode'))

    return month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status


def save_results(disagg_input_object, disagg_output_object, epoch_ao_hvac_true_backup, stage):
    """
    Utility function to save epoch output and debug dictionary objects

    Args:
         disagg_input_object        (dict)          : Dictionary containing all inputs
         disagg_output_object       (dict)          : Dictionary containing all outputs
         epoch_ao_hvac_true_backup  (np.ndarray)    :
         stage                      (str)           :

    Returns:
        None

    """
    static_params = hvac_static_params()

    config = disagg_input_object.get('config', {})
    timestamps = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_EPOCH_IDX]
    tend = int(np.nanmax(timestamps[timestamps > 0]))
    tstart = int(np.nanmin(timestamps[timestamps > 0]))
    disagg_mode = config.get("disagg_mode", "").lower()
    uuid = config.get("uuid", "")

    # dump consumption information if enabled
    if disagg_input_object['config']['disagg_mode'].lower() not in ['historical', 'incremental', 'mtd']:
        return None

    # dump files only if plot level > 1
    if disagg_input_object['switch']['plot_level'] < 1:
        return None

    if stage == 'post_processing':
        # initializing consumption frame
        consumption_frame = pd.DataFrame(epoch_ao_hvac_true_backup,
                                         columns=['epoch', 'baseload', 'ac_ao', 'ac_demand', 'sh_ao',
                                                  'sh_demand', 'net_consumption'])
        # adding ao hvac components
        consumption_frame['ac_ao'] = disagg_output_object['ao_seasonality']['epoch_cooling']
        consumption_frame['sh_ao'] = disagg_output_object['ao_seasonality']['epoch_heating']

        # adding raw consumption and temperature
        consumption_frame['temperature'] = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_TEMPERATURE_IDX]
        consumption_frame['net_energy'] = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # dump folder
        user_epoch_hvac_folder = static_params.get('path', {}).get('epoch_hvac_dir_postprocess')

        if not os.path.exists(user_epoch_hvac_folder):
            os.makedirs(user_epoch_hvac_folder)

        filepath = user_epoch_hvac_folder + '/' + "{disagg_mode}_{uuid}_{tstart}_{tend}.csv".format(disagg_mode=disagg_mode,
                                                                                                    tstart=tstart, tend=tend, uuid=uuid)

        # dump results
        consumption_frame.to_csv(filepath, index=False, header=True)

        print("Files Saved for HVAC - {}".format(stage))


def adjust_hsm_post_process(disagg_output_object, epoch_ao_hvac_true, hsm, disagg_mode, column_idx):
    """
    Function to set detection attributes in hsm to zero in case of suppression
    Args:
        disagg_output_object    (dict)       : Dictionary containing all outputs
        epoch_ao_hvac_true      (np.ndarray) : Array containing epoch level ao and hvac consumption
        hsm                     (dict)       : Update HSM dictionary
        disagg_mode             (str)        : Current disaggregation mode
        column_idx              (dict)       : Column identifier for ac
     """
    total_ac = np.sum(epoch_ao_hvac_true[:, column_idx.get('ac')])
    hvac_debug = disagg_output_object.get('hvac_debug', {})

    # Adjust HSM for MTD run if total estimation of hvac is zero
    if (disagg_mode != 'mtd') and (len(hsm) > 0) and (total_ac == 0):
        hsm['attributes']['no_ac_user'] = [1]
        hsm['attributes']['ac_found'] = [0]
        hsm['attributes']['ac_setpoint'] = [0]
        hsm['attributes']['ac_means'] = [123456789, 123456789]
        hsm['attributes']['ac_std'] = [0, 0]
        hsm['attributes']['ac_mu'] = [123456789]
        hsm['attributes']['ac_mode_limits_limits'] = [123456789.0, 123456789.0, 123456789.0, 123456789.0]

        hvac_debug.get('detection', {}).get('cdd', {}).update({'found': 0})
        hvac_debug.get('estimation', {}).get('cdd', {}).update({'exist': False})

    disagg_output_object['created_hsm']['hvac'] = hsm
    disagg_output_object['hvac_debug'] = hvac_debug

    return disagg_output_object


def generate_month_and_tou_plot(disagg_input_object, disagg_output_object, month_ao_hvac_res_net, epoch_ao_hvac_true,
                                column_index_month, column_index_epoch, stage):
    """
    Function to generate month and tou disagg plots
    Parameters:
        disagg_input_object     (dict)          : Dictionary containing user related all input infotmation
        disagg_output_object    (dict)          : Dictionary containing user related all output infotmation
        month_ao_hvac_res_net   (np.ndarray)    : Array containing monthly ao and hvac consumption
        epoch_ao_hvac_true      (np.ndarray)    : Array containing epoch level ao and hvac consumption
        column_index_month      (dict)          : Dictionary containing column identifiers of ao ac sh
        column_index_epoch      (dict)          : Dictionary containing column identifiers of ao ac sh and residue
    Returns:
        None
    """
    # Input should enable generate plot for hvac
    generate_plot = ('hvac' in disagg_input_object['config']['generate_plots']) or \
                    ('all' in disagg_input_object['config']['generate_plots'])

    if generate_plot and (disagg_input_object['switch']['plot_level'] >= 3) and \
            (disagg_input_object['config']['disagg_mode'] != 'mtd') and stage == 'processed':
        plot_monthly_bar(disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index_month,
                         'processed')

    if generate_plot and (disagg_input_object['switch']['plot_level'] >= 3) and disagg_input_object['config']['disagg_mode'] != 'mtd':
        generate_appliance_heatmap_new(disagg_input_object, disagg_output_object, epoch_ao_hvac_true, column_index_epoch, stage)


def hvac_disagg_wrapper(disagg_input_object, disagg_output_object):
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

    # Initialise disagg input object parameters and check for plotting flags
    hvac_control_centre(disagg_input_object)

    # Start of HVAC module
    t_hvac_start = datetime.now()
    error_list = []

    # Initialise global_config and hvac_exit_status
    global_config = disagg_input_object.get('config')

    if global_config is None:
        error_list.append('Key Error: config does not exist')

    hvac_exit_status = {
        'exit_code': -1,
        'error_list': [],
    }

    # Read latest HSM and  check if attributes are present for historical / incremental mode
    hsm_in = read_hsm_in(disagg_input_object, logger_hvac)
    hsm_update = []
    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (global_config.get("disagg_mode") == "mtd")

    # Initializing monthly and epoch level hvac consumption arrays as failsafe
    epoch_ao_hvac_true = np.array([])
    month_ao_hvac_true = np.array([])

    # hvac_true_neg flags if the user is not a hvac user to disable post-processing later
    # By default, kept at False i.e. could be a hvac user
    hvac_true_neg = {'AC': False}

    # If hsm read was successful
    if (global_config.get('run_mode') == 'prod' or global_config.get('run_mode') == 'custom') and (not hsm_fail):
        logger_hvac.info('user is residential. hvac run mode is | %s', global_config.get('run_mode'))
        # Main function wrapper for calculating epoch level hvac consumption and updated hsm attributes
        month_ao_hvac_true, epoch_ao_hvac_true, debug, hsm_update, hvac_exit_status = \
            run_default_hvac(global_config, disagg_input_object, disagg_output_object, logger_pass, logger_hvac,
                             hvac_exit_status)

        # If no-hvac is detected, flag in hvac_true_neg
        hvac_true_neg = {'AC': bool(debug.get('pre_pipeline', {}).get('all_flags', {}).get('isNotAC', False))}

    # General column index for epoch and month level arrays defined below
    column_index = {'ao': 1, 'ac': 2, 'sh': 3, 'residue': 4, 'net': 5, 'ao_ac': 6, 'od_ac': 7, 'ao_sh': 8, 'od_sh': 9}

    # Update hsm in disagg output object
    if len(hsm_update) > 0:
        disagg_output_object['created_hsm']['hvac'] = hsm_update
    disagg_mode = global_config.get('disagg_mode')

    # If hvac pipeline ran successfully
    if not hsm_fail:

        # Index to write results to disagg output object
        ao_out_idx = disagg_output_object.get('output_write_idx_map').get('ao')
        hvac_out_idx = disagg_output_object.get('output_write_idx_map').get('hvac')
        ac_out_idx = hvac_out_idx[0]
        sh_out_idx = hvac_out_idx[1]

        # Get monthly and epoch arrays with epoch, baseload, ac, sh, residue and net consumption values
        month_ao_hvac_res_net, epoch_ao_hvac_true = get_residues(disagg_input_object, global_config,
                                                                 disagg_output_object,
                                                                 epoch_ao_hvac_true, month_ao_hvac_true, column_index,
                                                                 logger_pass)

        # Post processing HVAC
        # to add / subtract hvac from residue based on month-to-month stability of residue if hvac is detected
        # to remove weekly clashes between ac and sh
        # check no epoch level overshoot from total consumption happens in  ac / sh estimates
        # Separately defined for non mtd and mtd mode
        logger_hvac.info('User is residential. Postprocessing results | ')

        # Output Monthly AO estimates before they get updated in post-processing in HVAC module
        time_format = '%b-%Y'
        monthly_output_log = [(datetime.utcfromtimestamp(month_ao_hvac_res_net[i, 0]).strftime(time_format),
                               month_ao_hvac_true[i, column_index['ao']]) for i in
                              range(month_ao_hvac_res_net.shape[0])]
        logger_hvac.info("The monthly always on consumption (in Wh) before overshoot adjustment is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_output_object = postprocess_results(global_config,
                                                                                              month_ao_hvac_res_net,
                                                                                              epoch_ao_hvac_true,
                                                                                              disagg_input_object,
                                                                                              disagg_output_object,
                                                                                              column_index,
                                                                                              hvac_true_neg,
                                                                                              logger_hvac)
        epoch_ao_hvac_true_od = copy.deepcopy(epoch_ao_hvac_true[:, [0, 1, 7, 9]])
        month_ao_hvac_res_net_od = copy.deepcopy(month_ao_hvac_res_net[:, [0, 1, 7, 9]])

        disagg_output_object = adjust_hsm_post_process(disagg_output_object, epoch_ao_hvac_true, hsm_update,
                                                       disagg_mode, column_index)

        # Create final hvac_output object
        dump_tou_flag = 'tou_all' in disagg_input_object.get('config').get('dump_csv')
        if dump_tou_flag:
            hvac_output = np.c_[disagg_input_object['input_data'][:, Cgbdisagg.INPUT_TEMPERATURE_IDX],
                                disagg_input_object['input_data'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                                copy.deepcopy(disagg_output_object['ao_seasonality']['epoch_cooling']),
                                copy.deepcopy(epoch_ao_hvac_true[:, column_index['ac']]),
                                copy.deepcopy(disagg_output_object['ao_seasonality']['epoch_heating']),
                                copy.deepcopy(epoch_ao_hvac_true[:, column_index['sh']])]

            disagg_output_object['hvac_output'] = hvac_output

        if disagg_mode != 'mtd':
            # hvac debug is not available in mtd
            disagg_output_object['hvac_debug']['write'] = {}
            disagg_output_object['hvac_debug']['write']['month_idx_dentify'] = {'ac_ao': 6, 'ac_od': 7, 'sh_ao': 8,
                                                                                'sh_od': 9}
            disagg_output_object['hvac_debug']['write']['month_ao_hvac_res_net'] = month_ao_hvac_res_net
            disagg_output_object['hvac_debug']['write']['epoch_idx_dentify'] = {'ac_ao': 6, 'ac_od': 7, 'sh_ao': 8,
                                                                                'sh_od': 9}
            disagg_output_object['hvac_debug']['write']['epoch_ao_hvac_true'] = epoch_ao_hvac_true

        month_ao_hvac_true = month_ao_hvac_res_net[:, :column_index['sh'] + 1]

        # Save the files and plots after post processing -> if plot level > 1 in hvac_control_center

        epoch_ao_hvac_true_subset = copy.deepcopy(epoch_ao_hvac_true[:, [0, 1, 6, 7, 8, 9, 5]])
        column_index_epoch = {'ao': 1, 'ac_ao': 2, 'sh_ao': 4, 'ac_od': 3, 'sh_od': 5, 'ac': 3, 'sh': 5, 'net': 6}
        column_idx_month = {'ao': 1, 'ac': 2, 'sh': 3, 'residual': 4, 'ac_od': 7, 'sh_od': 8}
        save_results(disagg_input_object, disagg_output_object, epoch_ao_hvac_true_subset, 'post_processing')

        generate_month_and_tou_plot(disagg_input_object, disagg_output_object, month_ao_hvac_res_net,
                                    epoch_ao_hvac_true_subset,
                                    column_idx_month, column_index_epoch, 'processed')
        # =========================================================================================================== #

        # Ensuring that updated AO values from hvac post processing are in WH
        month_ao_hvac_res_net[:, column_index['ao']] = month_ao_hvac_res_net[:, column_index['ao']] * Cgbdisagg.WH_IN_1_KWH

        # Write estimates for posting
        disagg_output_object = \
            write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['ao'], ao_out_idx, 'epoch')
        disagg_output_object = \
            write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['ac'], ac_out_idx, 'epoch')
        disagg_output_object = \
            write_estimate(disagg_output_object, epoch_ao_hvac_true, column_index['sh'], sh_out_idx, 'epoch')
        disagg_output_object = \
            write_estimate(disagg_output_object, month_ao_hvac_true, column_index['ao'], ao_out_idx, 'bill_cycle')
        disagg_output_object = \
            write_estimate(disagg_output_object, month_ao_hvac_true, column_index['ac'], ac_out_idx, 'bill_cycle')
        disagg_output_object = \
            write_estimate(disagg_output_object, month_ao_hvac_true, column_index['sh'], sh_out_idx, 'bill_cycle')

        # Writing the monthly output to log
        time_format = '%b-%Y'
        monthly_output_log = [(datetime.utcfromtimestamp(month_ao_hvac_true[i, 0]).strftime(time_format),
                               month_ao_hvac_true[i, column_index['ao']]) for i in
                              range(month_ao_hvac_true.shape[0])]
        logger_hvac.info("The monthly always on consumption (in Wh) is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [(datetime.utcfromtimestamp(month_ao_hvac_true[i, 0]).strftime(time_format),
                               month_ao_hvac_true[i, column_index['ac']]) for i in
                              range(month_ao_hvac_true.shape[0])]
        logger_hvac.info("The monthly cooling consumption (in Wh) is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [(datetime.utcfromtimestamp(month_ao_hvac_true[i, 0]).strftime(time_format),
                               month_ao_hvac_true[i, column_index['sh']]) for i in
                              range(month_ao_hvac_true.shape[0])]
        logger_hvac.info("The monthly heating consumption (in Wh) is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

        monthly_output_log = [(datetime.utcfromtimestamp(month_ao_hvac_res_net[i, 0]).strftime(time_format),
                               month_ao_hvac_res_net[i, column_idx_month['residual']]) for i in range(month_ao_hvac_res_net.shape[0])]
        logger_hvac.info("The monthly residue consumption (in Wh) is : | %s",
                         str(monthly_output_log).replace('\n', ' '))

    else:
        epoch_ao_hvac_true_od = []
        month_ao_hvac_res_net_od = []
        logger_hvac.warning("HVAC did not run because %s mode needs HSM and it is missing |",
                            global_config.get("disagg_mode"))

    t_hvac_end = datetime.now()
    logger_hvac.info('HVAC Estimation took | %.3f s ', get_time_diff(t_hvac_start, t_hvac_end))

    # ======================================== Dump AO-OD HVAC Results Separately =================================== #
    write_ao_od_hvac_at_epoch(disagg_input_object, disagg_output_object, epoch_ao_hvac_true_od)
    # ================================================ Dump HVAC Results ============================================ #
    write_analytics_month_and_epoch_results(disagg_input_object, disagg_output_object, month_ao_hvac_res_net_od)
    # =============================================================================================================== #

    hvac_exit_status['exit_code'] = int(not bool(hvac_exit_status['error_list']))

    # Write exit status time taken etc
    hvac_metrics = {
        'time': get_time_diff(t_hvac_start, t_hvac_end),
        'confidence': 1.0,
        'exit_status': hvac_exit_status,
    }

    disagg_output_object['disagg_metrics']['hvac'] = hvac_metrics

    disagg_output_object = \
        fill_and_validate_user_profile(disagg_mode, disagg_input_object, disagg_output_object, logger_pass)

    return disagg_input_object, disagg_output_object
