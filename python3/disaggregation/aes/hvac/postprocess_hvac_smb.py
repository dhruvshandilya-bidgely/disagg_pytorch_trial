"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to post-process disagg results
"""

# Import python packages

import copy
import scipy
import logging
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.get_smb_params import get_smb_params
from python3.disaggregation.aes.hvac.plot_utils_smb_hvac import plot_monthly_bar
from python3.disaggregation.aes.hvac.plot_utils_smb_hvac import generate_appliance_heatmap_new
from python3.disaggregation.aes.hvac.get_hvac_from_residue import get_hvac_from_residue
from python3.disaggregation.aes.hvac.get_residue_from_hvac import get_residue_from_hvac


def remove_fp_hvac(epoch_ao_hvac_true, month_ao_hvac_res_net, alt_dd_months, epoch_input_data, column_index, app,
                   logger_base):
    """
    Function to remove SMB fp hvac

    Parameters:

        epoch_ao_hvac_true      (np.ndarray)      : Array containing epoch level smb HVAC estimates
        month_ao_hvac_res_net   (np.ndarray)      : Array containing month level smb HVAC estimates
        alt_dd_months           (np.ndarray)      : Array containing month identifier
        epoch_input_data        (np.ndarray)      : Array containing epoch level input data
        column_index            (dict)            : Column identifier in input data
        app                     (str)             : HVAC appliance identifier
        logger_base             (logging object)  : Logger to record the code flow

    Returns:

        residue_from_hvac_fp    (np.ndarray)      : Array containing extracted residue from HVAC estimates

    """

    # initializing logger

    logger_local = logger_base.get("logger").getChild("remove_fp_hvac")
    logger_hvac_from_res = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    smb_params = get_smb_params()

    # getting bare low limit of ac and sh monthly consumptions

    if app == 'ac':
        month_low_limit = smb_params.get('postprocess').get('ac_fp_low_limit')
    else:
        month_low_limit = smb_params.get('postprocess').get('sh_fp_low_limit')

    logger_hvac_from_res.info("month low limit for {} is {} |".format(app, month_low_limit))

    # identifying false positive months

    fp_months = month_ao_hvac_res_net[:, 0][(alt_dd_months == 1) &
                                            (month_ao_hvac_res_net[:, column_index[app]] > 0) &
                                            (month_ao_hvac_res_net[:, column_index[app]] <= month_low_limit)]

    residue_from_hvac_fp = np.zeros((epoch_ao_hvac_true.shape[0], 1))

    # getting rid of false positive consumptions

    if np.any(fp_months):
        fp_indexes = np.isin(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], fp_months)

        epoch_ao_hvac_true_backup = copy.deepcopy(epoch_ao_hvac_true)

        residue_from_hvac_fp[fp_indexes, 0] = epoch_ao_hvac_true_backup[fp_indexes, column_index[app]]

        logger_hvac_from_res.info("fp {} removed |".format(app))

    return residue_from_hvac_fp


def remove_week_clashes(epoch_ao_hvac_true, month_ao_hvac_res_net, epoch_input_data, disagg_output_object, column_index,
                        month_idx, logger_base):
    """
    Function to handle cooling-heating clashes in a week

    Parameters:

        epoch_ao_hvac_true      (np.ndarray)    : Contains epoch level hvac consumption
        month_ao_hvac_res_net   (np.ndarray)    : Contains month level hvac consumption
        epoch_input_data        (np.ndarray)    : Contains input data of user
        disagg_output_object    (dict)          : Contains user specific key input payload
        column_index            (dict)          : Contains column identifier for different appliances
        month_idx               (np.ndarray)    : Contains current month identifier
        logger_base             (logging object): Logs the code flow

    Returns:

        epoch_ao_hvac_true      (np.ndarray)    : Contains epoch level hvac consumption
        month_ao_hvac_res_net   (np.ndarray)    : Contains month level hvac consumption
        disagg_output_object    (dict)          : Contains user specific key input payload
    """

    # initializing logger
    logger_local = logger_base.get("logger").getChild("remove_week_clashes")
    logger_week_clash = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    smb_params = get_smb_params()

    # reading raw data
    input_data_copy = copy.deepcopy(epoch_input_data)

    unique_weeks = np.unique(input_data_copy[:, Cgbdisagg.INPUT_WEEK_IDX])

    # getting net on demand and ao hvac copy
    ac_od_net = copy.deepcopy(epoch_ao_hvac_true[:, column_index.get('ac')])
    sh_od_net = copy.deepcopy(epoch_ao_hvac_true[:, column_index.get('sh')])
    ac_ao_net = copy.deepcopy(disagg_output_object.get('ao_seasonality').get('epoch_cooling'))
    sh_ao_net = copy.deepcopy(disagg_output_object.get('ao_seasonality').get('epoch_heating'))

    # initializing hvac to be removed
    ac_od_removed = np.zeros(ac_od_net.shape)
    sh_od_removed = np.zeros(sh_od_net.shape)
    ac_ao_removed = np.zeros(ac_ao_net.shape)
    sh_ao_removed = np.zeros(sh_ao_net.shape)

    logger_week_clash.info("Resolving week by week clashes between ac and sh |")

    for week_idx in range(len(unique_weeks)):

        # current week identifier
        current_week = unique_weeks[week_idx]
        current_week_arr = (input_data_copy[:, Cgbdisagg.INPUT_WEEK_IDX] == current_week).astype(int)

        # getting current week on demand and ao havc values
        current_week_ac_od = current_week_arr * ac_od_net
        current_week_sh_od = current_week_arr * sh_od_net
        current_week_ac_ao = current_week_arr * ac_ao_net
        current_week_sh_ao = current_week_arr * sh_ao_net

        # getting net hvac for current week at epoch level
        current_week_ac = current_week_ac_od + current_week_ac_ao
        current_week_sh = current_week_sh_od + current_week_sh_ao

        # getting total ac and sh consumption in current week
        current_week_ac_total = np.nansum(current_week_ac)
        current_week_sh_total = np.nansum(current_week_sh)

        # identifying fp hvac and modifying values at epoch level
        if current_week_ac_total >= current_week_sh_total:
            sh_od_net = sh_od_net - current_week_sh_od
            sh_ao_net = sh_ao_net - current_week_sh_ao
            sh_od_removed = sh_od_removed + current_week_sh_od
            sh_ao_removed = sh_ao_removed + current_week_sh_ao

        elif current_week_ac_total < current_week_sh_total:
            ac_od_net = ac_od_net - current_week_ac_od
            ac_ao_net = ac_ao_net - current_week_ac_ao
            ac_od_removed = ac_od_removed + current_week_ac_od
            ac_ao_removed = ac_ao_removed + current_week_ac_ao

        else:
            continue

    logger_week_clash.info("Updating ac and sh values at epoch level |")

    # updating new on demand hvac values in epoch level carrier
    epoch_ao_hvac_true[:, column_index['ac']] = ac_od_net
    epoch_ao_hvac_true[:, column_index['sh']] = sh_od_net

    # updating new always on hvac values in epoch level carrier
    disagg_output_object['ao_seasonality']['epoch_cooling'] = ac_ao_net
    disagg_output_object['ao_seasonality']['epoch_heating'] = sh_ao_net

    # getting month level aggregates for new hvac
    month_ao_cool = np.bincount(month_idx, ac_ao_net)
    month_ao_heat = np.bincount(month_idx, sh_ao_net)
    month_od_cool = np.bincount(month_idx, ac_od_net)
    month_od_cool = month_od_cool / Cgbdisagg.WH_IN_1_KWH
    month_od_heat = np.bincount(month_idx, sh_od_net)
    month_od_heat = month_od_heat / Cgbdisagg.WH_IN_1_KWH

    month_ac_od_removed = np.bincount(month_idx, ac_od_removed)
    month_ac_od_removed = month_ac_od_removed / Cgbdisagg.WH_IN_1_KWH
    month_sh_od_removed = np.bincount(month_idx, sh_od_removed)
    month_sh_od_removed = month_sh_od_removed / Cgbdisagg.WH_IN_1_KWH

    month_sh_ao_removed = np.bincount(month_idx, sh_ao_removed)
    month_sh_ao_removed = month_sh_ao_removed / Cgbdisagg.WH_IN_1_KWH
    month_ac_ao_removed = np.bincount(month_idx, ac_ao_removed)
    month_ac_ao_removed = month_ac_ao_removed / Cgbdisagg.WH_IN_1_KWH

    # updating month level always on hvac in ao seasonality carrier
    disagg_output_object['ao_seasonality']['cooling'] = month_ao_cool
    disagg_output_object['ao_seasonality']['heating'] = month_ao_heat

    month_ao_hvac_res_net[:, column_index.get('ac')] = month_od_cool
    month_ao_hvac_res_net[:, column_index.get('sh')] = month_od_heat

    res_col = smb_params.get('utility').get('residue_col')

    logger_week_clash.info("Updating ac and sh values at month level |")

    # updating month level new residues
    month_ao_hvac_res_net[:, res_col] = month_ao_hvac_res_net[:, res_col] + month_ac_od_removed + month_sh_od_removed + \
                                        month_sh_ao_removed + month_ac_ao_removed

    return epoch_ao_hvac_true, month_ao_hvac_res_net, disagg_output_object


def postprocess_hvac_smb(month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_input_object, disagg_output_object,
                         column_index, logger_base):
    """
    Function to postprocess hvac results in case of over/under estimation

    Parameters:

        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies
        disagg_input_object     (dict)          : Dictionary containing all input attributes
        disagg_output_object    (dict)          : Dictionary containing all output attributes
        column_index            (dict)          : Dictionary containing column identifier indices of ao-ac-sh
        logger_base             (logging object): Captures log during code flow

    Returns:

        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies (Processed)
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies (Processed)
    """

    smb_params = get_smb_params()

    # initializing logger
    logger_local = logger_base.get("logger").getChild("process_hvac_smb")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_process_hvac_smb = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting most common base residual
    residual_col = smb_params.get('utility').get('residue_col')
    residual = month_ao_hvac_res_net[:, residual_col]
    residual_to_meet = np.nanmedian(residual)

    logger_process_hvac_smb.info("Residual to meet : {} |".format(residual_to_meet))

    # accessing and making copy of input data
    epoch_input_data = copy.deepcopy(disagg_input_object.get('switch').get('hvac_input_data'))

    # getting month epochs and temperature
    month_epoch, idx_2, month_idx = scipy.unique(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True,
                                                 return_inverse=True)
    temperature = epoch_input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX]

    # identifying most probable underestimated hvac months
    underestimated_boolean = residual > (np.median(residual[1:- 1]) +
                                         smb_params.get('postprocess').get('underestimation_arm') * np.std(residual[1: -1]))

    logger_process_hvac_smb.info("Total underestimation base : {} |".format(np.nansum(underestimated_boolean)))

    # getting overestimated metric
    overestimation_metric = np.nanmedian(residual[1: -1]) - \
                            smb_params.get('postprocess').get('overestimation_arm') * np.std(residual[1: -1])

    # identifying overestimated months
    if overestimation_metric < 0:
        overestimated_boolean = residual < np.nanmedian(residual[1: -1])
    else:
        overestimated_boolean = residual < (np.nanmedian(residual[1: -1]) -
                                            smb_params.get('postprocess').get('overestimation_arm') * np.nanstd(residual[1: -1]))

    # sanitizing overestimated booleans
    overestimated_boolean[0] = False
    overestimated_boolean[-1] = False
    logger_process_hvac_smb.info("Total overestimation base : {} |".format(np.nansum(overestimated_boolean)))

    # getting epoch level base cdd and hdd
    cdd = np.fmax(temperature - smb_params.get('postprocess').get('cdd_reference_temp'), 0)
    hdd = np.fmax(0, smb_params.get('postprocess').get('hdd_reference_temp') - temperature)
    # aggregating to month level
    month_cdd = np.bincount(month_idx, cdd)
    month_hdd = np.bincount(month_idx, hdd)
    # ensuring no overlap of cdd or hdd over months
    month_cdd[month_cdd < month_hdd] = 0
    month_hdd[month_hdd < month_cdd] = 0
    # creating summer or winter identifiers
    cdd_months = (month_cdd > 0).astype(bool)
    hdd_months = (month_hdd > 0).astype(bool)

    # creating common appliance data
    appliance_df = pd.DataFrame()
    appliance_df['ao'] = epoch_ao_hvac_true[:, column_index['ao']]
    appliance_df['ac'] = epoch_ao_hvac_true[:, column_index['ac']]
    appliance_df['sh'] = epoch_ao_hvac_true[:, column_index['sh']]

    # creating input data
    input_df = pd.DataFrame(epoch_input_data)
    columns = Cgbdisagg.INPUT_COLUMN_NAMES

    input_df.columns = columns
    input_data_raw = input_df[['epoch', 'consumption', 'temperature']]

    # getting residue
    appliance_df['residue'] = input_data_raw['consumption'] - (appliance_df['ao'] + appliance_df['ac'] +
                                                               appliance_df['sh'] +
                                                               disagg_output_object['ao_seasonality']['epoch_grey'])
    appliance_df_deepcopy = copy.deepcopy(appliance_df)

    # identifying epoch level hvac contenders
    epoch_cooling_contenders = (cdd > 0).astype(bool)
    epoch_heating_contenders = (hdd > 0).astype(bool)

    # creating info carrier for regulating hvac estimates
    info_carrier = {'appliance_df': appliance_df,
                    'epoch_input_data': epoch_input_data,
                    'residual_to_meet': residual_to_meet,
                    'month_epoch': month_epoch}

    # Handling Obvious Underestimation at month level
    cooling_from_residue = get_hvac_from_residue(underestimated_boolean, epoch_cooling_contenders,
                                                 residual, appliance_df_deepcopy, cdd_months, info_carrier, logger_pass)
    heating_from_residue = get_hvac_from_residue(underestimated_boolean, epoch_heating_contenders,
                                                 residual, appliance_df_deepcopy, hdd_months, info_carrier, logger_pass)

    # Handling Obvious Overestimation at month level
    residue_from_cooling = get_residue_from_hvac(overestimated_boolean, epoch_cooling_contenders, residual,
                                                 cdd_months, 'ac', info_carrier, logger_pass)
    residue_from_heating = get_residue_from_hvac(overestimated_boolean, epoch_heating_contenders, residual,
                                                 hdd_months, 'sh', info_carrier, logger_pass)

    epoch_ao_hvac_true[:, column_index['ac']] = epoch_ao_hvac_true[:, column_index['ac']] + cooling_from_residue[:, 0] - residue_from_cooling[:, 0]

    epoch_ao_hvac_true[:, column_index['sh']] = epoch_ao_hvac_true[:, column_index['sh']] + heating_from_residue[:, 0] - residue_from_heating[:, 0]

    # aggregating ac estimates at month level for monthly estimates
    month_cooling_from_residue = np.bincount(month_idx, cooling_from_residue[:, 0])
    month_cooling_from_residue = month_cooling_from_residue / Cgbdisagg.WH_IN_1_KWH
    month_residue_from_cooling = np.bincount(month_idx, residue_from_cooling[:, 0])
    month_residue_from_cooling = month_residue_from_cooling / Cgbdisagg.WH_IN_1_KWH

    # aggregating sh estimates at month level for monthly estimates
    month_heating_from_residue = np.bincount(month_idx, heating_from_residue[:, 0])
    month_heating_from_residue = month_heating_from_residue / Cgbdisagg.WH_IN_1_KWH
    month_residue_from_heating = np.bincount(month_idx, residue_from_heating[:, 0])
    month_residue_from_heating = month_residue_from_heating / Cgbdisagg.WH_IN_1_KWH

    logger_process_hvac_smb.info("Month residue adjustments done |")

    # updating master array with hvac estimates
    month_ao_hvac_res_net[:, column_index.get('ac')] = (month_ao_hvac_res_net[:, column_index.get('ac')] +
                                                        month_cooling_from_residue) - month_residue_from_cooling
    month_ao_hvac_res_net[:, column_index.get('sh')] = (month_ao_hvac_res_net[:, column_index.get('sh')] +
                                                        month_heating_from_residue) - month_residue_from_heating

    # updating master array with residue estimates
    month_ao_hvac_res_net[:, residual_col] = (month_ao_hvac_res_net[:, residual_col] + month_residue_from_cooling +
                                              month_residue_from_heating) - \
                                             (month_cooling_from_residue + month_heating_from_residue)

    logger_process_hvac_smb.info("Appliance consumption matrix updated at month : {} |".format(np.nansum(overestimated_boolean)))

    # Block Handler 2 ----------------------------------- >

    epoch_ao_hvac_true, month_ao_hvac_res_net, disagg_output_object = remove_week_clashes(epoch_ao_hvac_true,
                                                                                          month_ao_hvac_res_net,
                                                                                          epoch_input_data,
                                                                                          disagg_output_object,
                                                                                          column_index, month_idx,
                                                                                          logger_pass)

    # Block Handler 3 ----------------------------------- >

    residue_from_cooling_fp = remove_fp_hvac(epoch_ao_hvac_true, month_ao_hvac_res_net, hdd_months,
                                             epoch_input_data, column_index, 'ac', logger_pass)
    residue_from_heating_fp = remove_fp_hvac(epoch_ao_hvac_true, month_ao_hvac_res_net, cdd_months,
                                             epoch_input_data, column_index, 'sh', logger_pass)

    epoch_ao_hvac_true[:, column_index.get('ac')] = epoch_ao_hvac_true[:, column_index.get('ac')] - residue_from_cooling_fp[:, 0]
    epoch_ao_hvac_true[:, column_index.get('sh')] = epoch_ao_hvac_true[:, column_index.get('sh')] - residue_from_heating_fp[:, 0]

    # aggregating ac estimates at month level for monthly estimates

    month_residue_from_cooling_fp = np.bincount(month_idx, residue_from_cooling_fp[:, 0])
    month_residue_from_cooling_fp = month_residue_from_cooling_fp / Cgbdisagg.WH_IN_1_KWH

    month_residue_from_heating_fp = np.bincount(month_idx, residue_from_heating_fp[:, 0])
    month_residue_from_heating_fp = month_residue_from_heating_fp / Cgbdisagg.WH_IN_1_KWH

    month_ao_hvac_res_net[:, column_index.get('ac')] = month_ao_hvac_res_net[:, column_index.get('ac')] - month_residue_from_cooling_fp
    month_ao_hvac_res_net[:, column_index.get('sh')] = month_ao_hvac_res_net[:, column_index.get('sh')] - month_residue_from_heating_fp

    month_ao_hvac_res_net[:, residual_col] = month_ao_hvac_res_net[:, residual_col] + month_residue_from_cooling_fp + \
                                             month_residue_from_heating_fp

    # Plotter --------------------------------------------- >

    # generate plots as required
    if (disagg_input_object.get('switch').get('plot_level') >= 1) and not (
            disagg_input_object.get('config').get('disagg_mode') == 'mtd'):
        plot_monthly_bar(disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index, 'processed')

    # generating plots as required
    if (disagg_input_object.get('switch').get('plot_level') >= 3) and not (
            disagg_input_object.get('config').get('disagg_mode') == 'mtd'):
        generate_appliance_heatmap_new(disagg_input_object, disagg_output_object, epoch_ao_hvac_true, 'processed')

    return month_ao_hvac_res_net, epoch_ao_hvac_true


def postprocess_results_smb(global_config, month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_input_object,
                            disagg_output_object, column_index, logger_base):
    """
    Function to postprocess hvac results in case of over/under estimation, except in mtd mode

    Parameters:
        global_config           (dict)          : Dictionary containing user level global config parameters
        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies
        disagg_input_object     (dict)          : Dictionary containing all input attributes
        disagg_output_object    (dict)          : Dictionary containing all output attributes
        column_index            (dict)          : Dictionary containing column identifier indices of ao-ac-sh

    Returns:

        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies (Processed)
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies (Processed)
    """

    # postprocessing hvac smb in non-mtd mode
    if not (global_config.get('disagg_mode') == 'mtd'):
        month_ao_hvac_res_net, epoch_ao_hvac_true = postprocess_hvac_smb(month_ao_hvac_res_net, epoch_ao_hvac_true,
                                                                         disagg_input_object, disagg_output_object,
                                                                         column_index, logger_base)
    return month_ao_hvac_res_net, epoch_ao_hvac_true


def filter_x_hour_consumption(mode_x_hour_output, x_hour_net, hvac_params_post_process, logger_base):
    """
    Function to postprocess day level hvac estimates

    Parameters:
        mode_x_hour_output (np.ndarray)     : 2D array of day level cooling-heating estimates
        x_hour_net (np.ndarray)             : Day level aggregate of energy consumption (flowing through hvac)
        hvac_params_post_process (dict)     : Dictionary containing hvac postprocess related initialized parameters
        logger_base (logging object)        : Writes logs during code flow
    Returns:
        mode_x_hour_output (np.ndarray)     : 2D array of day level cooling-heating estimates based on daily kWh and %
    """

    logger_local = logger_base.get("logger").getChild("filter_daily_consumption")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # validity check on minimum daily kwh for hvac
    idx = np.logical_or(mode_x_hour_output < hvac_params_post_process['MIN_DAILY_KWH'] * 1000,
                        mode_x_hour_output < hvac_params_post_process['MIN_DAILY_PERCENT'] * x_hour_net)
    mode_x_hour_output[idx] = 0

    logger_hvac.info('post processing daily: ensuring consumption doesnt exceed daily max % and min % energy |')

    # validity check on maximum daily percentages for hvac w.r.t. daily consumption
    idx = mode_x_hour_output > hvac_params_post_process['MAX_DAILY_PERCENT'] * x_hour_net
    mode_x_hour_output[idx] = hvac_params_post_process['MAX_DAILY_PERCENT'] * x_hour_net[idx]

    return mode_x_hour_output


def fill_x_hour_consumption(x_hour_output, hvac_input_data, epoch_filtered_data, idx_identifier, logger_base):
    """
    Function to postprocess processed day level hvac estimates to get estimate epoch level estimates

        Parameters:
            x_hour_output (np.ndarray)            : Day level estimate of appliance (AC or SH) energy consumption
            hvac_input_data (np.ndarray)          : 2D Array of epoch level input data frame flowing into hvac module
            epoch_filtered_data (np.ndarray)      : Epoch level energy for AC or SH - After data sanity filtering
            idx_identifier (np.ndarray)           : Array of day identifier indexes (epochs of a day have same index)
            logger_base(logging object)           : Writes logs during code flow
        Returns:
            hourly_output (np.ndarray)            : 2D array of epoch level cooling-heating estimates
    """

    logger_local = logger_base.get("logger").getChild("fill_hourly_consumption")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting daily hour count
    daily_hour_count = np.bincount(idx_identifier, (epoch_filtered_data > 0).astype(int))

    # getting hvac hour count
    hourly_hour_count = daily_hour_count[idx_identifier]
    hourly_hour_count[epoch_filtered_data <= 0] = 0

    epoch_filtered_data[epoch_filtered_data > 0] = hvac_input_data[
        epoch_filtered_data > 0, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # getting daily net ao
    daily_net_bl = np.bincount(idx_identifier, epoch_filtered_data)

    temp_daily_output = copy.deepcopy(x_hour_output)
    temp_daily_net_bl = copy.deepcopy(daily_net_bl)
    temp_daily_net_bl[np.isnan(temp_daily_net_bl)] = 10 ** 10
    logger_hvac.info('post processing for hvac consumption: ensuring daily output doesnt exceed daily net energy |')

    # post processing for hvac consumption: ensuring daily output doesnt exceed daily net energy
    x_hour_output = np.minimum(temp_daily_output, temp_daily_net_bl)
    epoch_hvac_output = x_hour_output[idx_identifier] / hourly_hour_count
    epoch_hvac_output[epoch_filtered_data <= 0] = 0

    return epoch_hvac_output


def get_x_hour_and_epoch_hvac(x_hour_hvac_by_mode, ac_filter_info, sh_filter_info, hvac_input_data,
                              idx_identifier, hvac_params, logger_base):
    """
    Function to postprocess day level hvac estimates to epoch level estimates

        Parameters:
            x_hour_hvac_by_mode (dict)              : Day level estimate of appliance (AC and SH) energy consumption
            ac_filter_info (dict)                   : Epoch level energy for AC - After data sanity filtering
            sh_filter_info (dict)                   : Epoch level energy for SH - After data sanity filtering
            hvac_input_data (np.ndarray)            : 2D Array of epoch level input data frame flowing into hvac module
            idx_identifier (np.ndarray)             : Array of day identifier indexes (epochs of a day have same index)
            hvac_params (dict)                      : Dictionary containing hvac algo related initialized parameters
            logger_base(logging object)             : Writes logs during code flow
        Returns:
            daily_output (np.ndarray)               : 2D array of day level cooling-heating estimates
            hourly_output (np.ndarray)              : 2D array of epoch level cooling-heating estimates
    """

    logger_local = logger_base.get("logger").getChild("post_process_consumption")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}

    net_at_identifier_level = np.bincount(idx_identifier, hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    cooling_energy_modes = x_hour_hvac_by_mode['cooling']
    epoch_cooling = pd.DataFrame()

    for mode in cooling_energy_modes.keys():
        mode_x_hour_output = cooling_energy_modes[mode]['hvac_estimate_day']
        mode_x_hour_output = filter_x_hour_consumption(mode_x_hour_output, net_at_identifier_level,
                                                       hvac_params['postprocess']['AC'], logger_pass)

        mode_epoch_filtered_data = np.array(ac_filter_info['epoch_filtered_data'][mode])
        epoch_cooling[mode] = fill_x_hour_consumption(mode_x_hour_output, hvac_input_data, mode_epoch_filtered_data,
                                                      idx_identifier, logger_pass)
    if len(cooling_energy_modes.keys()) == 0:
        epoch_cooling[0] = np.zeros(np.shape(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])[0])

    heating_energy_modes = x_hour_hvac_by_mode['heating']
    epoch_heating = pd.DataFrame()

    for mode in heating_energy_modes.keys():
        mode_x_hour_output = heating_energy_modes[mode]['hvac_estimate_day']
        mode_x_hour_output = filter_x_hour_consumption(mode_x_hour_output, net_at_identifier_level,
                                                       hvac_params['postprocess']['SH'], logger_pass)

        mode_epoch_filtered_data = np.array(sh_filter_info['epoch_filtered_data'][mode])
        epoch_heating[mode] = fill_x_hour_consumption(mode_x_hour_output, hvac_input_data, mode_epoch_filtered_data,
                                                      idx_identifier, logger_pass)
    if len(heating_energy_modes.keys()) == 0:
        epoch_heating[0] = np.zeros(np.shape(hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])[0])

    epoch_cooling = epoch_cooling.sum(axis=1)
    epoch_heating = epoch_heating.sum(axis=1)
    epoch_hvac = np.c_[epoch_cooling, epoch_heating]

    return x_hour_hvac_by_mode, epoch_hvac
