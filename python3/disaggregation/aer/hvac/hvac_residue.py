"""
Author - Abhinav Srivastava
Date - 22nd Oct 2018
Call to get the reside left after hvac estimate
"""

# Import python packages
import logging
import os
import copy
import scipy
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.hvac_utility import get_residue_stability
from python3.disaggregation.aer.hvac.hvac_utility import quantize_month_degree_day
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def get_quantized_r2(cdd_months, hdd_months, residual, month_cdd_quantized, month_hdd_quantized, qualified_residual,
                     residual_quantized):
    """
    Function to calculate pearson correlation coefficient between quantized residual and cdd/hdd

    Parameters:
        cdd_months              (np.ndarray)  : Array containing information about validity of a month for cooling
        hdd_months              (np.ndarray)  : Array containing information about validity of a month for heating
        residual                (np.ndarray)  : Array containing monthly residue
        month_cdd_quantized     (np.ndarray)  : Array containing quantized cdd at month level
        month_hdd_quantized     (np.ndarray)  : Array containing quantized hdd at month level
        qualified_residual      (np.ndarray)  : Array containing info about valid residuals for getting r2 value
        residual_quantized      (np.ndarray)  : Array containing info about residuals in quantized form

    Returns:
        cool_r_square_quantized (np.ndarray)  : Array containing r square value of cooling
        heat_r_square_quantized (np.ndarray)  : Array containing r square value of heating
    """

    static_params = hvac_static_params()
    valid = 1
    quantized_r2_min_months = static_params['quantized_r2_min_months']
    quantized_low_lim = static_params['quantized_low_lim']
    quantized_high_lim = static_params['quantized_high_lim']

    # Calculate residual seasonality factor (r2) for cdd months
    if (sum(cdd_months) >= quantized_low_lim) and (np.max(residual[cdd_months == 1]) > quantized_high_lim):

        cdd_months_valid = len(month_cdd_quantized[(cdd_months == valid) & qualified_residual])
        residual_months_valid = len(residual_quantized[(cdd_months == valid) & qualified_residual])

        # Default r squared value is zero
        cool_r_square_quantized = 0
        if (cdd_months_valid >= quantized_r2_min_months) and (residual_months_valid >= quantized_r2_min_months):
            cool_r_square_quantized = pearsonr(month_cdd_quantized[(cdd_months == valid) & qualified_residual],
                                               residual_quantized[(cdd_months == valid) & qualified_residual])[0]
    else:

        cool_r_square_quantized = static_params['default_r_square_quantized']

    # Calculate residual seasonality factor (r2) for hdd months
    if (sum(hdd_months) >= quantized_low_lim) and (np.max(residual[hdd_months == 1]) > quantized_high_lim):
        hdd_months_valid = len(month_hdd_quantized[(hdd_months == valid) & qualified_residual])
        residual_months_valid = len(residual_quantized[(hdd_months == valid) & qualified_residual])

        # Default r squared value is zero
        heat_r_square_quantized = 0
        if (hdd_months_valid >= quantized_r2_min_months) and (residual_months_valid >= quantized_r2_min_months):
            heat_r_square_quantized = pearsonr(month_hdd_quantized[(hdd_months == valid) & qualified_residual],
                                               residual_quantized[(hdd_months == valid) & qualified_residual])[0]
    else:

        heat_r_square_quantized = static_params['default_r_square_quantized']

    return cool_r_square_quantized, heat_r_square_quantized


def get_residue_seasonality(residual, epoch_input_data, month_idx, qualified_residual, logger_base):
    """
    Function to get residue seasonality to measure the undere/over estimation candidate months for HVAC

     Parameters:
         residual           (np.ndarray)      : Array containing all the monthly residuals
         epoch_input_data   (np.ndarray)      : Array containing epoch level input temperature and consumption data
         month_idx          (int)             : Month identifier
         qualified_residual (np.ndarray)      : Array containing all the qualified monthly residuals
         logger_base        (logging object)  : Records code flow logs

     Returns:
         stability_metric   (tuple)           : Tuple carrying stability values
    """

    # Initiate logger for the hvac module
    logger_hvac_base = logger_base.get('logger').getChild('get_res_seasonality')
    logger_get_res_seasonality = logging.LoggerAdapter(logger_hvac_base, logger_base.get('logging_dict'))

    static_params = hvac_static_params()

    # getting residue data
    df = pd.DataFrame()
    df['residual'] = np.around(residual)
    residual_quantized = residual // static_params['residual_quantization_val']

    logger_get_res_seasonality.info("Got quantized residuals |")

    # reading temperature data
    temperature = epoch_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # getting cdd and hdd
    cdd = np.fmax(temperature - static_params['pivot_F'], 0)
    hdd = np.fmax(0, static_params['pivot_F'] - temperature)

    # month cdd hdd calculated
    month_cdd = np.bincount(month_idx, cdd)
    month_hdd = np.bincount(month_idx, hdd)

    logger_get_res_seasonality.info("got month cdd and hdd |")

    # degree day sanitized
    month_cdd[month_cdd < month_hdd] = 0
    month_hdd[month_hdd < month_cdd] = 0

    # cdd/hdd degree day month identifier initiated
    cdd_months = (month_cdd > 0).astype(int)
    hdd_months = (month_hdd > 0).astype(int)

    logger_get_res_seasonality.info("quantizing month degree days |")
    # quantized month degree days
    month_cdd_quantized, month_hdd_quantized = quantize_month_degree_day(cdd_months, month_cdd, hdd_months, month_hdd)

    # quantized r square
    cool_r_square_quantized, heat_r_square_quantized = get_quantized_r2(cdd_months, hdd_months, residual,
                                                                        month_cdd_quantized, month_hdd_quantized,
                                                                        qualified_residual, residual_quantized)
    logger_get_res_seasonality.info("quantized r square |")

    cool_r_square_quantized = np.around(cool_r_square_quantized, 2)
    heat_r_square_quantized = np.around(heat_r_square_quantized, 2)

    return cool_r_square_quantized, heat_r_square_quantized


def get_residues(disagg_input_object, global_config, disagg_output_object, epoch_ao_hvac_true,
                 monthly_ao_hvac_true, column_index, logger_base):
    """
    Function to get the month level residues and populate in monthly_ao_hvac_true array

    Parameters:

        disagg_input_object     (dict)            : Dicctionary containg all disagg inputs
        global_config           (dict)            : Dinctionary containing user level important config parameters
        disagg_output_object    (dict)            : Dicctionary containg all disagg related outputs
        epoch_ao_hvac_true      (np.ndarray)      : 2D array containing epoch timestamp, ac, sh and net consumption
        monthly_ao_hvac_true    (np.ndarray)      : Array that keeps a registry of month level AO-HVAC
        column_index            (dict)            : Dictionary containing index info for AO-AC-SH in monthly_ao_hvac_true array
        logger_base             (logging object)  : Records log during code flow

    Returns:

        month_ao_hvac_res_net  (np.ndarray)       : Array that keeps a registry of month level AO-AC-SH-Residue-Total energy
        epoch_ao_hvac_res_net  (np.ndarray)       : Array that keeps a registry of epoch level AO-AC-SH-Residue-Total energy
    """

    # Initiate logger for the hvac module
    logger_hvac_base = logger_base.get('logger').getChild('get_residues')
    logger_pass = {"logger": logger_hvac_base, "logging_dict": logger_base.get("logging_dict")}
    logger_get_residues = logging.LoggerAdapter(logger_hvac_base, logger_base.get('logging_dict'))

    static_params = hvac_static_params()

    ao_consumption = (monthly_ao_hvac_true[:, column_index['ao']] / static_params['kilo'])
    ao_grey = disagg_output_object['ao_seasonality']['grey'] / static_params['kilo']
    ac_consumption = (monthly_ao_hvac_true[:, column_index['ac']] / static_params['kilo'])
    sh_consumption = (monthly_ao_hvac_true[:, column_index['sh']] / static_params['kilo'])

    # getting month epoch identifiers, and epoch level markings of month epoch identifiers

    logger_get_residues.info("getting month epoch identifiers, and epoch level markings of month epoch identifiers |")

    # taking a deep-copy of input data before replacing nan values
    input_data = copy.deepcopy(disagg_input_object['input_data'])
    input_data[np.isnan(input_data)] = 0
    epoch_input_data = copy.deepcopy(input_data)
    month_epoch, _, month_idx = scipy.unique(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                             return_index=True, return_inverse=True)

    # aggregating net consumption at month level
    month_net = np.bincount(month_idx, epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    month_net = np.c_[month_epoch, month_net]
    net_consumption = (month_net[:, 1] / static_params['kilo'])

    # getting residue at month level
    residual = net_consumption - (ao_consumption + ac_consumption + sh_consumption) + ao_grey

    month_ao_hvac_res_net = np.c_[month_epoch, ao_consumption, ac_consumption, sh_consumption,
                                  residual, net_consumption]

    epoch_grey_component = disagg_output_object.get('ao_seasonality', {}).get('epoch_grey', 0)
    epoch_residue = epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - (epoch_ao_hvac_true[:, column_index.get('ao')] +
                                                                            epoch_ao_hvac_true[:, column_index.get('ac')] +
                                                                            epoch_ao_hvac_true[:, column_index.get('sh')]) + epoch_grey_component

    epoch_ao_hvac_res_net = np.c_[epoch_ao_hvac_true, epoch_residue]
    epoch_ao_hvac_res_net = np.c_[epoch_ao_hvac_res_net, epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]]

    logger_get_residues.info("got residuals at month level |")

    # rounding residuals
    residual_round = np.around(residual, -1)

    # getting residual stability
    if len(residual_round) > static_params['len_residual_round']:

        logger_get_residues.info("residual valid for seasonality |")

        residual_cut_off = np.sort(residual_round)[1]
        qualified_residual = residual_round > residual_cut_off

        residual_rsquare = get_residue_seasonality(residual, epoch_input_data, month_idx, qualified_residual,
                                                   logger_pass)
        residual_stability = get_residue_stability(residual, qualified_residual, logger_pass)
    else:
        logger_get_residues.info("using default residual r square and stability |")
        residual_rsquare = (static_params['default_residual_rsquare'], static_params['default_residual_rsquare'])
        residual_stability = (static_params['default_residual_stability'], static_params['default_residual_stability'])

    # deciding whether to dump residue or not
    if disagg_input_object['switch']['residue']:

        uuid = global_config['uuid']

        # making residue folder
        residue_dir = os.path.join('../', "residue")
        if not os.path.exists(residue_dir):
            os.makedirs(residue_dir)
        df_residual = pd.DataFrame()

        # preparing residue values for writing
        df_residual['residual'] = np.around(residual)
        pd.DataFrame.to_csv(df_residual, residue_dir + '/' + uuid + '.csv', header=False, index=False)

        # getting residue metrics
        residue_metric_dir = os.path.join('../', "residue_metrics")
        if not os.path.exists(residue_metric_dir):
            os.makedirs(residue_metric_dir)
        df_residual_metric = pd.DataFrame()

        # rounding values
        rounding_limit = 3

        # preparing residual metric frame
        df_residual_metric['res_summer_r2'] = [np.around(residual_rsquare[0], rounding_limit)]
        df_residual_metric['res_winter_r2'] = [np.around(residual_rsquare[1], rounding_limit)]
        df_residual_metric['stability_mean'] = [np.around(residual_stability[0], rounding_limit)]
        df_residual_metric['stability_std'] = [np.around(residual_stability[1], rounding_limit)]

        # dumping values
        pd.DataFrame.to_csv(df_residual_metric, residue_metric_dir + '/' + uuid + '.csv', header=False, index=False)

    logger_get_residues.info("assigning residual stability and r square to analytics |")

    # noting key metrics for analytics
    disagg_output_object['analytics']['values']['residual_stability'] = residual_stability
    disagg_output_object['analytics']['values']['residual_rsquare'] = residual_rsquare

    return month_ao_hvac_res_net, epoch_ao_hvac_res_net
