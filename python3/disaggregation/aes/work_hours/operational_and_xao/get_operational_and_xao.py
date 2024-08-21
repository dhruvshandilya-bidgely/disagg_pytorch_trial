"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to get smb specific consumptions
"""

# Import python packages
import copy
import scipy
import logging
import matplotlib
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.get_smb_params import get_smb_params
from python3.disaggregation.aes.hvac.homogenize_hvac import homogenize_hvac_maps

from python3.disaggregation.aes.plotting_utility.plot_smb_bar import plot_monthly_bar_smb
from python3.disaggregation.aes.plotting_utility.plot_smb_heatmap import generate_app_heatmap_smb

from python3.disaggregation.aes.work_hours.operational_and_xao.prepare_appliance_data import prepare_appliance_df
from python3.disaggregation.aes.work_hours.operational_and_xao.hour_wise_operational import get_hour_wise_operational
from python3.disaggregation.aes.work_hours.operational_and_xao.get_month_level_operational import \
    get_month_level_operational_load

from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import get_base_maps
from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import get_hvac_maps
from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import prepare_input_df
from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import get_sanitized_grey_data
from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import getting_others_sanity_by_op
from python3.disaggregation.aes.work_hours.operational_and_xao.smb_utility import getting_others_sanity_by_hvac

matplotlib.use('Agg')


def get_operational_and_xao(month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_input_object, disagg_output_object,
                            column_index, logger_base):
    """
    Function to get SMB characteristic Loads

    Parameters:

        month_ao_hvac_res_net   (np.ndarray)       : Array containing | month-ao-ac-sh-residue-net energies
        epoch_ao_hvac_true      (np.ndarray)       : Array containing | epoch-ao-ac-sh energies
        disagg_input_object     (dict)             : Dictionary containing all input attributes
        disagg_output_object    (dict)             : Dictionary containing all output attributes
        column_index            (dict)             : Dictionary containing column identifier indices of ao-ac-sh
        logger_base             (logging object)   : Writes logs during code flow

    Returns:

        month_ao_hvac_res_net   (np.ndarray)      : Array containing | month-ao-ac-sh-residue-net energies (Processed)
        epoch_ao_hvac_true      (np.ndarray)         : Array containing | epoch-ao-ac-sh energies (Processed)
    """

    smb_params = get_smb_params()

    # initializing logger object
    logger_local = logger_base.get("logger").getChild("op_ao")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_op_ao = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting input data and bill cycle indexes
    epoch_input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    month_epoch, idx_2, month_idx = scipy.unique(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX], return_index=True,
                                                 return_inverse=True)

    logger_op_ao.info("preparing input df |")
    # preparing input data for operational load and extra ao
    input_df = prepare_input_df(epoch_input_data, disagg_input_object)
    appliance_df = prepare_appliance_df(input_df, disagg_output_object, epoch_ao_hvac_true, column_index)
    appliance_df_deepcopy = copy.deepcopy(appliance_df)
    logger_op_ao.info("getting base maps")

    # getting consumption maps for operational load and extra ao estimates
    energy_map, temperature_map, baseload_map, external_light_map, grey_map = get_base_maps(appliance_df_deepcopy)

    # Ensuring extra-AO Sanity
    logger_op_ao.info("Sanitizing extra AO base load |")
    grey_map = get_sanitized_grey_data(grey_map, baseload_map)

    logger_op_ao.info("getting hvac maps |")

    # getting hvac maps
    ac_map, sh_map, residue_map = get_hvac_maps(appliance_df_deepcopy)

    # getting valid days and open close identifier arrays for smb

    open_close_table = disagg_input_object.get('switch').get('smb').get('open_close_table')
    open_close_map = pd.DataFrame(open_close_table, columns=residue_map.columns, index=residue_map.index)
    disagg_input_object['switch']['smb']['open_close_table_write_day'] = open_close_map

    # getting residue only in work hours
    residue_work_hours = residue_map * open_close_map
    day_validity = (np.sum(residue_work_hours, axis=1) > 0) * 1
    valid_day_residue = residue_work_hours.loc[day_validity == 1, :]

    # get hour wise reference operational load
    logger_op_ao.info("Getting hour wise operational load |")
    operational_map = copy.deepcopy(residue_work_hours)
    operational_map, hour_medians, operational_values = get_hour_wise_operational(valid_day_residue, operational_map)

    # get hvac open-close maps based on base open-close map
    ac_map_open = ac_map * open_close_map
    ac_map_close = ac_map - ac_map_open
    sh_map_open = sh_map * open_close_map
    sh_map_close = sh_map - sh_map_open
    ac_global = copy.deepcopy(ac_map_open.values)
    sh_global = copy.deepcopy(sh_map_open.values)

    # homogenizing hvac maps
    logger_op_ao.info("Homogenizing consumption maps |")
    operational_values, ac_global, sh_global = homogenize_hvac_maps(hour_medians, operational_values, open_close_map,
                                                                    ac_global, sh_global)
    # updating operational and hvac loads after homogenization
    operational_map[:] = operational_values
    ac_map_open[:] = ac_global
    sh_map_open[:] = sh_global

    # get updated residues maps from updated hvac maps
    residue_map = energy_map - (baseload_map + ac_map_open + ac_map_close + sh_map_open +
                                sh_map_close + grey_map + external_light_map)
    residue_work_hours = residue_map * open_close_map

    month_identifier = np.char.add(np.array([index.year for index in operational_map.index.values]).astype(str),
                                   np.array([index.month for index in operational_map.index.values]).astype(str))

    unique_months = np.unique(month_identifier)

    # reading updated operational loads
    operational_values = copy.deepcopy(operational_map.values)
    operational_values[day_validity == 0, :] = 0

    # assigning updated hvac loads separately to open and close hours
    ac_map_open = ac_map * open_close_map
    ac_map_close = ac_map - ac_map_open
    sh_map_open = sh_map * open_close_map
    sh_map_close = sh_map - sh_map_open
    ac_global = copy.deepcopy(ac_map_open.values)
    sh_global = copy.deepcopy(sh_map_open.values)

    # creating a bag of key attributes for getting month level operational load post homogenization
    month_level_dict = {'unique_months': unique_months, 'day_validity': day_validity,
                        'month_identifier': month_identifier, 'residue_work_hours': residue_work_hours,
                        'operational_values': operational_values, 'open_close_map': open_close_map,
                        'ac_global': ac_global, 'sh_global': sh_global, 'operational_map': operational_map}

    # estimating month level operational load
    logger_op_ao.info("Getting month level operational load |")
    operational_values, ac_global, sh_global = get_month_level_operational_load(month_level_dict, logger_pass)

    # updating operational load and hvac loads post operational estimates
    operational_map[:] = operational_values
    ac_map_open[:] = ac_global
    sh_map_open[:] = sh_global

    # getting residue left after ao, hvac and operational removal
    residue_map = energy_map - (baseload_map + ac_map_open + ac_map_close + sh_map_open +
                                sh_map_close + grey_map + external_light_map)
    operational_map[residue_map <= operational_map] = residue_map
    others_map = residue_map - operational_map

    # Handling negative residues by ac and sh
    logger_op_ao.info("Sanitizing hvac by ensuring no negative value flows in others |")
    ac_map_open, sh_map_open = getting_others_sanity_by_hvac(others_map, ac_map_open, sh_map_open)
    residue_map = energy_map - (baseload_map + ac_map_open + ac_map_close + sh_map_open +
                                sh_map_close + grey_map + external_light_map)
    others_map = residue_map - operational_map

    # Handling neg res by operational
    logger_op_ao.info("Sanitizing operational by ensuring no negative value flows in others |")
    operational_map = getting_others_sanity_by_op(others_map, operational_map)
    residue_map = energy_map - (baseload_map + ac_map_open + ac_map_close + sh_map_open +
                                sh_map_close + grey_map + external_light_map)
    others_map = residue_map - operational_map

    # -------------- Update SMB consumption to switch object ------
    logger_op_ao.info("Updating switch object of SMB |")
    factor = Cgbdisagg.SEC_IN_HOUR / disagg_input_object.get('config').get('sampling_rate')

    disagg_input_object['switch']['smb']['operational_load'] = np.nanmedian(
        operational_map[operational_map > 0]) * factor
    disagg_input_object['switch']['smb']['cooling_open_median'] = np.nanmedian(ac_map_open[ac_map_open > 0]) * factor
    disagg_input_object['switch']['smb']['cooling_close_median'] = np.nanmedian(ac_map_close[ac_map_close > 0]) * factor
    disagg_input_object['switch']['smb']['heating_open_median'] = np.nanmedian(sh_map_open[sh_map_open > 0]) * factor
    disagg_input_object['switch']['smb']['heating_close_median'] = np.nanmedian(sh_map_close[sh_map_close > 0]) * factor
    disagg_input_object['switch']['smb']['baseload_median'] = np.nanmedian(baseload_map[baseload_map > 0]) * factor
    disagg_input_object['switch']['smb']['residue_median'] = np.nanmedian(others_map[others_map > 0]) * factor

    # ----------- Generate SMB Heatmaps --------------------------
    plot_condition = (disagg_input_object['switch']['plot_level'] >= 1) and not \
        (disagg_input_object['config']['disagg_mode'] == 'mtd')

    if plot_condition:
        # prepare attribute dictionary object for plotting
        logger_op_ao.info("Heatmaps for smb will be generated |")
        plot_info = {'energy_heatmap': energy_map,
                     'temperature_heatmap': temperature_map,
                     'bl_heatmap': baseload_map,
                     'ao_grey': grey_map,
                     'operational_heatmap': operational_map,
                     'ac_heatmap_open': ac_map_open,
                     'ac_heatmap_close': ac_map_close,
                     'ac_heatmap': ac_map_open + ac_map_close,
                     'sh_heatmap_open': sh_map_open,
                     'sh_heatmap_close': sh_map_close,
                     'sh_heatmap': sh_map_open + sh_map_close,
                     'others_heatmap': others_map,
                     'open_close': open_close_map,
                     'external_lighting': external_light_map,
                     'alt_open_close': disagg_output_object.get('alt_open_close_table')
                    }

    # align epochs of all consumption
    epoch_df = input_df.pivot_table(index='date', columns=['time'], values='epoch', aggfunc=np.min)
    epochs = epoch_df.values.flatten()
    baseload = baseload_map.values.flatten()
    extra_ao = grey_map.values.flatten()
    external_light = external_light_map.values.flatten()
    operational_map = operational_map.values.flatten()
    ac_map_open = ac_map_open.values.flatten()
    ac_map_close = ac_map_close.values.flatten()
    sh_map_open = sh_map_open.values.flatten()
    sh_map_close = sh_map_close.values.flatten()
    others_map = others_map.values.flatten()
    open_close_values = open_close_map.values.flatten()
    logger_op_ao.info("aligned epochs of all consumption |")

    # initialize consumption values
    input_df['external_light'] = 0
    input_df['baseload'] = 0
    input_df['extra_ao'] = 0
    input_df['operational_heatmap'] = 0
    input_df['ac_heatmap_open'] = 0
    input_df['ac_heatmap_close'] = 0
    input_df['sh_heatmap_open'] = 0
    input_df['sh_heatmap_close'] = 0
    input_df['others_heatmap'] = 0

    # update consumption values
    _, idx_mem_1, idx_mem_2 = np.intersect1d(epochs, input_df['epoch'], return_indices=True)
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['external_light']] = external_light[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['bl']] = baseload[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['x-ao']] = extra_ao[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['op']] = operational_map[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['ac_open']] = ac_map_open[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['ac_close']] = ac_map_close[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['sh_open']] = sh_map_open[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['sh_close']] = sh_map_close[idx_mem_1]
    input_df.iloc[idx_mem_2, smb_params['input_df_col']['oth']] = others_map[idx_mem_1]

    logger_op_ao.info("updated consumption values |")

    # keep a backup copy of consumption values
    input_df_backup = copy.deepcopy(input_df)
    input_df_backup['open_close'] = 0
    input_df_backup.iloc[idx_mem_2, -1] = open_close_values[idx_mem_1]
    input_df_backup['provisional_operational_removed_data'] = copy.deepcopy(disagg_input_object.get('switch')
                                                                            .get('hvac').get('operational_removed'))
    disagg_input_object['switch']['smb']['open_close_table_write_epoch'] = np.array(input_df_backup['open_close'])

    # SMB v2.0 Change
    if plot_condition:
        generate_app_heatmap_smb(disagg_input_object, input_df_backup, plot_info)

    # checking sanity of all consumption values
    input_df['baseload'][input_df['baseload'] < 0] = 0
    input_df['extra_ao'][input_df['extra_ao'] < 0] = 0
    input_df['operational_heatmap'][input_df['operational_heatmap'] < 0] = 0
    input_df['ac_heatmap_open'][input_df['ac_heatmap_open'] < 0] = 0
    input_df['ac_heatmap_close'][input_df['ac_heatmap_close'] < 0] = 0
    input_df['sh_heatmap_open'][input_df['sh_heatmap_open'] < 0] = 0
    input_df['sh_heatmap_close'][input_df['sh_heatmap_close'] < 0] = 0

    # aggregating estimates at BC level for monthly estimates
    month_baseload = np.bincount(month_idx, input_df['baseload'])
    month_baseload = month_baseload / Cgbdisagg.WH_IN_1_KWH
    month_extra_ao = np.bincount(month_idx, input_df['extra_ao'])
    month_extra_ao = month_extra_ao / Cgbdisagg.WH_IN_1_KWH
    month_operational = np.bincount(month_idx, input_df['operational_heatmap'])
    month_operational = month_operational / Cgbdisagg.WH_IN_1_KWH
    month_ac = np.bincount(month_idx, input_df['ac_heatmap_open'])
    month_ac = month_ac / Cgbdisagg.WH_IN_1_KWH
    month_ac_close = np.bincount(month_idx, input_df['ac_heatmap_close'])
    month_ac_close = month_ac_close / Cgbdisagg.WH_IN_1_KWH
    month_sh = np.bincount(month_idx, input_df['sh_heatmap_open'])
    month_sh = month_sh / Cgbdisagg.WH_IN_1_KWH
    month_sh_close = np.bincount(month_idx, input_df['sh_heatmap_close'])
    month_sh_close = month_sh_close / Cgbdisagg.WH_IN_1_KWH
    month_others = np.bincount(month_idx, input_df['others_heatmap'])
    month_others = month_others / Cgbdisagg.WH_IN_1_KWH
    logger_op_ao.info("month values generated |")

    # Assigning estimates at BC level for plotting and writing
    monthly_info = {'day_extra_ao': {}, 'month_epoch': month_epoch, 'month_baseload': month_baseload,
                    'month_extra_ao': month_extra_ao, 'month_operational': month_operational, 'month_ac': month_ac,
                    'month_ac_close': month_ac_close, 'month_sh': month_sh, 'month_sh_close': month_sh_close,
                    'month_others': month_others}

    # Assigning extra ao at month level
    for month in month_epoch:
        month_df = input_df[input_df['month'] == month]
        day_epoch, idx_2, day_idx = scipy.unique(month_df['day'], return_index=True, return_inverse=True)
        extra_ao_at_day = np.bincount(day_idx, month_df['extra_ao'])
        monthly_info['day_extra_ao'][month] = {}
        monthly_info['day_extra_ao'][month]['day_epoch'] = day_epoch.astype(float)
        monthly_info['day_extra_ao'][month]['extra_ao_at_day'] = extra_ao_at_day.astype(float)

    disagg_output_object['special_outputs']['smb_outputs'] = monthly_info

    logger_op_ao.info("month info assigned |")

    # generating month level consumption plots
    if plot_condition:
        plot_monthly_bar_smb(month_ao_hvac_res_net, disagg_input_object, disagg_output_object, monthly_info)

    # assigning month level consumption values to main month level object
    month_ao_hvac_res_net[:, 1] = month_baseload
    month_ao_hvac_res_net[:, 2] = month_ac + month_ac_close
    month_ao_hvac_res_net[:, 3] = month_sh + month_sh_close

    # assigning epoch level consumption values to main epoch level object
    epoch_ao_hvac_true[:, 1] = input_df['baseload']
    epoch_ao_hvac_true[:, 2] = input_df['ac_heatmap_open'] + input_df['ac_heatmap_close']
    epoch_ao_hvac_true[:, 3] = input_df['sh_heatmap_open'] + input_df['sh_heatmap_close']

    # preparing month summary of ao, hvac, operational, extra-ao, and residue consumption
    month_ao_hvac_res_net = np.c_[month_ao_hvac_res_net, month_extra_ao, month_ac, month_ac_close, month_sh,
                                  month_sh_close, month_operational]

    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, input_df['extra_ao'], input_df['ac_heatmap_open'],
                               input_df['ac_heatmap_close'], input_df['sh_heatmap_open'], input_df['sh_heatmap_close'],
                               input_df['operational_heatmap'], input_df['external_light']]

    logger_op_ao.info("SMB Appliance results ready |")

    return month_ao_hvac_res_net, epoch_ao_hvac_true
