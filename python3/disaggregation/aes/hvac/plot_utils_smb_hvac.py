"""
Author - Mirambika Sikdar
Date - 21/12/2023
All plotting functions used in smb hvac pipeline
"""

# Import python packages
import os
import pandas as pd
import numpy as np
import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.init_hourly_smb_hvac_params import hvac_static_params


def bar_appmap_baseline(generate_plot, disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index,
                        epoch_ao_hvac_true):
    """
    Function to postprocess hvac results in case of over/under estimation, except in mtd mode

    Parameters:

        generate_plot (bool)                : Boolean flag indicating if the plots has to be generated or not
        disagg_input_object (dict)          : Dictionary containing all input attributes
        disagg_output_object (dict)         : Dictionary containing all output attributes
        month_ao_hvac_res_net (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        column_index (dict)                 : Dictionary containing column identifier indices of ao-ac-sh
        epoch_ao_hvac_true (np.ndarray)       : Array containing | epoch-ao-ac-sh energies

    Returns:
    """

    # plot monthly bar if enabled
    if generate_plot and (disagg_input_object['switch']['plot_level'] >= 1):
        plot_monthly_bar(disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index, 'true')

    # plot heatmap if enabled
    if generate_plot and (disagg_input_object['switch']['plot_level'] >= 3) and \
            (disagg_input_object['config']['disagg_mode'] != 'mtd'):
        generate_appliance_heatmap_new(disagg_input_object, disagg_output_object, epoch_ao_hvac_true, 'true')


def generate_appliance_heatmap_new(disagg_input_object, disagg_output_object, epoch_ao_hvac_true, stage):
    """
    Function to dump the appmap in order = Net-Baseload-AO Cooling - AC - AO Heating - SH - Residue

    Paramaters:

        disagg_input_object (dict)          : Dictionary containing all input attributes
        disagg_output_object (dict)         : Dictionary containing all output attributes
        epoch_ao_hvac_true (np.ndarray)       : Array containing | epoch-ao-ac-sh energies
        stage (dict): appliance identifier, whether AC or SH

    Returns:
        None
    """

    static_params = hvac_static_params()

    features = disagg_output_object['analytics']['values']
    global_config = disagg_input_object['config']
    disagg_mode = global_config.get('disagg_mode', '').lower()

    if (disagg_mode != 'historical') and (disagg_mode != 'incremental'):
        print("HVAC Heatmap plotting is disabled for MTD mode-")
        return

    hvac_detection = disagg_output_object['hvac_debug']['detection']
    hvac_estimation = disagg_output_object['hvac_debug']['estimation']
    input_df = pd.DataFrame(disagg_input_object['input_data'])
    appliance_df = pd.DataFrame()
    appliance_df['ao'] = disagg_output_object['epoch_estimate'][:, disagg_output_object['output_write_idx_map']['ao']]

    appliance_df['ao_baseload'] = disagg_output_object['ao_seasonality']['epoch_baseload']
    appliance_df['ao_cooling'] = disagg_output_object['ao_seasonality']['epoch_cooling']
    appliance_df['ao_heating'] = disagg_output_object['ao_seasonality']['epoch_heating']
    appliance_df['ao_grey'] = disagg_output_object['ao_seasonality']['epoch_grey']

    # 2nd column in epoch_ao_hvac matrix is ac and 3rd column is sh
    appliance_df['ac'] = epoch_ao_hvac_true[:, 2]
    appliance_df['sh'] = epoch_ao_hvac_true[:, 3]

    columns = Cgbdisagg.INPUT_COLUMN_NAMES
    input_df.columns = columns

    input_data_raw = input_df[['epoch', 'consumption', 'temperature']]
    input_data_raw['timestamp'] = pd.to_datetime(input_data_raw['epoch'], unit='s')

    timezone = disagg_input_object['home_meta_data']['timezone']
    try:
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC',
                                                                              ambiguous='infer').dt.tz_convert(timezone)
    except (ValueError, IndexError, KeyError):
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC',
                                                                              ambiguous='NaT').dt.tz_convert(timezone)

    input_data_raw['date'] = input_data_raw['timestamp'].dt.date
    input_data_raw['time'] = input_data_raw['timestamp'].dt.time
    input_data_raw['year'] = input_data_raw['timestamp'].dt.year
    input_data_raw['month'] = input_data_raw['timestamp'].dt.month

    appliance_df['date'] = input_data_raw['date']
    appliance_df['time'] = input_data_raw['time']
    appliance_df['residue'] = input_data_raw['consumption'] - (
            appliance_df['ao'] + appliance_df['ac'] + appliance_df['sh'])

    input_data_raw['consumption'][input_data_raw['consumption'] > input_data_raw['consumption'].quantile(0.97)] = \
        input_data_raw['consumption'].quantile(0.97)
    input_data_raw['temperature'][input_data_raw['temperature'] > input_data_raw['temperature'].quantile(0.97)] = \
        input_data_raw['temperature'].quantile(0.97)

    input_data_raw['is_mid_temperature'] = \
        np.logical_and(input_data_raw['temperature'] >= hvac_detection['mid']['temp'][0][0],
                       input_data_raw['temperature'] <= hvac_detection['mid']['temp'][1][0]).astype(int)

    input_data_raw['is_ac_temperature'] = input_data_raw['temperature'] >= hvac_estimation['cdd']['setpoint']
    input_data_raw['is_ac_temperature'] = input_data_raw.get('is_ac_temperature').astype(int)

    input_data_raw['is_sh_temperature'] = input_data_raw['temperature'] <= hvac_estimation['hdd']['setpoint']
    input_data_raw['is_sh_temperature'] = input_data_raw.get('is_sh_temperature').astype(int)

    energy_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='consumption', aggfunc=sum)
    energy_heatmap = energy_heatmap.fillna(0)
    energy_heatmap = energy_heatmap.astype(int)

    temperature_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='temperature', aggfunc=sum)
    temperature_heatmap = temperature_heatmap.fillna(0)
    temperature_heatmap = temperature_heatmap.astype(int)

    mid_temperature_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='is_mid_temperature',
                                                         aggfunc=sum)
    mid_temperature_heatmap = mid_temperature_heatmap.fillna(0)
    mid_temperature_heatmap = mid_temperature_heatmap.astype(int)

    ac_temperature_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='is_ac_temperature',
                                                        aggfunc=sum)
    ac_temperature_heatmap = ac_temperature_heatmap.fillna(0)
    ac_temperature_heatmap = ac_temperature_heatmap.astype(int)

    sh_temperature_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='is_sh_temperature',
                                                        aggfunc=sum)
    sh_temperature_heatmap = sh_temperature_heatmap.fillna(0)
    sh_temperature_heatmap = sh_temperature_heatmap.astype(int)

    ao_baseload_heatmap = appliance_df.pivot_table(index='date', columns=['time'], values='ao_baseload',
                                                   aggfunc=sum)
    ao_baseload_heatmap = ao_baseload_heatmap.fillna(0)
    ao_baseload_heatmap = ao_baseload_heatmap.astype(int)

    ao_cooling_heatmap = appliance_df.pivot_table(index='date', columns=['time'], values='ao_cooling',
                                                  aggfunc=sum)
    ao_cooling_heatmap = ao_cooling_heatmap.fillna(0)
    ao_cooling_heatmap = ao_cooling_heatmap.astype(int)

    ao_heating_heatmap = appliance_df.pivot_table(index='date', columns=['time'], values='ao_heating',
                                                  aggfunc=sum)
    ao_heating_heatmap = ao_heating_heatmap.fillna(0)
    ao_heating_heatmap = ao_heating_heatmap.astype(int)

    ac_heatmap = appliance_df.pivot_table(index='date', columns=['time'], values='ac',
                                          aggfunc=sum)
    ac_heatmap = ac_heatmap.fillna(0)
    ac_heatmap = ac_heatmap.astype(int)

    sh_heatmap = appliance_df.pivot_table(index='date', columns=['time'], values='sh',
                                          aggfunc=sum)
    sh_heatmap = sh_heatmap.fillna(0)
    sh_heatmap = sh_heatmap.astype(int)

    residue_heatmap = appliance_df.pivot_table(index='date', columns=['time'], values='residue',
                                               aggfunc=sum)
    residue_heatmap = residue_heatmap.fillna(0)
    residue_heatmap = residue_heatmap.astype(int)

    fig_heatmap, axn = plt.subplots(1, 8, sharey=True)
    fig_heatmap.set_size_inches(25, 10)

    residual_stability = disagg_output_object['analytics']['values']['residual_stability']

    fig_heatmap.suptitle('\n App-Maps   |    Pilot id : {}   |   Sampling rate : {}   |   Stability : {} \n \n '
                         'AC : {} {}F : mu {} : std {}        '
                         'SH : {} {}F : mu {} : std {} '.format(global_config['pilot_id'],
                                                                global_config['sampling_rate'], residual_stability,
                                                                features['cooling']['setpoint']['exist'],
                                                                features['cooling']['setpoint']['setpoint'],
                                                                features['cooling']['detection']['means'],
                                                                features['cooling']['detection']['std'],
                                                                features['heating']['setpoint']['exist'],
                                                                features['heating']['setpoint']['setpoint'],
                                                                features['heating']['detection']['means'],
                                                                features['heating']['detection']['std']),
                         fontsize=10)

    e_max = np.max(energy_heatmap.max())
    t_max = np.max(temperature_heatmap.max())
    t_min = np.min(input_data_raw['temperature'])

    if t_max < 110:
        t_max = 110
    elif t_max > 120:
        t_max = 120

    if t_min > 10:
        t_min = 10

    sns.heatmap(energy_heatmap, ax=axn.flat[0], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(temperature_heatmap, ax=axn.flat[1], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=t_min, vmax=t_max)
    sns.heatmap(ao_baseload_heatmap, ax=axn.flat[2], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(ao_cooling_heatmap, ax=axn.flat[3], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(ac_heatmap, ax=axn.flat[4], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(ao_heating_heatmap, ax=axn.flat[5], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(sh_heatmap, ax=axn.flat[6], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(residue_heatmap, ax=axn.flat[7], cmap='hot', cbar=True, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)

    axn.flat[0].set_title("Raw Energy")
    axn.flat[0].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[1].set_title("Temperature (F)")
    axn.flat[1].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[2].set_title("Baseload")
    axn.flat[2].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[3].set_title("AO Cooling")
    axn.flat[3].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[4].set_title("Demand Cooling")
    axn.flat[4].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[5].set_title("AO Heating")
    axn.flat[5].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[6].set_title("Demand Heating")
    axn.flat[6].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[7].set_title("Residue")
    axn.flat[7].tick_params(axis='x', which='major', labelsize=7)

    # Turning off axes labels

    for axis_idx in range(len(axn.flat)):
        x_axis = axn.flat[axis_idx].get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)

        y_axis = axn.flat[axis_idx].get_yaxis()
        y_label = y_axis.get_label()
        y_label.set_visible(False)

    plt.yticks(rotation=0)

    plot_dir = static_params.get('path').get('hvac_plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_dir = plot_dir + '/' + disagg_input_object['config']['uuid']
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = plot_dir + '/heatmap_appliance_new_' + stage + '_' + disagg_input_object['config']['uuid'] + '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location, dpi=250)
    plt.close()
    del fig_heatmap


def get_title_text(hvac_modes):
    """
    Function to get the title text for regression figure, for both AC/SH

    Parameters:
        hvac_modes (dict)       : HVAC Mode identifier
    Returns:
        title_text (str)        : String to be displayed on top of plots
    """

    title_text = ''

    for mode in hvac_modes.keys():
        for cluster in hvac_modes[mode]['cluster_info'].keys():

            if hvac_modes[mode]['cluster_info'][cluster]['validity']:
                title_text = title_text + 'Regression Kind : {} | Coefficient : {} \n '.format(
                    hvac_modes[mode]['cluster_info'][cluster]['regression_kind'],
                    np.around(hvac_modes[mode]['cluster_info'][cluster]['coefficient'][0][0]))

    if title_text == '':
        title_text = 'Cooling does not exist  '

    return title_text


def plot_predictions_trend(cluster_info, hvac_modes, modes, axis_flat, marker, appliance):
    '''
    Function to plot model predictions as trend line

    Parameters:
        cluster_info (dict)         : Dictionary containing HVAC clusters related key information
        hvac_modes (dict)           : Dictionary containing HVAC modes
        modes (int)                 : Integers indicating modes
        axis_flat (np.ndarray)      : Axis array for plotting
        marker (list)               : Markers to be used as plotting marks
        appliance (str)             : Appliance identifier (AC/SH)
    Returns:
        None
    '''

    if appliance == 'AC':
        axis_identifier = 0
    else:
        axis_identifier = 1

    for cluster_id in cluster_info.keys():

        if cluster_info[cluster_id]['validity']:

            cluster_df = hvac_modes[modes]['regression_df'][
                hvac_modes[modes]['regression_df']['day_hvac_cluster'] == cluster_id]
            predictions = np.zeros(cluster_df['degree_day'].shape)
            if cluster_info[cluster_id]['regression_kind'] == 'linear':
                predictions = cluster_info[cluster_id]['model'].predict(
                    np.array(cluster_df['degree_day']).reshape(-1, 1))
            elif cluster_info[cluster_id]['regression_kind'] == 'root':
                predictions = cluster_info[cluster_id]['model'].predict(
                    np.sqrt(np.array(cluster_df['degree_day'])).reshape(-1, 1))

            axis_flat[axis_identifier].scatter(cluster_df['degree_day'], predictions, marker=marker[modes], s=10,
                                               c='black', alpha=0.5)


def plot_regression_clusters_smb(hvac_input_data, x_hour_hvac_by_mode, estimation_debug, global_config):
    """
    Function to plot Regression

    Parameters:
        hvac_input_data (np.ndarray)       : Array containing consumption points
        x_hour_hvac_by_mode (np.ndarray)   : Mode wise qualified points for regression
        estimation_debug (dict)            : Dictionary containing estimation related key information
        global_config (dict)               : Dictionary containing user level key config parameters
    Returns:
        None
    """

    static_params = hvac_static_params()

    figure, axis_array = plt.subplots(2, 2)
    axis_flat = axis_array.flatten()

    cooling_modes = x_hour_hvac_by_mode['cooling']
    heating_modes = x_hour_hvac_by_mode['heating']

    marker = ['o', 's']
    ac_color = ['blue', 'cyan', 'slateblue', 'blueviolet']
    sh_color = ['red', 'magenta']

    ylabel = "Consumption (Wh)"

    for modes in cooling_modes.keys():

        ac_scatter_df_mode = cooling_modes[modes]['regression_df']

        # noinspection PyBroadException
        try:
            cooling_color_map = {-1: 'grey', 0: ac_color[modes], 1: ac_color[modes], 2: ac_color[modes],
                                 3: ac_color[modes]}
            cooling_bar_color = [cooling_color_map[i] for i in ac_scatter_df_mode['day_hvac_cluster']]
            axis_flat[0].scatter(ac_scatter_df_mode['degree_day'], ac_scatter_df_mode['filter_cons'],
                                 marker=marker[modes], s=10, c=cooling_bar_color, alpha=0.5)
        except (IndexError, KeyError):
            cooling_color_map = {-1: 'grey', 0: ac_color[modes], 1: ac_color[modes], 2: ac_color[modes],
                                 3: ac_color[modes]}

        axis_flat[0].set_ylabel(ylabel, fontsize=7)
        axis_flat[0].set_xlabel('cdd ', fontsize=7)
        axis_flat[0].tick_params(axis='both', which='major', labelsize=5)
        cooling_title_text = get_title_text(cooling_modes)
        axis_flat[0].set_title(cooling_title_text[:-2], fontsize=7)
        ac_cluster_info = cooling_modes[modes]['cluster_info']
        plot_predictions_trend(ac_cluster_info, cooling_modes, modes, axis_flat, marker, 'AC')

    for modes in heating_modes.keys():

        sh_scatter_df_mode = heating_modes[modes]['regression_df']

        # noinspection PyBroadException
        try:
            heating_color_map = {-1: 'grey', 0: sh_color[modes], 1: sh_color[modes], 2: sh_color[modes],
                                 3: sh_color[modes]}
            heating_bar_color = [heating_color_map[i] for i in sh_scatter_df_mode['day_hvac_cluster']]
            axis_flat[1].scatter(sh_scatter_df_mode['degree_day'], sh_scatter_df_mode['filter_cons'],
                                 marker=marker[modes], s=10, c=heating_bar_color, alpha=0.5)
        except (IndexError, KeyError):
            heating_color_map = {-1: 'grey', 0: sh_color[modes], 1: sh_color[modes], 2: sh_color[modes],
                                 3: sh_color[modes]}

        axis_flat[1].set_ylabel(ylabel, fontsize=7)
        axis_flat[1].set_xlabel('hdd ', fontsize=7)
        axis_flat[1].tick_params(axis='both', which='major', labelsize=5)
        heating_title_text = get_title_text(heating_modes)
        axis_flat[1].set_title(heating_title_text[:-2], fontsize=7)
        sh_cluster_info = heating_modes[modes]['cluster_info']
        plot_predictions_trend(sh_cluster_info, heating_modes, modes, axis_flat, marker, 'SH')

    _, day_idx = np.unique(hvac_input_data[:, Cgbdisagg.INPUT_DAY_IDX], return_inverse=True)
    temp = hvac_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    keep = ~np.isnan(temp)
    daily_consumption = np.bincount(day_idx[keep], temp[keep])
    temp = np.maximum(
        (-hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] + float(estimation_debug['hdd']['setpoint'])), 0)
    temp[np.isnan(temp)] = 0
    keep = ~np.isnan(temp)
    hdd_daily = np.bincount(day_idx[keep], temp[keep])
    temp = np.maximum(
        (hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX] - float(estimation_debug['cdd']['setpoint'])), 0)
    temp[np.isnan(temp)] = 0
    keep = ~np.isnan(temp)
    cdd_daily = np.bincount(day_idx[keep], temp[keep])

    axis_flat[3].scatter(hdd_daily, daily_consumption, marker='o', s=10, c='red', alpha=0.5)
    axis_flat[3].set_title('hdd - all days', fontsize=7)
    axis_flat[3].set_ylabel(ylabel, fontsize=7)
    axis_flat[3].set_xlabel('hdd ', fontsize=7)
    axis_flat[3].tick_params(axis='both', which='major', labelsize=5)

    axis_flat[2].scatter(cdd_daily, daily_consumption, marker='o', s=10, c='blue', alpha=0.5)
    axis_flat[2].set_title('cdd - all days', fontsize=7)
    axis_flat[2].set_ylabel(ylabel, fontsize=7)
    axis_flat[2].set_xlabel('cdd ', fontsize=7)
    axis_flat[2].tick_params(axis='both', which='major', labelsize=5)

    figure.tight_layout()

    plot_dir = static_params.get('path').get('hvac_plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_dir = plot_dir + '/' + global_config['uuid']
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = plot_dir + '/regression_clusters_' + global_config['uuid'] + '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    if plot_dir:
        plt.savefig(image_location, dpi=250)

    plt.close()
    del (figure)


def plot_monthly_bar(disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index, stage):
    """
    Function to generate month level hvac consumtion bar plots

    Parameters:

        disagg_input_object (dict)          : Dictionary containing all input attributes
        disagg_output_object (dict)         : Dictionary containing all output attributes
        month_ao_hvac_res_net (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        column_index (dict)                 : Dictionary containing column identifier indices of ao-ac-sh
        stage (str)                         : String to identify in bar plot is being made before/after post-processing

    Returns:
        None
    """

    if stage == 'processed':

        static_params = hvac_static_params()

        features = disagg_output_object['analytics']['values']
        residual_stability = features['residual_stability']
        residual_rsquare = features['residual_rsquare']

        global_config = disagg_input_object.get('config')

        month_labels = [datetime.utcfromtimestamp(month_ao_hvac_res_net[i, 0]).strftime('%b-%Y') for i in
                        range(month_ao_hvac_res_net.shape[0])]

        ao_cooling = disagg_output_object['ao_seasonality']['cooling'] / 1000
        ao_heating = disagg_output_object['ao_seasonality']['heating'] / 1000
        ao_baseload = disagg_output_object['ao_seasonality']['baseload'] / 1000

        ac_consumption = month_ao_hvac_res_net[:, column_index['ac']]
        sh_consumption = month_ao_hvac_res_net[:, column_index['sh']]

        residual = month_ao_hvac_res_net[:, 4]

        if disagg_input_object['switch']['plot_level'] >= 1:

            figure, axis_array = plt.subplots(figsize=(10, 7))
            array_twinx = axis_array.twinx()

            width = 0.85
            index = np.arange(len(month_labels))
            p1 = axis_array.bar(index, ao_baseload, width, color='limegreen', alpha=1, edgecolor='green')
            p8 = axis_array.bar(index, ao_cooling, width, bottom=ao_baseload,
                                color='blue', alpha=1, edgecolor='green')
            p2 = axis_array.bar(index, ac_consumption, width, bottom=ao_baseload + ao_cooling,
                                color='dodgerblue', alpha=1, edgecolor='blue')
            p9 = axis_array.bar(index, ao_heating, width, bottom=ao_baseload + ao_cooling + ac_consumption,
                                color='red', alpha=1, edgecolor='green')
            p3 = axis_array.bar(index, sh_consumption, width,
                                bottom=ao_baseload + ao_cooling + ac_consumption + ao_heating,
                                color='orangered', alpha=1, edgecolor='red')
            p4 = axis_array.bar(index, residual, width,
                                bottom=ao_baseload + ao_cooling + ac_consumption + ao_heating + sh_consumption,
                                color='black', alpha=0.85, edgecolor='black')

            axis_array.set_ylabel('Monthly Consumption (kwh)', fontsize=9)
            axis_array.set_title('Monthly Disagg   |    Pilot id : {}   |   Sampling rate : {}   |   '
                                 'Stability : {}  |  Residue R2 : {} \n \n '
                                 'AC : {} {}F : mu {} : std {}        '
                                 'SH : {} {}F : mu {} : std {} '.format(global_config['pilot_id'],
                                                                        global_config['sampling_rate'],
                                                                        residual_stability, residual_rsquare,
                                                                        features['cooling']['setpoint']['exist'],
                                                                        features['cooling']['setpoint']['setpoint'],
                                                                        features['cooling']['detection']['means'],
                                                                        features['cooling']['detection']['std'],
                                                                        features['heating']['setpoint']['exist'],
                                                                        features['heating']['setpoint']['setpoint'],
                                                                        features['heating']['detection']['means'],
                                                                        features['heating']['detection']['std']),
                                 fontsize=9)

            axis_array.set_xticks(index)

            epoch_input_data = disagg_input_object['input_data']

            columns = Cgbdisagg.INPUT_COLUMN_NAMES

            epoch_input_data = epoch_input_data[:, :len(columns)]

            input_df = pd.DataFrame(epoch_input_data)

            input_df.columns = columns

            df_min_mean_max = input_df.groupby('month')['temperature'].agg([pd.np.min, pd.np.max, pd.np.mean])

            p5 = array_twinx.plot(index, df_min_mean_max['amin'], 'b:', label='Min Temperature', linewidth=1.5)
            p6 = array_twinx.plot(index, df_min_mean_max['mean'], 'y:', label='Mean Temperature', linewidth=2)
            p7 = array_twinx.plot(index, df_min_mean_max['amax'], 'r:', label='Max Temperature', linewidth=1.5)
            array_twinx.set_ylabel('Temperature (F)', fontsize=7)
            array_twinx.set_yticks([10 * i for i in range(13)])

            axis_array.set_xticklabels(month_labels, fontdict={'fontsize': 7, 'verticalalignment': 'top'}, minor=False,
                                       rotation=90)

            plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]),
                       ('AO', 'AC', 'SH', 'OTHER', 'Min Temperature', 'Mean Temperature', 'Max Temperature',
                        'AO-Cooling', 'AO-Heating'),
                       fontsize=7, loc='upper center', ncol=7, framealpha=0.1)

            plt.tight_layout()

            plot_dir = static_params.get('path').get('hvac_plots')

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            plot_dir = plot_dir + '/' + global_config['uuid']
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            image_location = \
                plot_dir + '/bar_plot_' + stage + '_' + global_config['disagg_mode'] + '_' + global_config['uuid'] + \
                '.png'

            if os.path.isfile(image_location):
                os.remove(image_location)

            if plot_dir:
                plt.savefig(image_location, dpi=250)

            plt.close()

            del figure
