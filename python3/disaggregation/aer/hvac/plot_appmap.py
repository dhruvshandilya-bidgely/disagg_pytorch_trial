"""
Author - Abhinav Srivastava
Date - 22/10/18
Call the hvac disaggregation module and get results
"""

# Import python packages
import os
import pandas as pd
import numpy as np
import scipy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def generate_appliance_heatmap_new(disagg_input_object, disagg_output_object, epoch_ao_hvac_true, column_index, stage):
    """
    Function to dump the appmap in order = Net-Baseload-AO Cooling - AC - AO Heating - SH - Residue

    Paramaters:

        disagg_input_object (dict)          : Dictionary containing all input attributes
        disagg_output_object (dict)         : Dictionary containing all output attributes
        epoch_ao_hvac_true (np.ndarray)       : Array containing | epoch-ao-ac-sh energies
        column_index (dict)           : column name to id mapping
        stage (dict): appliance identifier, whether AC or SH

    Returns:
        None
    """

    static_params = hvac_static_params()
    global_config = disagg_input_object.get('config', {})
    disagg_mode = global_config.get('disagg_mode', '').lower()

    if (disagg_mode != 'historical') and (disagg_mode != 'incremental'):
        print("HVAC Heatmap plotting is disabled for MTD mode-")
        return

    uuid = global_config['uuid']
    features = disagg_output_object['analytics']['values']
    hvac_detection = disagg_output_object['hvac_debug']['detection']
    hvac_estimation = disagg_output_object['hvac_debug']['estimation']
    input_data = disagg_input_object['switch']['hvac_input_data_timed_removed'][:, Cgbdisagg.INPUT_EPOCH_IDX]
    start, end = int(np.nanmin(input_data)), int(np.nanmax(input_data))

    input_df = pd.DataFrame(disagg_input_object['input_data'])
    appliance_df = pd.DataFrame()
    appliance_df['ao'] = disagg_output_object['epoch_estimate'][:, disagg_output_object['output_write_idx_map']['ao']]

    appliance_df['ao_baseload'] = disagg_output_object['ao_seasonality']['epoch_baseload']
    appliance_df['ao_cooling'] = disagg_output_object['ao_seasonality']['epoch_cooling']
    appliance_df['ao_heating'] = disagg_output_object['ao_seasonality']['epoch_heating']
    appliance_df['ao_grey'] = disagg_output_object['ao_seasonality']['epoch_grey']

    # 2nd column in epoch_ao_hvac matrix is ac and 3rd column is sh
    appliance_df['ac'] = epoch_ao_hvac_true[:, column_index.get('ac')]
    appliance_df['sh'] = epoch_ao_hvac_true[:, column_index.get('sh')]

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

    plot_dir = plot_dir + '/' + str(uuid)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = plot_dir + '/{disagg_mode}_heatmap_combined_'.format(disagg_mode=disagg_mode) + stage + '_'
    image_location = image_location + str(uuid) + '_{start}_{end}.png'.format(start=start, end=end)

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location, dpi=250)
    plt.close()
    del fig_heatmap
