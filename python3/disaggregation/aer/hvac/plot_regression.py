"""
Author - Abhinav
Date - 10/10/2018
Estimating HVAC
"""

# import python packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# import function from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def get_title_text(hvac_modes):
    """
    Function to get the title text for regression figure, for both AC/SH

    Parameters:
        hvac_modes (dict)        : HVAC Mode identifier
    Returns:
        title_text (str)         : String to be displayed on top of plots
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


def plot_predictions_trend(cluster_info, hvac_modes, modes, axis_flat, marker, axis_identifier):

    """
    Function to plot model predictions as trend line

    Parameters:
        cluster_info (dict)         : Dictionary containing HVAC clusters related key information
        hvac_modes (dict)           : Dictionary containing HVAC modes
        modes (int)                 : Integers indicating modes
        axis_flat (np.ndarray)      : Axis array for plotting
        marker (list)               : Markers to be used as plotting marks
    Returns:
        None
    """

    for cluster_id in cluster_info.keys():
        if cluster_info[cluster_id]['validity']:
            cluster_df = hvac_modes[modes]['regression_df'][
                hvac_modes[modes]['regression_df']['day_hvac_cluster'] == cluster_id]

            if cluster_info[cluster_id]['regression_kind'] == 'linear':
                predictions = cluster_info[cluster_id]['model'].predict(np.array(cluster_df['degree_day']).reshape(-1, 1))
                axis_flat[axis_identifier].scatter(cluster_df['degree_day'], predictions, marker=marker[modes], s=10,
                                                   c='black', alpha=0.5)

            elif cluster_info[cluster_id]['regression_kind'] == 'root':
                predictions = cluster_info[cluster_id]['model'].predict(np.sqrt(np.array(cluster_df['degree_day'])).reshape(-1, 1))
                axis_flat[axis_identifier].scatter(cluster_df['degree_day'], predictions, marker=marker[modes], s=10,
                                                   c='black', alpha=0.5)


def plot_regression_clusters(hvac_input_data, x_hour_hvac_by_mode, global_config):

    """
    Function to plot Regression

    Parameters:
        hvac_input_data     (np.ndarray)       : Array containing consumption points
        x_hour_hvac_by_mode (dict)             : Dictionary containing hvac mode related key information
        global_config       (dict)             : Dictionary containing user level key config parameters

    Returns:
        None
    """

    static_params = hvac_static_params()

    figure, axis_array = plt.subplots(2, 2)
    axis_flat = axis_array.flatten()
    disagg_mode = global_config['disagg_mode'].lower()

    cooling_modes = x_hour_hvac_by_mode['cooling']
    heating_modes = x_hour_hvac_by_mode['heating']

    marker = ['o', 's']
    ac_color = ['blue', 'cyan', 'slateblue', 'blueviolet']
    sh_color = ['red', 'magenta']
    mode_id_to_axis_id_map = {0: 0, 1: 2}

    for modes in cooling_modes.keys():
        axis_identifier = mode_id_to_axis_id_map[modes]
        ac_scatter_df_mode = cooling_modes[modes]['regression_df']

        # noinspection PyBroadException
        try:
            cooling_color_map = {-1: 'grey', 0: ac_color[modes], 1: ac_color[modes], 2: ac_color[modes], 3: ac_color[modes]}
            cooling_bar_color = [cooling_color_map[i] for i in ac_scatter_df_mode['day_hvac_cluster']]
            axis_flat[axis_identifier].scatter(ac_scatter_df_mode['degree_day'], ac_scatter_df_mode['filter_cons'], marker=marker[modes], s=10, c=cooling_bar_color, alpha=0.5)
        except (IndexError, KeyError):
            cooling_color_map = {-1: 'grey', 0: ac_color[modes], 1: ac_color[modes], 2: ac_color[modes], 3: ac_color[modes]}

        y_label_text = "Consumption (Wh)"
        axis_flat[0].set_ylabel(y_label_text, fontsize=7)
        axis_flat[0].set_xlabel('cdd ', fontsize=7)
        axis_flat[0].tick_params(axis='both', which='major', labelsize=5)
        axis_flat[0].set_title('Cooling : Mode 0')
        axis_flat[2].set_ylabel(y_label_text, fontsize=7)
        axis_flat[2].set_xlabel('cdd ', fontsize=7)
        axis_flat[2].tick_params(axis='both', which='major', labelsize=5)
        axis_flat[2].set_title('Cooling : Mode 1')

        ac_cluster_info = cooling_modes[modes]['cluster_info']
        plot_predictions_trend(ac_cluster_info, cooling_modes, modes, axis_flat, marker, axis_identifier)

    mode_id_to_axis_id_map = {0: 1, 1: 3}
    for modes in heating_modes.keys():
        axis_identifier = mode_id_to_axis_id_map[modes]
        sh_scatter_df_mode = heating_modes[modes]['regression_df']

        # noinspection PyBroadException
        try:
            heating_color_map = {-1: 'grey', 0: sh_color[modes], 1: sh_color[modes], 2: sh_color[modes], 3: sh_color[modes]}
            heating_bar_color = [heating_color_map[i] for i in sh_scatter_df_mode['day_hvac_cluster']]
            axis_flat[axis_identifier].scatter(sh_scatter_df_mode['degree_day'], sh_scatter_df_mode['filter_cons'], marker=marker[modes], s=10, c=heating_bar_color, alpha=0.5)
        except (IndexError, KeyError):
            heating_color_map = {-1: 'grey', 0: sh_color[modes], 1: sh_color[modes], 2: sh_color[modes], 3: sh_color[modes]}

        y_label_text = "Consumption (Wh)"
        axis_flat[1].set_ylabel(y_label_text, fontsize=7)
        axis_flat[1].set_xlabel('hdd ', fontsize=7)
        axis_flat[1].tick_params(axis='both', which='major', labelsize=5)
        axis_flat[1].set_title('Heating : Mode 0')
        axis_flat[3].set_ylabel(y_label_text, fontsize=7)
        axis_flat[3].set_xlabel('hdd ', fontsize=7)
        axis_flat[3].tick_params(axis='both', which='major', labelsize=5)
        axis_flat[3].set_title('Heating : Mode 1')
        sh_cluster_info = heating_modes[modes]['cluster_info']
        plot_predictions_trend(sh_cluster_info, heating_modes, modes, axis_flat, marker, axis_identifier)

    figure.tight_layout()
    start, end = int(np.nanmin(hvac_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])), int(np.nanmax(hvac_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]))

    plot_dir = static_params.get('path', {}).get('hvac_plots', '')
    plot_dir = plot_dir + '/' + global_config['uuid']
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = plot_dir + "/{disagg_mode}_regression_clusters_{uuid}_{start}_{end}.png".format(disagg_mode=disagg_mode, start=start, end=end,
                                                                                                     uuid=global_config['uuid'])

    if os.path.isfile(image_location):
        os.remove(image_location)

    if plot_dir:
        plt.savefig(image_location, dpi=250)

    plt.close()
    del figure
