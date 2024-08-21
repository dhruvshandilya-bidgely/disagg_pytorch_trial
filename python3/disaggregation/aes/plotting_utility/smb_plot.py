"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to generate smb plots
"""

# Import python packages

import os
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Import functions from within the project

from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def plot_diff_heatmap(energy_heatmap, diff_heatmap, disagg_input_object):
    """
    Function to generate month level consumption bar plots

    Parameters:

        energy_heatmap      (np.ndarray)       : Contains month level ao and hvac consumption
        diff_heatmap        (np.ndarray)       : Contains month level  diff consumptions
        disagg_input_object (dict)             : Contains user level input information

    Returns:

        None

    """

    static_params = hvac_static_params()

    smb_validity = disagg_input_object.get('switch').get('smb').get('validity')
    general_opening_time = disagg_input_object.get('switch').get('smb').get('opening_time')
    general_closing_time = disagg_input_object.get('switch').get('smb').get('closing_time')
    opening_time_conf = disagg_input_object.get('switch').get('smb').get('opening_time_conf')
    closing_time_conf = disagg_input_object.get('switch').get('smb').get('closing_time_conf')

    fig_heatmap, axn = plt.subplots(1, 2, sharey=False)
    fig_heatmap.set_size_inches(15, 10)

    fig_heatmap.suptitle('\n Diff-Maps   |    SMB Validity : {}    |    Opening Time : {} @ {}   |   '
                         'closing_time : {} @ {} '.format(smb_validity, general_opening_time, opening_time_conf,
                                                          general_closing_time, closing_time_conf), fontsize=10)

    sns.heatmap(energy_heatmap, ax=axn.flat[0], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)

    sns.heatmap(diff_heatmap, ax=axn.flat[1], cmap='jet', cbar=True, xticklabels=4, yticklabels=30)

    axn.flat[0].set_title("Raw Energy")
    axn.flat[0].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[1].set_title("Energy difference")
    axn.flat[1].tick_params(axis='x', which='major', labelsize=7)

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

    image_location = plot_dir + '/heatmap_difference_' + disagg_input_object['config']['uuid'] + '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location, dpi=250)
    plt.close()
    del fig_heatmap


def plot_start_end(smb_month_info, disagg_input_object, marker=''):
    """
    Function to generate month level consumption bar plots

    Parameters:

        smb_month_info      (dict)              : Contains month level smb key attributes info
        disagg_input_object (dict)              : Contains user level input information
        marker              (str)               : Plot identifier in argument

    Returns:

        None

    """

    static_params = hvac_static_params()

    energy_heatmap_list = np.array([])
    open_close_list = np.array([])
    k_means_list = np.array([])

    for month, info in smb_month_info.items():
        if month == 'unique_months':
            continue
        if open_close_list.shape[0] == 0:
            open_close_list = info.get('open_close_table')
            energy_heatmap_list = info['energy_heatmap']
            k_means_list = info.get('k_means_open_close')
        else:
            open_close_list = np.r_[open_close_list, info.get('open_close_table')]
            energy_heatmap_list = np.r_[energy_heatmap_list, info.get('energy_heatmap')]
            k_means_list = np.r_[k_means_list, info.get('k_means_open_close')]

    fig_heatmap, axn = plt.subplots(1, 3, sharey=False)
    fig_heatmap.set_size_inches(15, 10)

    fig_heatmap.suptitle('\n Diff-Maps   |    Open : {}    |    Close : {}'.format('O', 'C'), fontsize=10)

    sns.heatmap(energy_heatmap_list, ax=axn.flat[0], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)

    sns.heatmap(k_means_list, ax=axn.flat[1], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)

    sns.heatmap(open_close_list, ax=axn.flat[2], cmap='jet', cbar=True, xticklabels=4, yticklabels=30)

    if marker == 'actual':
        axn.flat[0].set_title("Raw - AO")
    else:
        axn.flat[0].set_title("Raw Energy")

    axn.flat[0].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[1].set_title("K Means")
    axn.flat[1].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[2].set_title("Energy difference")
    axn.flat[2].tick_params(axis='x', which='major', labelsize=7)

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

    plot_dir = plot_dir + '/' + disagg_input_object.get('config').get('uuid')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = plot_dir + '/heatmap_difference_' + marker + '_' + disagg_input_object['config']['uuid'] + '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location, dpi=250)
    plt.close()
    del fig_heatmap
