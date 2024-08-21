"""
Author - Anand Kumar Singh
Date - 12th March 2021
Function to plot multiple  heatmaps
"""

# Import python packages

import os
import pickle
import matplotlib.pyplot as plt

# Import functions from within the project

from python3.analytics.hvac_inefficiency.plotting_functions.divergence_scores import plot_app_change
from python3.analytics.hvac_inefficiency.plotting_functions.plot_cycling_based import plot_cycling_plots
from python3.analytics.hvac_inefficiency.plotting_functions.divergence_scores import plot_app_degradation
from python3.analytics.hvac_inefficiency.plotting_functions.divergence_scores import plot_behavior_change
from python3.analytics.hvac_inefficiency.plotting_functions.plot_cycling_based import plot_cycling_combined
from python3.analytics.hvac_inefficiency.plotting_functions.abrupt_tou_plots import plot_abrupt_change_in_tou
from python3.analytics.hvac_inefficiency.plotting_functions.plot_abrupt_change import plot_abrupt_hvac_plot_combined
from python3.analytics.hvac_inefficiency.plotting_functions.plot_abrupt_change import plot_abrupt_hvac_plot_single_device
from python3.analytics.hvac_inefficiency.plotting_functions.plot_abrupt_change import plot_abrupt_hvac_plot_single_device_local


def plot_cycling_plots_condition(cycling_dict, input_inefficiency_object, output_inefficiency_object,
                                 device, plot_prefix):

    """
    Function to plot cycling
    """

    if len(cycling_dict) != 0:
        plot_cycling_plots(input_inefficiency_object, output_inefficiency_object, device,
                           destination_prefix=plot_prefix)


def plot_relevant_plots(input_inefficiency_object, output_inefficiency_object):
    """
    Function to generate inefficiency plots

    Parameters:
        input_inefficiency_object   (dict): Dictionary containing inefficiency input objects
        output_inefficiency_object  (dict): Dictionary containing inefficiency output objects

    Returns:
        None
    """

    input_inefficiency_object['color_map'] = {
        0: 'red',
        1: 'blue',
        2: 'black',
        3: 'green',
        4 : 'cyan',
        5: 'yellow'
    }

    uuid = input_inefficiency_object.get('uuid')
    plot_directory = '../hvac_inefficiency_plot/{}'.format(uuid)

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    device_list = ['ac',  'sh']

    for device in device_list:

        plot_prefix = '{}/cycling_{}'.format(plot_directory, device)

        cycling_dict = output_inefficiency_object.get(device).get('cycling_debug_dictionary', dict({}))

        plot_cycling_plots_condition(cycling_dict, input_inefficiency_object, output_inefficiency_object, device,
                                     plot_prefix)

        # Plotting Abrupt change in AO HVAC

        plot_loc = '{}/abrupt_ao_{}.png'.format(plot_directory, device)
        key = 'abrupt_ao_hvac'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))

        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device(input_inefficiency_object, output_inefficiency_object,
                                                device, destination=plot_loc, key=key)

        plot_loc = '{}/abrupt_ao_local_{}.png'.format(plot_directory, device)
        key = 'abrupt_ao_hvac'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))

        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device_local(input_inefficiency_object, output_inefficiency_object,
                                                      device, destination=plot_loc, key=key)

        # Plotting Abrupt change in Amplitude

        plot_loc = '{}/abrupt_amp_{}.png'.format(plot_directory, device)
        key = 'abrupt_amplitude'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))

        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device(input_inefficiency_object, output_inefficiency_object, device,
                                                destination=plot_loc, key=key)

        plot_loc = '{}/abrupt_amp_local_{}.png'.format(plot_directory, device)
        key = 'abrupt_amplitude'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))

        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device_local(input_inefficiency_object, output_inefficiency_object, device,
                                                      destination=plot_loc, key=key)

        # Plotting Abrupt change in HVAC hours

        plot_loc = '{}/abrupt_hours_{}.png'.format(plot_directory, device)
        key = 'abrupt_hvac_hours'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))
        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device(input_inefficiency_object, output_inefficiency_object, device,
                                                destination=plot_loc, key=key)

        plot_loc = '{}/abrupt_hours_local_{}.png'.format(plot_directory, device)
        key = 'abrupt_hvac_hours'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))
        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device_local(input_inefficiency_object, output_inefficiency_object, device,
                                                      destination=plot_loc, key=key)

        plot_loc = '{}/abrupt_net_consumption_{}.png'.format(plot_directory, device)
        key = 'net_consumption_outlier'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))
        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device(input_inefficiency_object, output_inefficiency_object, device,
                                                destination=plot_loc, key=key)

        plot_loc = '{}/abrupt_net_ao_{}.png'.format(plot_directory, device)
        key = 'net_ao_outlier'
        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))
        if len(abrupt_dict) != 0:
            plot_abrupt_hvac_plot_single_device(input_inefficiency_object, output_inefficiency_object, device,
                                                destination=plot_loc, key=key)

        plot_loc = '{}/divergence_app_degradation_{}.png'.format(plot_directory, device)
        plot_app_degradation(input_inefficiency_object, output_inefficiency_object, device, plot_loc)

        plot_loc = '{}/divergence_app_change_{}.png'.format(plot_directory, device)
        plot_app_change(input_inefficiency_object, output_inefficiency_object, device, plot_loc)

        plot_loc = '{}/divergence_behavior_change_{}.png'.format(plot_directory, device)
        plot_behavior_change(input_inefficiency_object, output_inefficiency_object, device, plot_loc)

    # Combined Summary Plots
    plot_prefix = '{}/abrupt_tou'.format(plot_directory)
    plot_abrupt_change_in_tou(input_inefficiency_object, output_inefficiency_object, destination_prefix=plot_prefix)

    plot_prefix = '{}/cycling_combined'.format(plot_directory)
    plot_cycling_combined(input_inefficiency_object, output_inefficiency_object, destination_prefix=plot_prefix)

    plot_prefix = '{}/combined_abrupt_ao.png'.format(plot_directory)
    plot_abrupt_hvac_plot_combined(input_inefficiency_object, output_inefficiency_object, destination=plot_prefix,
                                   key='abrupt_ao_hvac')

    plot_prefix = '{}/combined_abrupt_amp.png'.format(plot_directory)
    plot_abrupt_hvac_plot_combined(input_inefficiency_object, output_inefficiency_object, destination=plot_prefix,
                                   key='abrupt_amplitude')

    plot_prefix = '{}/combined_abrupt_hours.png'.format(plot_directory)
    plot_abrupt_hvac_plot_combined(input_inefficiency_object, output_inefficiency_object, destination=plot_prefix,
                                   key='abrupt_hvac_hours')

    plot_prefix = '{}/combined_abrupt_net_consumption.png'.format(plot_directory)
    plot_abrupt_hvac_plot_combined(input_inefficiency_object, output_inefficiency_object, destination=plot_prefix,
                                   key='net_consumption_outlier')

    plot_prefix = '{}/combined_abrupt_net_ao.png'.format(plot_directory)
    plot_abrupt_hvac_plot_combined(input_inefficiency_object, output_inefficiency_object, destination=plot_prefix,
                                   key='net_ao_outlier')

    file_name = open("{}/{}_input.pkl".format(plot_directory, uuid), 'wb')
    pickle.dump(input_inefficiency_object, file_name)
    file_name.close()

    file_name = open("{}/{}_output.pkl".format(plot_directory, uuid), 'wb')
    pickle.dump(output_inefficiency_object, file_name)
    file_name.close()

    plt.cla()
    plt.clf()
    plt.close()
