"""
Author - Anand Kumar Singh
Date - 12th March 2021
Function to plot multiple  heatmaps
"""

# Import python packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.analytics.hvac_inefficiency.plotting_functions.side_by_side_heatmap import plot_columns_of_heatmap


def plot_cycling_plots(input_inefficiency_object, output_inefficiency_object, device='ac', destination_prefix=None):

    """
        Function ot plot cyclingd

        Parameters:
            input_inefficiency_object   (dict): Dictionary containing inputs
            output_inefficiency_object  (dict): Dictionary containing outputs
            device                      (str): HVAC id
            destination_prefix          (str): Location

        Returns:
            None
        """

    uuid = input_inefficiency_object.get('uuid')
    pilot_id = input_inefficiency_object.get('pilot_id')
    sampling_rate = input_inefficiency_object.get('sampling_rate')
    color_map = input_inefficiency_object.get('color_map')

    input_data = input_inefficiency_object.get('raw_input_data')
    cycling_debug_dictionary = output_inefficiency_object.get(device).get('cycling_debug_dictionary', dict({}))

    # Day list
    timezone = input_inefficiency_object.get('meta_data').get('timezone')

    # Prepare short cycling array
    short_cycling_array = cycling_debug_dictionary.get('short_cycling')

    # Prepare compressor consumption array
    compressor_consumption = cycling_debug_dictionary.get('compressor')

    # Prepare duty cycle 1 array
    duty_cycle_mode_1_array = cycling_debug_dictionary.get('duty_cycle_mode_1_array')

    # Prepare net consumption array
    net_consumption = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Prepare net consumption array
    temperature = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # epoch values
    epoch_values = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    # Prepare largest cluster consumption array
    unrolled_data_largest_cluster = cycling_debug_dictionary.get('unrolled_data_largest_cluster')

    # Prepare duty cycle largest array
    duty_cycle_array_largest = cycling_debug_dictionary.get('duty_cycle_array_largest')

    cycling_dataframe = np.c_[epoch_values, net_consumption, temperature, compressor_consumption,
                              duty_cycle_mode_1_array, unrolled_data_largest_cluster, duty_cycle_array_largest,
                              short_cycling_array]
    cycling_dataframe = pd.DataFrame(cycling_dataframe,
                                     columns=['epoch', 'Net', 'Temperature', 'Compressor', 'DC Mode 1',
                                              'Top Mode Cons', 'DC Top Mode', 'Short Cycling'])

    cycling_dataframe['datetime'] = pd.to_datetime(cycling_dataframe['epoch'],
                                                   unit='s').dt.tz_localize('UTC').dt.tz_convert(timezone)
    cycling_dataframe['date'] = cycling_dataframe['datetime'].dt.date
    cycling_dataframe['time'] = cycling_dataframe['datetime'].dt.time

    # Keeping saturation and pre  sat information
    saturation_temperature = cycling_debug_dictionary.get('saturation_temp')
    pre_saturation_temperature = cycling_debug_dictionary.get('pre_saturation_temperature')

    # Creating Pivot tables
    net_consumption = cycling_dataframe.pivot_table(index='date', columns='time', values='Net', aggfunc=np.nansum)

    temperature = cycling_dataframe.pivot_table(index='date', columns='time', values='Temperature', aggfunc=np.nansum)

    compressor_consumption = cycling_dataframe.pivot_table(index='date', columns='time', values='Compressor',
                                                           aggfunc=np.nansum)

    duty_cycle_mode_1_array = cycling_dataframe.pivot_table(index='date', columns='time', values='DC Mode 1',
                                                            aggfunc=np.nansum)

    unrolled_data_largest_cluster = cycling_dataframe.pivot_table(index='date', columns='time', values='Top Mode Cons',
                                                                  aggfunc=np.nansum)

    duty_cycle_array_largest = cycling_dataframe.pivot_table(index='date', columns='time', values='DC Top Mode',
                                                             aggfunc=np.nansum)

    short_cycling_array = cycling_dataframe.pivot_table(index='date', columns='time', values='Short Cycling',
                                                        aggfunc=np.nansum)

    list_of_dataframes = [net_consumption, temperature, compressor_consumption, duty_cycle_mode_1_array,
                          unrolled_data_largest_cluster, duty_cycle_array_largest, short_cycling_array]

    cmaps = ['hot', 'jet', 'hot', 'hot', 'hot', 'hot', 'RdBu']
    titles = ['Net', 'Temperature', 'Compressor', 'DC Mode 1', 'Top Mode Cons', 'DC Top Mode', 'Short Cycling']
    mins = [0, None, 0, 0, 0,  0, 0]

    cluster_count = cycling_debug_dictionary.get('all_cluster_information').get('cluster_count')

    main_title =\
        """uuid: {} | sampling_rate: {} | pilot: {}| cluster count: {} | sat: {} | pre-sat: {}""".format(uuid,
                                                                                                         sampling_rate,
                                                                                                         pilot_id,
                                                                                                         cluster_count,
                                                                                                         saturation_temperature,
                                                                                                         pre_saturation_temperature)

    destination = '{}_short_cycling.png'.format(destination_prefix)
    plot_columns_of_heatmap(list_of_dataframes, cmaps=cmaps, titles=titles, sup_titles=main_title,
                            save_path=destination, mins=mins)

    # Memory Management: Free stray memory
    del input_data
    del temperature
    del net_consumption
    del cycling_dataframe
    del list_of_dataframes
    del short_cycling_array
    del compressor_consumption
    del duty_cycle_mode_1_array
    del duty_cycle_array_largest
    del unrolled_data_largest_cluster

    # Plotting HVAC Duty Cycle Relationship
    dataframe = cycling_debug_dictionary.get('duty_cycle_relationship')

    title_string = """Relationship between duty cycle and Temeprature
                    uuid: {} | pilot_id: {}| sat: {} | pre_sat: {}""".format(uuid, pilot_id, saturation_temperature,
                                                                             pre_saturation_temperature)

    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe[:, 0], dataframe[:, 1])
    plt.xlabel('Temperature (F)')
    plt.ylabel('Duty Cycle')
    plt.title(title_string)
    plt.savefig('{}_duty_cycle_relationship.png'.format(destination_prefix))

    # Plotting Clustering
    hvac_valid_consumption = cycling_debug_dictionary.get('valid_hvac')
    predicted_clusters = cycling_debug_dictionary.get('predicted_clusters')
    fcc = cycling_debug_dictionary.get('full_cycle_consumption')

    invalid_idx = (hvac_valid_consumption < 0) | (np.isnan(hvac_valid_consumption))
    temp_array = hvac_valid_consumption[~invalid_idx]

    upper_value_hvac_consumption = super_percentile(temp_array, 95)
    temp_array[temp_array > upper_value_hvac_consumption] = upper_value_hvac_consumption

    initial_clustering_information = cycling_debug_dictionary.get('initial_clustering_information')
    original_predicted_cluster = initial_clustering_information.get('clustering_out')
    hvac_consumption_original = initial_clustering_information.get('hvac_consumption')
    unique_ids, counts = np.unique(original_predicted_cluster, return_counts=True)
    bins = np.histogram(temp_array, bins=50)[1]

    centers = []
    plt.figure(figsize=(8, 8))
    for id_ in unique_ids:
        plt.hist(hvac_consumption_original[original_predicted_cluster == id_], bins=bins,
                 color=color_map.get(id_, 'slategrey'))
        centers.append(np.median(hvac_consumption_original[original_predicted_cluster == id_]))
    centers = np.array(centers)
    centers = np.round(centers, 2)

    title_string = """HVAC Clustering | uuid: {} | pilot_id: {} |
                        FCC: {:.3f} | means = {}""".format(uuid, pilot_id, fcc, centers)
    plt.axvline(x=fcc)
    plt.xlabel('HVAC consumption in Wh')
    plt.title(title_string)
    plt.savefig('{}_initial_clustering.png'.format(destination_prefix))

    plt.figure(figsize=(8, 8))
    predicted_clusters = predicted_clusters[~invalid_idx]
    unique_ids, counts = np.unique(predicted_clusters, return_counts=True)
    bins = np.histogram(temp_array, bins=50)[1]

    centers = []
    for id_ in unique_ids:
        plt.hist(temp_array[predicted_clusters == id_], bins=bins, color=color_map.get(id_, 'slategrey'))
        centers.append(np.median(temp_array[predicted_clusters == id_]))
    centers = np.array(centers)
    centers = np.round(centers, 2)

    title_string = """Corrected HVAC Clustering | uuid: {} | pilot_id: {} |
                    FCC: {:.3f} | means = {}""".format(uuid, pilot_id, fcc, centers)
    plt.axvline(x=fcc)
    plt.xlabel('HVAC consumption in Wh')
    plt.title(title_string)
    plt.savefig('{}_corrected_clustering.png'.format(destination_prefix))

    plt.cla()
    plt.clf()
    plt.close()


def plot_cycling_combined(input_inefficiency_object, output_inefficiency_object, destination_prefix=None):

    """
    Function ot plot cycling combined

    Parameters:
        input_inefficiency_object   (dict): Dictionary containing inputs
        output_inefficiency_object  (dict): Dictionary containing outputs
        destination_prefix          (str): Location

    Returns:
        None
    """

    uuid = input_inefficiency_object.get('uuid')
    pilot_id = input_inefficiency_object.get('pilot_id')
    sampling_rate = input_inefficiency_object.get('sampling_rate')
    timezone = input_inefficiency_object.get('meta_data').get('timezone')

    input_data = input_inefficiency_object.get('raw_input_data')
    cycling_debug_dictionary_ac = output_inefficiency_object.get('ac').get('cycling_debug_dictionary', dict({}))
    cycling_debug_dictionary_sh = output_inefficiency_object.get('sh').get('cycling_debug_dictionary', dict({}))

    # Day list

    # Plotting Columns of Heatmap

    # Prepare short cycling array
    short_cycling_array_ac = cycling_debug_dictionary_ac.get('short_cycling', np.zeros_like(input_data[:, 1]))
    short_cycling_array_sh = cycling_debug_dictionary_sh.get('short_cycling', np.zeros_like(input_data[:, 1]))

    # Prepare compressor consumption array
    compressor_consumption_ac = cycling_debug_dictionary_ac.get('compressor', np.zeros_like(input_data[:, 1]))
    compressor_consumption_sh = cycling_debug_dictionary_sh.get('compressor', np.zeros_like(input_data[:, 1]))

    # Prepare net consumption array

    net_consumption = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Prepare net consumption array
    temperature = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # epoch values
    epoch_values = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    cycling_dataframe = np.c_[epoch_values, net_consumption, temperature, compressor_consumption_ac,
                              short_cycling_array_ac, compressor_consumption_sh, short_cycling_array_sh]
    cycling_dataframe = pd.DataFrame(cycling_dataframe,
                                     columns=['epoch', 'Net', 'Temperature', 'AC Compressor', 'AC Short Cycling',
                                              'SH Compressor', 'SH Short Cycling'])

    cycling_dataframe['datetime'] = pd.to_datetime(cycling_dataframe['epoch'],
                                                   unit='s').dt.tz_localize('UTC').dt.tz_convert(timezone)
    cycling_dataframe['date'] = cycling_dataframe['datetime'].dt.date
    cycling_dataframe['time'] = cycling_dataframe['datetime'].dt.time

    # Creating Pivot tables
    net_consumption = cycling_dataframe.pivot_table(index='date', columns='time', values='Net', aggfunc=np.nansum)

    temperature = cycling_dataframe.pivot_table(index='date', columns='time', values='Temperature', aggfunc=np.nansum)

    ac_compressor_consumption = cycling_dataframe.pivot_table(index='date', columns='time', values='AC Compressor',
                                                              aggfunc=np.nansum)
    sh_compressor_consumption = cycling_dataframe.pivot_table(index='date', columns='time', values='SH Compressor',
                                                              aggfunc=np.nansum)

    ac_short_cycling_array = cycling_dataframe.pivot_table(index='date', columns='time', values='AC Short Cycling',
                                                           aggfunc=np.nansum)
    sh_short_cycling_array = cycling_dataframe.pivot_table(index='date', columns='time', values='SH Short Cycling',
                                                           aggfunc=np.nansum)

    list_of_dataframes = [net_consumption, temperature, ac_compressor_consumption, ac_short_cycling_array,
                          sh_compressor_consumption, sh_short_cycling_array]

    mins = [0, None, 0, 0, 0, 0]

    cmaps = ['hot', 'jet', 'hot', 'hot', 'hot', 'hot']

    titles = ['Net', 'Temperature', 'AC Compressor', 'AC Short Cycling', 'SH Compressor', 'SH Short Cycling']

    # Keeping saturation and pre  sat information
    ac_saturation_temperature = cycling_debug_dictionary_ac.get('saturation_temp')
    ac_pre_saturation_temperature = cycling_debug_dictionary_ac.get('pre_saturation_temperature')

    # Keeping saturation and pre  sat information
    sh_saturation_temperature = cycling_debug_dictionary_sh.get('saturation_temp')
    sh_pre_saturation_temperature = cycling_debug_dictionary_sh.get('pre_saturation_temperature')

    main_title = """uuid: {} | pilot: {} | sampling_rate: {} | ac_sat: {} | ac_pre-sat: {}
                        sh_sat: {} | sh_pre_sat: {}""".format(uuid, pilot_id, sampling_rate, ac_saturation_temperature,
                                                              ac_pre_saturation_temperature, sh_saturation_temperature,
                                                              sh_pre_saturation_temperature)

    destination = '{}_short_cycling.png'.format(destination_prefix)
    plot_columns_of_heatmap(list_of_dataframes, cmaps=cmaps, titles=titles, sup_titles=main_title,
                            save_path=destination, mins=mins)

    # Memory Management: removing large variables
    del input_data
    del temperature
    del net_consumption
    del cycling_dataframe
    del list_of_dataframes
    del ac_short_cycling_array
    del sh_short_cycling_array
    del ac_compressor_consumption
    del sh_compressor_consumption

    plt.cla()
    plt.clf()
    plt.close()
