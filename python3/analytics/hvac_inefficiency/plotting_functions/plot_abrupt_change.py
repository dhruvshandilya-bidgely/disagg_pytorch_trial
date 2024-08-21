"""
Author - Anand Kumar Singh
Date - 12th March 2021
Function to plot abrupt change in HVAC
"""

# Import python packages

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile
from python3.analytics.hvac_inefficiency.plotting_functions.side_by_side_heatmap import plot_columns_of_heatmap


def plot_abrupt_hvac_plot_single_device(input_inefficiency_object, output_inefficiency_object, device, destination,
                                        key):
    """
    Function to plot abrupt hvac

    Parameters:
        input_inefficiency_object   (dict): Dictionary containing inputs
        output_inefficiency_object  (dict): Dictionary containing outputs
        device                      (str): HVAC identifier
        destination                 (str): location
        key                         (str) : Key

    Returns:
        None
    """
    uuid = input_inefficiency_object.get('uuid')
    pilot_id = input_inefficiency_object.get('pilot_id')
    timezone = input_inefficiency_object.get('meta_data').get('timezone')
    hvac_consumption_matrix = output_inefficiency_object.get(device).get(key).get('hvac_consumption_matrix')
    hvac_potential_matrix = output_inefficiency_object.get(device).get(key).get('hvac_potential_matrix')
    dates = output_inefficiency_object.get(device).get(key).get('row')
    time = output_inefficiency_object.get(device).get(key).get('columns')
    final_outlier_days = output_inefficiency_object.get(device).get(key).get('final_outlier_days')
    string_for_heading = output_inefficiency_object.get(device).get(key).get('string')
    outlier_idx = np.isin(dates, final_outlier_days)
    dates = pd.to_datetime(pd.Series(dates), unit='s', ).dt.tz_localize('UTC').dt.tz_convert(timezone)

    # Prepare net consumption array
    input_data = input_inefficiency_object.get('raw_input_data')
    epoch_values = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]
    net_consumption = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    temperature = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    cycling_dataframe = np.c_[epoch_values, net_consumption, temperature]
    cycling_dataframe = pd.DataFrame(cycling_dataframe,
                                     columns=['epoch', 'Net', 'Temperature'])

    cycling_dataframe['datetime'] = pd.to_datetime(cycling_dataframe['epoch'],
                                                   unit='s').dt.tz_localize('UTC').dt.tz_convert(timezone)
    cycling_dataframe['date'] = cycling_dataframe['datetime'].dt.date
    cycling_dataframe['time'] = cycling_dataframe['datetime'].dt.time

    # Creating Pivot tables
    net_consumption = cycling_dataframe.pivot_table(index='date', columns='time', values='Net', aggfunc=np.nansum)

    temperature = cycling_dataframe.pivot_table(index='date', columns='time', values='Temperature', aggfunc=np.nansum)

    hvac_dataframe = pd.DataFrame(hvac_consumption_matrix, index=dates.dt.date, columns=time)
    hvac_potential_dataframe = pd.DataFrame(hvac_potential_matrix, index=dates.dt.date, columns=time)

    outlier_dataframe = copy.deepcopy(hvac_dataframe)
    outlier_dataframe[~outlier_idx] = 0

    if device == 'ac':
        potential_cmap = 'RdBu'
    else:
        potential_cmap = 'RdBu_r'

    cmaps = ['hot', 'jet', potential_cmap, 'hot', 'hot']
    titles = ['Net', 'Temperature', 'HVAC Potential', 'HVAC', key]
    list_of_dataframes = [net_consumption, temperature, hvac_potential_dataframe, hvac_dataframe, outlier_dataframe]
    main_title = """uuid: {} | pilot: {} |{} \n {}""".format(uuid, pilot_id, key, string_for_heading)
    mins = [0, None, None, 0, 0, 0 ]

    plot_columns_of_heatmap(list_of_dataframes, cmaps=cmaps, titles=titles, sup_titles=main_title,
                            save_path=destination, mins=mins)


def plot_scatter_plot(outlier_function_return, column_number, ax):

    """
    Function to plot scatter

    Parameters:
        outlier_function_return (dict): Dictionary containing outliers and inliers
        column_number           (int): Column identifier
        ax                      (axis): plot axis

    Returns:
        ax                      (axis) : plot axis

    """

    inliers = outlier_function_return['inliers']
    ax.scatter(inliers[0][:, column_number], inliers[1], marker='o', color='b', alpha=0.5, label='Inliers')

    # Plotting High Outliers
    outliers = outlier_function_return['high_outliers']
    ax.scatter(outliers['quad'][0][:, column_number], outliers['quad'][1], marker='x', color='black', alpha=1,
               label='High Quad')

    ax.scatter(outliers['ransac'][0][:, column_number], outliers['ransac'][1], marker='o', color='black', alpha=0.5,
               label='High RANSAC')

    # Plotting Low outliers
    outliers = outlier_function_return['low_outliers']
    ax.scatter(outliers['quad'][0][:, column_number], outliers['quad'][1], marker='x', color='r', alpha=1,
               label='Low Quad')

    ax.scatter(outliers['ransac'][0][:, column_number], outliers['ransac'][1], marker='o', color='r', alpha=0.5,
               label='Low RANSAC')

    # Computing limits of axes

    all_x_val = np.r_[outliers['quad'][0][:, column_number], outliers['ransac'][0][:, column_number],
                      inliers[0][:, column_number]]

    x_low_lim = super_percentile(all_x_val, 5)
    x_high_lim = super_percentile(all_x_val, 95)

    x_range_offset = (x_high_lim - x_low_lim) * 0.15

    x_low_lim = x_low_lim - x_range_offset
    x_high_lim = x_high_lim + x_range_offset

    # Plotting trend line
    if (outlier_function_return.get('regressor') is not None) & (len(all_x_val) > 5) & (x_range_offset != 0):
        input_trend_line = np.arange(x_low_lim, x_high_lim, step = (x_high_lim - x_low_lim)/ 100)
        input_trend_line = input_trend_line.reshape(-1, 1)

        regressor = outlier_function_return.get('regressor')[0]
        output_trend_line = regressor.predict(input_trend_line)

        outlier_function_return['trendline'] = [input_trend_line, output_trend_line]

    ax.scatter(outlier_function_return['trendline'][0], outlier_function_return['trendline'][1], marker=',', color='b',
               alpha=0.2, s=15, label='Trendline')

    # ax.set_ylim(y_low_lim , y_high_lim)
    ax.set_ylim(0, None)
    ax.set_xlim(x_low_lim, x_high_lim)

    return ax


def plot_abrupt_hvac_plot_combined(input_inefficiency_object, output_inefficiency_object, destination, key):

    """
    Function ot plot abrupt hvac combined

    Parameters:
        input_inefficiency_object   (dict): Dictionary containing inputs
        output_inefficiency_object  (dict): Dictionary containing outputs
        destination                 (str): Plot destination
        key                         (str): key

    Returns:
        None
    """

    uuid = input_inefficiency_object.get('uuid')
    pilot_id = input_inefficiency_object.get('pilot_id')

    fig = plt.figure(figsize=(16,8))

    initial_device_list = ['ac', 'sh']

    device_list = []

    for device in initial_device_list:

        abrupt_dict = output_inefficiency_object.get(device).get(key, dict({}))
        if len(abrupt_dict) != 0:
            device_list.append(device)

    device_row_map = {'ac': 0, 'sh': 1}
    shape_of_grid = (2,3)

    for device in device_list:
        row_idx = device_row_map.get(device)

        ax = plt.subplot2grid(shape_of_grid, (row_idx, 0), colspan=1, rowspan=1)
        outlier_function_return = output_inefficiency_object.get(device).get(key).get('return_dictionary')
        column_number = 0
        if len(outlier_function_return) != 0:
            ax = plot_scatter_plot(outlier_function_return, column_number, ax)
        ax.set_title('RANSAC: Potential and {}'.format(device))


        ax = plt.subplot2grid(shape_of_grid, (row_idx, 1), colspan=1, rowspan=1)
        outlier_function_return = output_inefficiency_object.get(device).get(key).get('return_dictionary_saturation')

        column_number = 1
        if len(outlier_function_return) != 0:
            ax = plot_scatter_plot(outlier_function_return, column_number, ax)
        ax.set_title('{}: Saturated Potential'.format(device))

        ax = plt.subplot2grid(shape_of_grid, (row_idx, 2), colspan=1, rowspan=1)
        outlier_function_return = output_inefficiency_object.get(device).get(key).get('return_dictionary_zero_saturation')

        column_number = 1
        if len(outlier_function_return) != 0:
            ax = plot_scatter_plot(outlier_function_return, column_number, ax)
        ax.set_title('{}: No Potential'.format(device))

        del outlier_function_return

    title_string = """uuid: {} | pilot: {} |{}
                        Outliers in black and red, trendline in light blue""".format(uuid, pilot_id, key)

    fig.suptitle(title_string)
    plt.savefig(destination)

    plt.clf()
    plt.cla()
    plt.close()


def plot_abrupt_hvac_plot_single_device_local(input_inefficiency_object, output_inefficiency_object, device,
                                              destination, key):

    """
        Function ot plot abrupt hvac single

        Parameters:
            input_inefficiency_object   (dict): Dictionary containing inputs
            output_inefficiency_object  (dict): Dictionary containing outputs
            destination                 (str): Plot destination
            key                         (str): key

        Returns:
            None
        """
    # Meta information

    uuid = input_inefficiency_object.get('uuid')
    pilot_id = input_inefficiency_object.get('pilot_id')
    timezone = input_inefficiency_object.get('meta_data').get('timezone')

    # Data matrix
    hvac_consumption_matrix = output_inefficiency_object.get(device).get(key).get('hvac_consumption_matrix')
    hvac_potential_matrix = output_inefficiency_object.get(device).get(key).get('hvac_potential_matrix')
    dates = output_inefficiency_object.get(device).get(key).get('row')
    time = output_inefficiency_object.get(device).get(key).get('columns')

    # Outlier days
    final_outlier_days = output_inefficiency_object.get(device).get(key).get('final_outlier_days')

    string_for_heading = output_inefficiency_object.get(device).get(key).get('string')

    outlier_idx = np.isin(dates, final_outlier_days)

    dates = pd.to_datetime(pd.Series(dates), unit='s', ).dt.tz_localize('UTC').dt.tz_convert(timezone)

    # Prepare net consumption array
    input_data = input_inefficiency_object.get('raw_input_data')
    epoch_values = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]
    net_consumption = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    temperature = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    cycling_dataframe = np.c_[epoch_values, net_consumption, temperature]
    cycling_dataframe = pd.DataFrame(cycling_dataframe,
                                     columns=['epoch', 'Net', 'Temperature'])

    cycling_dataframe['datetime'] = pd.to_datetime(cycling_dataframe['epoch'],
                                                   unit='s').dt.tz_localize('UTC').dt.tz_convert(timezone)
    cycling_dataframe['date'] = cycling_dataframe['datetime'].dt.date
    cycling_dataframe['time'] = cycling_dataframe['datetime'].dt.time

    cycling_dataframe['Net'] = cycling_dataframe['Net'].fillna(0)
    cycling_dataframe['Temperature'] = cycling_dataframe['Temperature'].fillna(0)

    # Creating Pivot tables
    net_consumption = cycling_dataframe.pivot_table(index='date', columns='time', values='Net', aggfunc=np.nansum)

    temperature = cycling_dataframe.pivot_table(index='date', columns='time', values='Temperature', aggfunc=np.nansum)

    hvac_dataframe = pd.DataFrame(hvac_consumption_matrix, index=dates.dt.date, columns=time)
    hvac_potential_dataframe = pd.DataFrame(hvac_potential_matrix, index=dates.dt.date, columns=time)

    outlier_dataframe = copy.deepcopy(hvac_dataframe)
    outlier_dataframe[~outlier_idx] = 0

    # Filtering based on HVAC region

    hvac_date_idx = np.nansum(hvac_consumption_matrix, axis=1) > 0

    temperature = temperature[hvac_date_idx]
    hvac_dataframe = hvac_dataframe[hvac_date_idx]

    net_consumption = net_consumption[hvac_date_idx]
    hvac_potential_dataframe = hvac_potential_dataframe[hvac_date_idx]

    outlier_dataframe = outlier_dataframe[hvac_date_idx]

    if device == 'ac':
        potential_cmap = 'RdBu'
    else:
        potential_cmap = 'RdBu_r'

    cmaps = ['hot', 'jet', potential_cmap, 'hot', 'hot']
    titles = ['Net', 'Temperature', 'HVAC Potential', 'HVAC', key]
    list_of_dataframes = [net_consumption, temperature, hvac_potential_dataframe, hvac_dataframe, outlier_dataframe]
    main_title = """uuid: {} | pilot: {} |{} \n {}""".format(uuid, pilot_id, key, string_for_heading)
    mins = [0, None, None, 0, 0, 0]

    plot_columns_of_heatmap(list_of_dataframes, cmaps=cmaps, titles=titles, sup_titles=main_title,
                            save_path=destination, mins=mins)
