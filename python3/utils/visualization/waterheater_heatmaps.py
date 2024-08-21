"""
Author - Nikhil Singh Chauhan
Date - 02/11/18
This module contains function to create multiple heat map for different configuration
"""

# Import python packages

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def wh_heatmaps(raw, wh, uuid, sr, save_dir, pilot=0, tag="", conf=0, num_runs=0):
    """
    Dumps heat map for 2 different data matrices

    Parameters:
        raw           (np.ndarray)      : Input data 21-column matrix
        wh              (np.ndarray)    : Water heater output 21-column matrix
        uuid            (str)           : User id
        sr              (np.ndarray)    : Sampling rate
        save_dir        (str)           : Path to save the plots
        tag             (str)           : Information / marker for the plot
        conf            (float)         : Confidence score
        pilot           (int)           : Pilot id of the user
        num_runs        (int)           : Number of runs (if timed appliance)
    """

    # Water heater of total

    wh_percent = np.sum(wh[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]) / np.sum(raw[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    wh_percent = np.round(100 * wh_percent, 3)

    # Convert numpy arrays to pandas data frame

    raw = pd.DataFrame(raw)
    wh = pd.DataFrame(wh)

    # Define the column names for the data frame

    cols = ['Month', 'Week', 'Day', 'DOW', 'HOD', 'Hour', 'Energy', 'Temp', 'sky_cover', 'wind', 'dew', 'sunrise',
            'sunset', 'feels_like', 'precipitation', 'snow', 'pressure', 'sp_humid', 'rel_hum', 'wet_bulb', 'wind_dir',
            'visibility', 'cooling_pot', 'heating_pot', 'wh_pot', 'is_cold_event', 'is_hot_event', 's_label']

    # Assign the data frame column names

    raw.columns = cols
    wh.columns = cols

    # Take time difference of the epoch (Hour) column

    diff = datetime.utcfromtimestamp(raw['Hour'][0]).timetuple().tm_hour - raw['HOD'][0]

    # Convert the 24-hour system to 12-hour system

    if abs(diff) > int(Cgbdisagg.HRS_IN_DAY // 2):
        if diff > 0:
            diff = diff - Cgbdisagg.HRS_IN_DAY
        else:
            diff = diff + Cgbdisagg.HRS_IN_DAY

    # Assign the timestamps to the relevant column

    raw['timestamp'] = pd.to_datetime(raw['Hour'] - diff * Cgbdisagg.SEC_IN_HOUR, unit='s')
    wh['timestamp'] = raw['timestamp']

    # Extract the date and time components and assign separately

    raw['date'] = raw['timestamp'].dt.date
    raw['time'] = raw['timestamp'].dt.time

    # Cap the raw data energy to 99.9 % for avoiding distorted plots due to outliers

    raw['Energy'][raw['Energy'] > raw['Energy'].quantile(0.999)] = raw['Energy'].quantile(0.999)
    raw['Energy'][raw['Energy'] < raw['Energy'].quantile(0.001)] = raw['Energy'].quantile(0.001)

    # Extract the date and time components and assign separately

    wh['date'] = wh['timestamp'].dt.date
    wh['time'] = wh['timestamp'].dt.time

    # Make pivot table for the raw data

    heat_1 = raw.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_1 = heat_1.fillna(0)
    heat_1 = heat_1.astype(int)

    # Make pivot table for the water heater data

    heat_2 = wh.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_2 = heat_2.fillna(0)
    heat_2 = heat_2.astype(int)

    # Declare the plots object with 2 subplots with the given size

    fig_heatmap, axn = plt.subplots(1, 2, sharey=True)
    fig_heatmap.set_size_inches(20, 14)

    # Add master title to the main plot

    fig_heatmap.suptitle('Energy Heatmaps for: ' + uuid + ', Num_runs: ' + str(num_runs) + '\n' +
                         tag + ', Confidence: ' + str(conf) + ', WH_of_Total: ' + str(wh_percent) +
                         ',\nPilot: ' + str(pilot) + ', Sampling Rate: ' + str(sr) + 's)', fontsize=15)

    # Find the maximum values for each pivot table

    max_1 = np.max(heat_1.max())
    max_2 = np.max(heat_2.max())

    # Make heat map for each pivot table

    sns.heatmap(heat_1, ax=axn.flat[0], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=max_1)
    sns.heatmap(heat_2, ax=axn.flat[1], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=max_2)

    # Set title for each subplot

    axn.flat[0].set_title('Input Data')
    axn.flat[1].set_title('Water Heater')

    # Align the ticks and their orientation

    axn.flat[0].tick_params(axis='y', labelrotation=0)
    axn.flat[1].tick_params(axis='y', labelrotation=0)
    plt.xticks(rotation=90)

    # Check if the target directory exists (if not, create one)

    plot_dir = save_dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Dump the plot and close the file

    plt.savefig(plot_dir + '/' + uuid + '_' + tag + '_heatmap.png')
    plt.close()

    return


def heatmaps_3(raw, wh, gt, uuid, sr, save_dir, tag="", conf=0, pilot=0):
    """
    Dumps heat map for 3 different data matrices

    Parameters:
        raw             (np.ndarray)    : Input data 21-column matrix
        wh              (np.ndarray)    : Water heater output 21-column matrix
        gt              (np.ndarray)    : Actual water heater consumption (if available)
        uuid            (str)           : User id
        sr              (np.ndarray)    : Sampling rate
        save_dir        (str)           : Path to save the plots
        tag             (str)           : Information / marker for the plot
        conf            (float)         : Confidence score
        pilot           (int)           : Pilot id of the user
    """

    # Convert numpy arrays to pandas data frame

    raw = pd.DataFrame(raw)
    wh = pd.DataFrame(wh)
    gt = pd.DataFrame(gt)

    # Define the column names for the data frame

    cols = ['Month', 'Week', 'Day', 'DOW', 'HOD', 'Hour', 'Energy', 'Temp', 'A', 'B', 'C', 'D', 'E']

    # Assign the data frame column names

    raw.columns = cols
    wh.columns = cols
    gt.columns = cols

    # Take time difference of the epoch (Hour) column

    diff = datetime.utcfromtimestamp(raw['Hour'][0]).timetuple().tm_hour - raw['HOD'][0]

    # Convert the 24-hour system to 12-hour system

    if abs(diff) > int(Cgbdisagg.HRS_IN_DAY // 2):
        if diff > 0:
            diff = diff - Cgbdisagg.HRS_IN_DAY
        else:
            diff = diff + Cgbdisagg.HRS_IN_DAY

    # Assign the timestamps to the relevant column

    raw['timestamp'] = pd.to_datetime(raw['Hour'] - diff * Cgbdisagg.SEC_IN_HOUR, unit='s')
    wh['timestamp'] = raw['timestamp']
    gt['timestamp'] = raw['timestamp']

    # Extract the date and time components and assign separately

    raw['date'] = raw['timestamp'].dt.date
    raw['time'] = raw['timestamp'].dt.time

    # Cap the raw data energy to 99.9 % for avoiding distorted plots due to outliers

    raw['Energy'][raw['Energy'] > raw['Energy'].quantile(0.999)] = raw['Energy'].quantile(0.999)
    raw['Energy'][raw['Energy'] < raw['Energy'].quantile(0.001)] = raw['Energy'].quantile(0.001)

    # Extract the date and time components and assign separately

    wh['date'] = wh['timestamp'].dt.date
    wh['time'] = wh['timestamp'].dt.time

    gt['date'] = gt['timestamp'].dt.date
    gt['time'] = gt['timestamp'].dt.time

    # Make pivot table for the raw data

    heat_1 = raw.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_1 = heat_1.fillna(0)
    heat_1 = heat_1.astype(int)

    # Make pivot table for the water heater data

    heat_2 = wh.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_2 = heat_2.fillna(0)
    heat_2 = heat_2.astype(int)

    # Make pivot table for the actual water heater data

    heat_3 = gt.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_3 = heat_3.fillna(0)
    heat_3 = heat_3.astype(int)

    # Declare the plots object with 3 subplots with the given size

    fig_heatmap, axn = plt.subplots(1, 3, sharey=True)
    fig_heatmap.set_size_inches(20, 14)

    # Add master title to the main plot

    fig_heatmap.suptitle('Energy Heatmaps for: ' + uuid + '\n' +
                         tag + ', Confidence: ' + str(conf) +  ',\nPilot: ' + str(pilot) + ', Sampling Rate: ' +
                         str(sr) + 's', fontsize=15)

    # Find the maximum values for each pivot table

    max_1 = np.max(heat_1.max())
    max_2 = np.max(heat_2.max())
    max_3 = np.max(heat_3.max())

    maxi = np.max([max_2, max_3])

    # Make heat map for each pivot table

    sns.heatmap(heat_1, ax=axn.flat[0], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=max_1)
    sns.heatmap(heat_2, ax=axn.flat[1], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=maxi)
    sns.heatmap(heat_3, ax=axn.flat[2], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=maxi)

    # Set title for each subplot

    axn.flat[0].set_title('Raw Data')
    axn.flat[1].set_title('WH Output')
    axn.flat[2].set_title('GT Output')

    # Align the ticks and their orientation

    axn.flat[0].tick_params(axis='y', labelrotation=0)
    axn.flat[1].tick_params(axis='y', labelrotation=0)
    axn.flat[2].tick_params(axis='y', labelrotation=0)
    plt.xticks(rotation=90)

    # Check if the target directory exists (if not, create one)

    plot_dir = save_dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Dump the plot and close the file

    plt.savefig(plot_dir + '/' + uuid + '_' + tag + '_heatmap.png')
    plt.close()

    return


def heatmaps_4(raw, wh, gt, df, uuid, sr, save_dir, params, tag="", conf=0):
    """
    Dumps heat map for 4 different data matrices

    Parameters:
        raw             (np.ndarray)    : Input data 21-column matrix
        wh              (np.ndarray)    : Water heater output 21-column matrix
        gt              (np.ndarray)    : Actual water heater consumption (if available)
        df              (np.ndarray)    : Any consumption as per requirement
        uuid            (str)           : User id
        sr              (np.ndarray)    : Sampling rate
        save_dir        (str)           : Path to save the plots
        params          (np.ndarray)    : Parameters required in the plots
        tag             (str)           : Information / marker for the plot
        conf            (float)         : Confidence score
    """

    # Convert numpy arrays to pandas data frame

    raw = pd.DataFrame(raw)
    wh = pd.DataFrame(wh)
    gt = pd.DataFrame(gt)
    df = pd.DataFrame(df)

    # Define the column names for the data frame

    cols = ['Month', 'Week', 'Day', 'DOW', 'HOD', 'Hour', 'Energy', 'Temp', 'A', 'B', 'C', 'D', 'E']

    # Assign the data frame column names

    raw.columns = cols
    wh.columns = cols
    gt.columns = cols
    df.columns = cols

    # Take time difference of the epoch (Hour) column

    diff = datetime.utcfromtimestamp(raw['Hour'][0]).timetuple().tm_hour - raw['HOD'][0]

    # Convert the 24-hour system to 12-hour system

    if abs(diff) > int(Cgbdisagg.HRS_IN_DAY // 2):
        if diff > 0:
            diff = diff - Cgbdisagg.HRS_IN_DAY
        else:
            diff = diff + Cgbdisagg.HRS_IN_DAY

    # Assign the timestamps to the relevant column

    raw['timestamp'] = pd.to_datetime(raw['Hour'] - diff * Cgbdisagg.SEC_IN_HOUR, unit='s')
    wh['timestamp'] = raw['timestamp']
    gt['timestamp'] = raw['timestamp']
    df['timestamp'] = raw['timestamp']

    # Extract the date and time components and assign separately

    raw['date'] = raw['timestamp'].dt.date
    raw['time'] = raw['timestamp'].dt.time

    # Cap the raw data energy to 99.9 % for avoiding distorted plots due to outliers

    raw['Energy'][raw['Energy'] > raw['Energy'].quantile(0.999)] = raw['Energy'].quantile(0.999)
    raw['Energy'][raw['Energy'] < raw['Energy'].quantile(0.001)] = raw['Energy'].quantile(0.001)

    # Extract the date and time components and assign separately

    wh['date'] = wh['timestamp'].dt.date
    wh['time'] = wh['timestamp'].dt.time

    gt['date'] = gt['timestamp'].dt.date
    gt['time'] = gt['timestamp'].dt.time

    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    # Make pivot table for the raw data

    heat_1 = raw.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_1 = heat_1.fillna(0)
    heat_1 = heat_1.astype(int)

    # Make pivot table for the water heater data

    heat_2 = wh.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_2 = heat_2.fillna(0)
    heat_2 = heat_2.astype(int)

    # Make pivot table for the actual water heater data

    heat_3 = gt.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_3 = heat_3.fillna(0)
    heat_3 = heat_3.astype(int)

    # Make pivot table for the given data

    heat_4 = df.pivot_table(index='date', columns=['time'], values='Energy', aggfunc=sum)
    heat_4 = heat_4.fillna(0)
    heat_4 = heat_4.astype(int)

    # Declare the plots object with 4 subplots with the given size

    fig_heatmap, axn = plt.subplots(1, 4, sharey=True)
    fig_heatmap.set_size_inches(20, 14)

    # Add master title to the main plot

    fig_heatmap.suptitle('Energy Heatmaps for: ' + uuid + '\n' +
                         tag + ', Confidence: ' + str(conf) +  ',\nPilot: ' + str(params['pilot']) +
                         ', Sampling Rate: ' + str(sr) + 's),\n' + 'Mu: ' +
                         str(np.round(params['thin_peak_energy'], 2)) +
                         ', Fat Range: (' + str(np.round(params['lower_fat'], 2)) + ', ' +
                         str(np.round(params['upper_fat'], 2)) + ')\n' + 'Thin deficit %: ' +
                         str(params['thin_deficit_percent']) + ', Thin scale factor: ' +
                         str(params['thin_scale_factor']) + ', Fat scale factor: ' +
                         str(params['fat_scale_factor']) + '\n' + 'Error visible: ' + str(params['error_250']) + ' %',
                         fontsize=12)

    # Find the maximum values for each pivot table

    max_1 = np.max(heat_1.max())
    max_2 = np.max(heat_2.max())
    max_3 = np.max(heat_3.max())
    max_4 = np.max(heat_4.max())

    maxi = np.max([max_1, max_2, max_3, max_4])

    # Make heat map for each pivot table

    sns.heatmap(heat_1, ax=axn.flat[0], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=max_1)
    sns.heatmap(heat_2, ax=axn.flat[1], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=maxi)
    sns.heatmap(heat_3, ax=axn.flat[2], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=maxi)
    sns.heatmap(heat_4, ax=axn.flat[3], cmap='jet', cbar=True, xticklabels=8, yticklabels=30, vmin=0,
                vmax=maxi)

    # Set title for each subplot

    axn.flat[0].set_title('Raw Data')
    axn.flat[1].set_title('WH Output')
    axn.flat[2].set_title('GT Output')
    axn.flat[3].set_title('GT Output (above 250)')

    # Align the ticks and their orientation

    axn.flat[0].tick_params(axis='y', labelrotation=0)
    axn.flat[1].tick_params(axis='y', labelrotation=0)
    axn.flat[2].tick_params(axis='y', labelrotation=0)
    axn.flat[3].tick_params(axis='y', labelrotation=0)

    plt.xticks(rotation=90)

    # Check if the target directory exists (if not, create one)

    plot_dir = save_dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Dump the plot and close the file

    plt.savefig(plot_dir + '/' + uuid + '_' + tag + '_heatmap.png')
    plt.close()

    return


def data_histogram(data, uuid, sampling_rate, plot_dir, thin_peak_energy=0, lower_fat=0, upper_fat=0, error=0,
                   small=0, tag='Timed'):
    """
    Parameters:
        data                (np.ndarray)        : Input data 21-column matrix
        uuid                (str)               : Used id
        sampling_rate       (int)               : Sampling rate of the data
        plot_dir            (str)               : Path to store values
        thin_peak_energy    (float)             : Thin pulse energy per data point
        lower_fat           (float)             : Lower bound of fat pulse
        upper_fat           (float)             : Upper bound of fat pulse
        error               (float)             : Error percentage w.r.t GT
        small               (float)             : Consumption below 250 Wh
        tag                 (str)               : Marking timed/non-timed
    """

    energy = data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    bins = int(np.max(energy) // 50) + 1

    non_zero_energy = energy[energy > 0]

    plt.hist(non_zero_energy, bins=bins)

    plt.title(tag + ' data distribution,\n' + uuid + '\n' + ', Sampling Rate: ' + str(sampling_rate) + 's)\n' +
              'Mu: ' + str(np.round(thin_peak_energy, 2)) + ', Fat Range: (' + str(np.round(lower_fat, 2)) + ', ' +
              str(np.round(upper_fat, 2)) + ')\n' + 'Error: ' + str(error) + ' %' + ', Below 250: ' +
              str(small) + ' %', fontsize=10)

    plt.savefig(plot_dir + '/' + uuid + '_hist.png')
    plt.close()

    return
