"""
Author - Nikhil Singh Chauhan
Date - 29/05/2020
This module saves the heatmap for EV data
"""

# Import python packages

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy import stats, unique
from datetime import datetime, date

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.file_utils import file_unzip, file_zip


def unix_to_date(unix_time, format="%b"):
    """
    Return calendar month of a timestamp
    Parameters:
        unix_time        (int)   : timestamp to be converted to date
        format           (str)   : format of required date string

    Returns:
        month            (str)    : String containing the calendar month
    """
    dt = datetime.fromtimestamp(unix_time)
    dv = datetime.timetuple(dt)
    month = date(int(dv[0]), int(dv[1]), int(dv[2])).strftime(format)

    return month


def make_dir(path):
    """
    Parameters:
        path        (str)   : Path to create directory for

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    return


# Day of week mapping

day_of_week = {0: 'Sunday',
               1: 'Monday',
               2: 'Tuesday',
               3: 'Wednesday',
               4: 'Thursday',
               5: 'Friday',
               6: 'Saturday'}


def ev_heatmaps(params, uuid, sampling_rate, pilot, user_dir, logger):
    """
    This functions generates heatmaps for EV debugging

    Parameters:
        params              (dict)       : Parameters dictionary
        uuid                (str)        : UUID
        sampling_rate       (int)        : Sampling Rate
        pilot               (int)        : Pilot id
        user_dir            (str)        : Directory for saving debug plots
        logger              (logger)       : Logger object to write logs

    Returns:
    """
    # Update current working directory to uuid folder

    current_working_dir = os.getcwd()
    original_pipeline_path = deepcopy(current_working_dir)

    logger.info('Current Working directory | {}'.format(current_working_dir))

    # noinspection PyBroadException
    try:
        # Check if the current working directory changes

        os.chdir(current_working_dir + '/' + user_dir)
    except:
        # If the current working directory didn't change, get the default one

        current_working_dir = '/'
        original_pipeline_path = deepcopy(current_working_dir)
        os.chdir(current_working_dir + '/' + user_dir)

    logger.info('The new current working directory | {}'.format(current_working_dir + '/' + user_dir))

    # Unzip the files (if present in the target location)

    file_unzip(logger)

    # Define the file names that are to be loaded for heatmap

    month_ts_csv = uuid + '_month_ts.csv'
    raw_csv = uuid + '_raw.csv'
    input_csv = uuid + '_input.csv'
    detection_csv = uuid + '_detection.csv'
    ev_csv = uuid + '_ev.csv'
    residual_csv = uuid + '_residual.csv'

    month_ts = pd.read_csv(month_ts_csv, header=None, index_col=None)

    raw_data = pd.read_csv(raw_csv, header=None, index_col=None)
    input_data = pd.read_csv(input_csv, header=None, index_col=None)
    detection_data = pd.read_csv(detection_csv, header=None, index_col=None)
    ev_data = pd.read_csv(ev_csv, header=None, index_col=None)
    residual = pd.read_csv(residual_csv, header=None, index_col=None)

    f5, ar1 = plt.subplots(1, 5, figsize=(30, 15), dpi=100, )
    axarr = ar1.flatten()

    plot_title = 'Energy Heatmaps for: ' + uuid
    plot_title += '\n' + 'Confidence: ' + str(params['confidence_list']) + ', Amplitude: ' + str(
        params['amplitude']) + ' W,' + ' Charger Type: ' + str(params['charger_type'])

    if 'post_processing_param' in params:
        plot_title += '\nFirst EV month: ' + str(
            params['post_processing_param']['first_ev_month']) + ', Last EV month: ' + str(
                params['post_processing_param']['last_ev_month'])
        plot_title += ', Charging freq: ' + str(
            params['post_processing_param']['charging_freq']) + ', Charges per day: ' + str(
                params['post_processing_param']['charges_per_day'])
        plot_title += ', frac_multi_charge: ' + str(params['post_processing_param']['frac_multi_charge'])
        plot_title += '\nmonth_count_var: ' + str(
            params['post_processing_param']['box_monthly_count_var']) + ', monthly_presence_var: ' + str(
                params['post_processing_param']['box_monthly_presence_var'])
        plot_title += ', seasonal_count_var: ' + str(params['post_processing_param']['box_seasonal_count_var']) + ', seasonal_box_frac: ' + str(
            params['post_processing_param']['seasonal_boxes_frac'])
        plot_title += ', prom_smr_hrs: ' + str(
            params['post_processing_param']['prom_smr_hrs']) + ', prom_wtr_hrs: ' + str(
                params['post_processing_param']['prom_wtr_hrs'])

    plot_title += '\nPilot: ' + str(pilot) + ', Region: ' + params['region'] + ', Sampling Rate: ' + str(
        sampling_rate) + ' sec'

    f5.suptitle(plot_title, fontsize=15)

    f5.subplots_adjust(wspace=0.32, hspace=0.1)

    # Adding raw data heatmap

    im0 = axarr[0].imshow(raw_data, cmap='jet', aspect='auto')

    frac_pd = Cgbdisagg.HRS_IN_DAY / raw_data.shape[1]
    xticks = np.arange(0, Cgbdisagg.HRS_IN_DAY / frac_pd, 4 / frac_pd)
    label_r = np.arange(0, Cgbdisagg.HRS_IN_DAY / frac_pd, 4 / frac_pd)

    month_daywise = stats.mode(month_ts, axis=1)[0]
    hour_tick = np.arange(0, Cgbdisagg.HRS_IN_DAY - frac_pd, frac_pd)
    frac_tick = (hour_tick - np.floor(hour_tick)) * 0.6
    day_points = np.floor(hour_tick) + frac_tick

    months, month_idx, accum_idx = unique(month_daywise, return_index=True, return_inverse=True)

    xtick_labels = list(map(int, day_points[label_r.astype(int)]))
    axarr[0].xaxis.set_ticks(xticks)
    axarr[0].set_xticklabels(xtick_labels)

    yticks = month_idx
    ytick_labels = []

    # For overlapping dates issue

    new_yticks = []
    new_months = []
    for i in range(len(yticks)):
        if i == 0 or (yticks[i] - yticks[i-1]) >= 10:
            new_yticks.append(yticks[i])
            new_months.append(months[i])

    months = new_months
    yticks = new_yticks

    for j in range(0, len(months)):
        t1 = unix_to_date(months[j], '%d-%b-%y')
        ytick_labels.append(t1)

    axarr[0].yaxis.set_ticks(yticks)
    axarr[0].set_yticklabels(ytick_labels)

    axarr[0].set_title('Raw Data')

    f5.colorbar(im0, ax=axarr[0])

    # Adding input data heatmap

    im1 = axarr[1].imshow(input_data, cmap='jet', aspect='auto')

    axarr[1].xaxis.set_ticks(xticks)
    axarr[1].set_xticklabels(xtick_labels)

    axarr[1].yaxis.set_ticks(yticks)
    axarr[1].set_yticklabels(ytick_labels)

    axarr[1].set_title('Input Data')

    f5.colorbar(im1, ax=axarr[1])

    # Adding detection boxes heatmap
    #
    im2 = axarr[2].imshow(detection_data, cmap='jet', aspect='auto')

    axarr[2].xaxis.set_ticks(xticks)
    axarr[2].set_xticklabels(xtick_labels)

    axarr[2].yaxis.set_ticks(yticks)
    axarr[2].set_yticklabels(ytick_labels)

    axarr[2].set_title('Detection boxes')

    f5.colorbar(im2, ax=axarr[2])

    # Adding ev output heatmap

    im3 = axarr[3].imshow(ev_data, cmap='jet', aspect='auto')

    axarr[3].xaxis.set_ticks(xticks)
    axarr[3].set_xticklabels(xtick_labels)

    axarr[3].yaxis.set_ticks(yticks)
    axarr[3].set_yticklabels(ytick_labels)

    axarr[3].set_title('EV output (' + str(params['amplitude']) + ' W)')

    f5.colorbar(im3, ax=axarr[3])

    # Adding residual heatmap

    im4 = axarr[4].imshow(residual, cmap='jet', aspect='auto')

    axarr[4].xaxis.set_ticks(xticks)
    axarr[4].set_xticklabels(xtick_labels)

    axarr[4].yaxis.set_ticks(yticks)
    axarr[4].set_yticklabels(ytick_labels)

    axarr[4].set_title('Residual Data')

    f5.colorbar(im4, ax=axarr[4])
    # Save the plot to the debug folder

    plt.rcParams["axes.grid"] = False
    plt.savefig(uuid + '_heatmap.png')
    plt.close()

    # Zipping files to a folder for memory reasons

    file_zip(logger)

    # Update the current working directory back to the pipeline

    os.chdir(original_pipeline_path)

    logger.info('The restored current working directory | {}'.format(original_pipeline_path))

    return
