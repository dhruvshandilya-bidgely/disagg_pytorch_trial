"""
Author: Mayank Sharan
Created: 17-Jul-2020
Dump season plots for the user for testing and debugging purposes
"""

# Import python packages

import os
import pytz
import numpy as np

from datetime import datetime

# Import project functions and classes


# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dump_season_plots(nbi_input_data):

    """
    Dump season plots
    Parameters:
        nbi_input_data          (dict)          : Dictionary containing data needed for the engine run
    """

    # Initialize variables

    debug_dir = '../season_plots/'
    meta_data = nbi_input_data.get('meta_data', {})
    day_wise_data = nbi_input_data.get('weather').get('day_wise_data', {})
    season_detection_dict = nbi_input_data.get('weather').get('season_detection_dict', {})
    hvac_potential_dict = nbi_input_data.get('weather').get('hvac_potential_dict', {})

    if len(season_detection_dict) == 0 or len(hvac_potential_dict) == 0:
        return

    # Extract data to create the season plot

    s_label = season_detection_dict.get('s_label')
    max_tr_temp = season_detection_dict.get('max_tr_temp')
    max_winter_temp = season_detection_dict.get('max_winter_temp')
    class_name = season_detection_dict.get('class_name')

    cooling_pot = hvac_potential_dict.get('cooling_pot')
    heating_pot = hvac_potential_dict.get('heating_pot')
    heating_max_thr = hvac_potential_dict.get('heating_max_thr')
    cooling_min_thr = hvac_potential_dict.get('cooling_min_thr')

    day_temp = day_wise_data.get('temp')
    day_fl = day_wise_data.get('fl')
    day_ts = day_wise_data.get('day_ts')

    # Extract meta data

    uuid = meta_data.get('uuid')
    city = meta_data.get('city')

    if city is None:
        city = 'N/A'

    country = meta_data.get('country')

    if country is None:
        country = 'N/A'

    pilot_id = meta_data.get('pilot_id')
    current_ts = meta_data.get('current_ts')
    tz = pytz.timezone(meta_data.get('timezone', 'UTC'))

    num_days_data = len(s_label)

    # Compute values based on extracted variables

    daily_avg = np.nanmean(day_temp, axis=1)
    ticks = np.arange(num_days_data - (12 * (num_days_data // 12)) - 1, num_days_data, num_days_data // 12)

    # Initialize user plot dir

    user_plot_dir = debug_dir + uuid + '_' + str(current_ts) + '/'

    # Assign a label to each day in the ticks range

    tick_labels = []

    for day_idx in ticks:
        day_dt_time = datetime.fromtimestamp(day_ts[day_idx], tz)
        tick_labels.append(str(day_dt_time.day) + '-' + str(day_dt_time.month) + '-' + str(day_dt_time.year))

    tick_labels = np.array(tick_labels)

    # Create arrays to populate by season

    x_arr = np.arange(0, num_days_data)

    smr_x = x_arr[s_label == 1]
    trs_x = x_arr[s_label == 0.5]
    tr_x = x_arr[s_label == 0]
    trw_x = x_arr[s_label == -0.5]
    wtr_x = x_arr[s_label == -1]

    smr_temp = daily_avg[s_label == 1]
    trs_temp = daily_avg[s_label == 0.5]
    tr_temp = daily_avg[s_label == 0]
    trw_temp = daily_avg[s_label == -0.5]
    wtr_temp = daily_avg[s_label == -1]

    # Dump the season plot

    fig, ax = plt.subplots(figsize=(15, 10))
    ax2 = ax.twinx()

    ax.plot(daily_avg, label='daily_avg')

    ax.scatter(smr_x, smr_temp, c='r', label='Summer')
    ax.scatter(trs_x, trs_temp, c='y', label='Transition Summer')
    ax.scatter(tr_x, tr_temp, c='g', label='Transition')
    ax.scatter(trw_x, trw_temp, c='c', label='Transition Winter')
    ax.scatter(wtr_x, wtr_temp, c='b', label='Winter')

    if not (max_winter_temp == 'NA'):
        mwt_arr = np.full_like(x_arr, fill_value=max_winter_temp)
        ax.plot(x_arr, mwt_arr, 'b--', label='Max Winter Temp')

    if not (max_tr_temp == 'NA'):
        mtt_arr = np.full_like(x_arr, fill_value=max_tr_temp)
        ax.plot(x_arr, mtt_arr, 'g--', label='Max Transition Temp')

    plt.title(uuid + ' | City: ' + city + ' | Country: ' + country + ' | Pilot ID: ' + str(pilot_id) +
              '\nMax Wtr Temp: ' + str(max_winter_temp) + ' F | Max Tr Temp: ' + str(max_tr_temp) +
              ' F | Koppen Class: ' + class_name)

    t_c = lambda t_f: (t_f - 32) / 1.8

    # get left axis limits
    ymin, ymax = ax.get_ylim()

    # apply function and set transformed values to right axis limits
    ax2.set_ylim((t_c(ymin), t_c(ymax)))

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    ax.set_ylabel('Daily average temperature (F)')
    ax2.set_ylabel('Daily average temperature (C)')
    plt.xlabel('Date')
    ax.grid()
    ax2.grid()
    plt.grid()
    ax.legend()

    # Create directory to dump in if does not exist

    if not os.path.exists(user_plot_dir):
        os.makedirs(user_plot_dir)

    plt.savefig(user_plot_dir + uuid + '_season')
    plt.close()

    # Dump HVAC potential heatmap

    xticks = np.arange(0, 24, 4)

    fig, ar1 = plt.subplots(1, 4, figsize=(20, 10), dpi=80,)

    axarr = ar1.flatten()
    fig.subplots_adjust(wspace=0.7, hspace=0.1)

    colormap_str = 'jet'

    # Dump cooling potential

    im0 = axarr[0].imshow(cooling_pot, cmap=colormap_str, aspect='auto')

    axarr[0].xaxis.set_ticks(xticks)
    axarr[0].set_xticklabels(xticks)

    axarr[0].yaxis.set_ticks(ticks)
    axarr[0].set_yticklabels(tick_labels)

    axarr[0].set_title('Cooling Potential')

    fig.colorbar(im0, ax=axarr[0])

    # Dump Temperature

    im1 = axarr[1].imshow(day_temp, cmap=colormap_str, aspect='auto')

    axarr[1].xaxis.set_ticks(xticks)
    axarr[1].set_xticklabels(xticks)

    axarr[1].yaxis.set_ticks(ticks)
    axarr[1].set_yticklabels(tick_labels)

    axarr[1].set_title('Temperature (F)')

    fig.colorbar(im1, ax=axarr[1])

    # Dump Feels Like temperature

    im2 = axarr[2].imshow(day_fl, cmap=colormap_str, aspect='auto')

    axarr[2].xaxis.set_ticks(xticks)
    axarr[2].set_xticklabels(xticks)

    axarr[2].yaxis.set_ticks(ticks)
    axarr[2].set_yticklabels(tick_labels)

    axarr[2].set_title('Feels Like Temperature (F)')

    fig.colorbar(im2, ax=axarr[2])

    # Dump Heating potential

    im3 = axarr[3].imshow(heating_pot, cmap=colormap_str, aspect='auto')

    axarr[3].xaxis.set_ticks(xticks)
    axarr[3].set_xticklabels(xticks)

    axarr[3].yaxis.set_ticks(ticks)
    axarr[3].set_yticklabels(tick_labels)

    axarr[3].set_title('Heating Potential')

    fig.colorbar(im3, ax=axarr[3])

    fig.suptitle(uuid + ' | City: ' + city + ' | Country: ' + country + ' | Pilot ID: ' + str(pilot_id) +
                 ' | Koppen Class: ' + class_name + '\nMax Wtr Temp: ' + str(max_winter_temp) +
                 ' F | Max Heating Temp: ' + str(heating_max_thr) + ' F' + '\nMax Tr Temp: ' + str(max_tr_temp) +
                 ' F | Min Cooling Temp: ' + str(cooling_min_thr) + ' F')

    # Create directory to dump in if does not exist

    if not os.path.exists(user_plot_dir):
        os.makedirs(user_plot_dir)

    plt.savefig(user_plot_dir + uuid + '_pot_heatmap')
    plt.close()
