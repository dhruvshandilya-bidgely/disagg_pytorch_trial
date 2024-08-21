
"""
Author - Nisha Agarwal
Date - 10th Nov 2020
Dump results of itemization module
"""

# Import python packages

import os
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# import functions from within the project

from python3.itemization.aer.functions.plot_heatmaps import plot_appliance_heatmaps

from python3.itemization.init_itemization_config import init_itemization_params

matplotlib.style.use('ggplot')


def dump_output(item_input_object, item_output_object):

    """ Function for dumping required line plots and heatmaps for lighting module """

    uuid = item_input_object.get("config").get("uuid")
    pilot = item_input_object.get("config").get("pilot_id")

    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    activity_curve = item_input_object.get("weekday_activity_curve")
    sleep_hours = item_output_object.get('debug').get("profile_attributes_dict").get("sleep_hours")
    active_hours = item_output_object.get('debug').get("profile_attributes_dict").get("active_hours")
    active_levels = item_output_object.get('debug').get("profile_attributes_dict").get("activity_levels")
    input_data = item_output_object.get('debug').get("input_data_dict").get("day_input_data")
    raw_input_data = item_input_object.get("item_input_params").get("input_data")[6, :, :]
    lighting_output2 = item_output_object.get('debug').get("lighting_module_dict").get("lighting_day_estimate_after_postprocess")

    lighting_output1 = item_output_object.get('debug').get("lighting_v1_debug").get("data").get('lighting')

    itemization_config = init_itemization_params()

    sunrise_sunset_data = item_output_object.get('debug').get("lighting_module_dict").get("sunrise_sunset_data")
    date_ts_list = item_output_object.get('debug').get("input_data_dict").get("date_ts_list")
    disagg_mode = item_input_object.get('config').get('disagg_mode')
    t_start = item_input_object.get('gb_disagg_event').get('start')
    t_end = item_input_object.get('gb_disagg_event').get('end')
    capacity = item_output_object.get('debug').get("lighting_module_dict").get("lighting_capacity")
    clean_days = item_output_object.get('debug').get("lighting_module_dict").get("top_cleanest_days")
    perc = item_input_object.get("perc_value")

    date_list = [datetime.utcfromtimestamp(ts) for ts in date_ts_list]
    date_list = pd.DatetimeIndex(date_list).date

    dump_line_plots = [1, 0, 0]

    path_name = itemization_config.get('results_folder')

    path_name = path_name + "/" + uuid + "/"

    if not os.path.isdir(path_name):
        os.makedirs(path_name)

    if dump_line_plots[0]:

        plt.plot(activity_curve, label="final activity curve", alpha=0.5)

        for level in np.unique(active_levels):
            plt.plot(np.ones(len(activity_curve)) * np.round(level, 2), linestyle="dashed")

        plt.title("uuid : " + uuid + " | pilot :" + str(pilot))

        plt.xticks(np.arange(0, len(activity_curve), 1*samples_per_hour), labels=np.arange(0, 24))
        plt.yticks(np.arange(0, np.round(np.max(activity_curve), 2) + 0.1, 0.05))

        plt.savefig(path_name + uuid + "_activity_seq.png")
        plt.close()

    if dump_line_plots[1]:

        plt.plot(activity_curve, label="final activity curve", alpha=0.5)

        plt.scatter(np.where(active_hours == 1)[0], activity_curve[np.where(active_hours == 1)[0]], c='yellow',
                    label='Active', s=15)
        plt.legend()
        plt.scatter(np.where(active_hours == 0)[0], activity_curve[np.where(active_hours == 0)[0]], c='brown',
                    label='Non active', s=15)
        plt.legend()

        plt.title("uuid : " + uuid + " | pilot :" + str(pilot))

        plt.xticks(np.arange(0, len(activity_curve), 1 * samples_per_hour), labels=np.arange(0, 24))
        plt.yticks(np.arange(0, np.round(np.max(activity_curve), 2) + 0.1, 0.05))

        plt.savefig(path_name + uuid + "_active_hours.png")
        plt.close()

    if dump_line_plots[2]:

        plt.plot(activity_curve, label="final activity curve", alpha=0.5)

        plt.scatter(np.where(sleep_hours == 1)[0], activity_curve[np.where(sleep_hours == 1)[0]], c='yellow',
                    label='Non sleep hours', s=15)
        plt.legend()
        plt.scatter(np.where(sleep_hours == 0)[0], activity_curve[np.where(sleep_hours == 0)[0]], c='brown',
                    label='sleep hours', s=15)
        plt.legend()

        plt.title("uuid : " + uuid + " | pilot :" + str(pilot))

        plt.xticks(np.arange(0, len(activity_curve), 1 * samples_per_hour), labels=np.arange(0, 24))
        plt.yticks(np.arange(0, np.round(np.max(activity_curve), 2) + 0.1, 0.05))

        plt.savefig(path_name + uuid + "_sleep_hours.png")
        plt.close()

    list_of_df = [raw_input_data, input_data, sunrise_sunset_data, lighting_output1, lighting_output2]

    avg_capacity = lighting_output2.sum(axis=1)
    avg_capacity = avg_capacity[avg_capacity > 0]
    avg_capacity = np.mean(avg_capacity)

    old_avg_capacity = lighting_output1.sum(axis=1)
    old_avg_capacity = old_avg_capacity[old_avg_capacity > 0]
    old_avg_capacity = np.mean(old_avg_capacity)

    old_lighting_hours_count = lighting_output1 > 0
    old_lighting_hours_count = np.sum(old_lighting_hours_count, axis=1)
    old_lighting_hours_count = old_lighting_hours_count[old_lighting_hours_count > 0]
    old_lighting_hours_count = np.mean(old_lighting_hours_count)

    lighting_hours_count = lighting_output2 > 0
    lighting_hours_count = np.sum(lighting_hours_count, axis=1)
    lighting_hours_count = lighting_hours_count[lighting_hours_count > 0]
    lighting_hours_count = np.mean(lighting_hours_count) / samples_per_hour

    old_capacity = np.percentile(lighting_output1, 95) * samples_per_hour

    suptitle = "uuid : " + uuid + " | Pilot id : " + str(pilot) + " | Sampling rate : " + str(3600/samples_per_hour) + \
               " | Top cleanest days count : " + str(clean_days) + " | perc :" + str(perc) + " | Occupants count : " +\
               str(item_input_object.get('home_meta_data').get('numOccupants')) + \
               " | Rooms count : " + str(item_input_object.get('home_meta_data').get('totalRooms'))

    title_1 = "\n" + "Avg lighting hours : " + str(np.round(old_lighting_hours_count / samples_per_hour)) + "\n" + \
              "Hourly Capacity : " + str(np.round(old_capacity)) + " Wh" + "\n" + " Avg daily consumption : " +\
              str(np.round(old_avg_capacity)) + " Wh"

    title_2 = "\n" + "Avg lighting hours : " + str(np.round(lighting_hours_count)) + "\n" + \
              "Hourly Capacity : " + str(np.round(capacity)) + " Wh" + "\n" + "Avg daily consumption : " + \
              str(np.round(avg_capacity)) + " Wh"

    title_list = ['Raw Input data', "Preprocessed Input data", "Sunrise-Sunset", title_1, title_2]

    path_name = path_name + uuid + "_" + disagg_mode + "_" + str(t_start) + "_" + str(t_end)

    plot_appliance_heatmaps(date_list, list_of_df, title_list, suptitle, path_name, n_row=1, n_col=len(list_of_df))

    return
