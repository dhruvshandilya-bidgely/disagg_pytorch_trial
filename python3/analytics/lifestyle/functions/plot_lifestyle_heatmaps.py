"""
Author - Prasoon Patidar
Date - 13/07/20
plot lifestyle heatmaps for debugging and QA purposes
"""

# import python packages

import os
import copy
import logging
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff


def prepare_and_dump_plot(seasons, season_exists, axn, samples_per_day, fn_weekend_attrs, suptitle_meta_info,
                          suptitle_weekend_info, fig_lineplot, plot_dir, disagg_input_object):
    """Dump lineplots for weekend warrior"""

    for season_id in range(seasons.count.value):

        if season_exists[season_id]:

            weekday_hourly_mean = fn_weekend_attrs(season_id, 'weekday_hourly_mean')
            weekend_hourly_mean = fn_weekend_attrs(season_id, 'weekend_hourly_mean')

            if (weekday_hourly_mean is not None) & (weekday_hourly_mean.shape[0] > 0):
                axn[season_id].plot(weekday_hourly_mean, color='r', label='Weekday Energy')
            if (weekend_hourly_mean is not None) & (weekend_hourly_mean.shape[0] > 0):
                axn[season_id].plot(weekend_hourly_mean, color='b', label='Weekend Energy')
            axn[season_id].set_title('{} Energy'.format(seasons(season_id).name))
            axn[season_id].legend()
            axn[season_id].grid()
            axn[season_id].set_ylabel("Consumption(wh)")
            axn[season_id].set_xlabel('Hour of Day')
            axn[season_id].set_xticks(range(0, samples_per_day, int(samples_per_day / 24)))
            axn[season_id].set_xticklabels(np.arange(0, 24), rotation=90)

    fig_lineplot.suptitle('{}\n{}'.format(suptitle_meta_info, suptitle_weekend_info), fontsize=12)

    image_location = plot_dir + '/lineplot_lifestyle_' + disagg_input_object['config']['uuid'] + '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location, dpi=250)
    plt.close()
    del (fig_lineplot)

    return


def plot_lifestyle_heatmaps(lifestyle_input_object, lifestyle_output_object,
                            disagg_input_object, disagg_output_object, logger_pass):
    """
    Parameters:
        lifestyle_input_object (dict)           : Dictionary containing all lifestyle inputs
        lifestyle_output_object(dict)           : Dictionary containing all lifestyle outputs
        disagg_input_object (dict)              : Dictionary containing all disagg inputs
        disagg_output_object(dict)              : Dictionary containing all disagg outputs

    Returns:
        exit_status(bool)                       : status for plotting
        exit_code(str)                          : error string if plotting code breaks
    """

    t_lifestyle_plots_start = datetime.now()

    # Initialize the logger

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('plot_lifestyle_heatmaps')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # get relevant inputs from input and output objects

    global_config = disagg_input_object.get('config')

    raw_data_config = lifestyle_input_object.get('raw_data_config')

    weather_config = lifestyle_input_object.get('weather_config')

    debug_config = lifestyle_input_object.get('debug_config')

    input_df = pd.DataFrame(disagg_input_object['input_data'])

    columns = ['billcycle', 'week', 'day', 'day_of_week', 'hour_of_day', 'epoch', 'consumption',
               'temperature', 'sky_cover', 'wind', 'dew', 'sunrise', 'sunset', 'feels_like', 'prec', 'snow', 'sl_press',
               'spc_hum', 'rel_hum', 'wet_bulb', 'wind_dir', 'visibility', 'cooling_pot', 'heating_pot', 'wh_pot',
               'is_cold_event', 'is_hot_event', 's_label']

    input_df.columns = columns

    input_data_raw = input_df[['billcycle', 'epoch', 'consumption', 'day', 'day_of_week', 'hour_of_day', 'temperature']]

    input_data_raw['timestamp'] = pd.to_datetime(input_data_raw['epoch'], unit='s')

    # get required outputs for bill cycle, season and annual level

    annual_output = lifestyle_output_object.get('annual')

    seasons = weather_config.get('season')

    season_fractions = list(map(float, annual_output.get('season_fraction')))

    season_exists = [(sf > 0) for sf in season_fractions]

    # Get date and time based on user timezone from epoch

    timezone = disagg_input_object['home_meta_data']['timezone']

    try:
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC',
                                                                              ambiguous='infer').dt.tz_convert(timezone)
    except:
        input_data_raw['timestamp'] = input_data_raw.timestamp.dt.tz_localize('UTC',
                                                                              ambiguous='NaT').dt.tz_convert(timezone)

    input_data_raw['date'] = input_data_raw['timestamp'].dt.date
    input_data_raw['time'] = input_data_raw['timestamp'].dt.time
    input_data_raw['year'] = input_data_raw['timestamp'].dt.year
    input_data_raw['month'] = input_data_raw['timestamp'].dt.month

    # -----Get all heatmaps for debug plots------

    # trim outlier consumption

    max_quantile_val = raw_data_config.get('max_percentile_val') / 100
    fill_quantile_val = max_quantile_val + 0.01

    input_data_raw['consumption'][
        input_data_raw['consumption'] > input_data_raw['consumption'].quantile(max_quantile_val)] = \
        input_data_raw['consumption'].quantile(fill_quantile_val)
    input_data_raw['temperature'][
        input_data_raw['temperature'] > input_data_raw['temperature'].quantile(max_quantile_val)] = \
        input_data_raw['temperature'].quantile(fill_quantile_val)

    # Get non_pp consumption

    pp_epoch_estimate = np.zeros(input_data_raw.shape[0])

    PP_CONSUMPTION_COL = 1

    special_outputs_poolpump = disagg_output_object.get('special_outputs').get('pp_consumption', None)

    if special_outputs_poolpump is not None:
        # If poolpump is special outputs module, subtract poolpump from epoch output

        pp_epoch_estimate = special_outputs_poolpump[:, PP_CONSUMPTION_COL]

        pp_epoch_estimate[np.isnan(pp_epoch_estimate)] = 0.

    input_data_raw['non_pp_consumption'] = input_data_raw['consumption'] - pp_epoch_estimate

    input_data_raw.loc[input_data_raw['non_pp_consumption'] < 0, 'non_pp_consumption'] = 0.

    # get season value for at day level

    day_input_idx = lifestyle_input_object.get('day_input_data_index')

    day_season_vals = lifestyle_input_object.get('day_input_data_seasons')

    df_seasons = pd.DataFrame(day_season_vals, index=day_input_idx, columns=['season'])

    input_data_raw = pd.merge(input_data_raw, df_seasons, left_on='day', right_index=True)

    # Normalise consumption at seasonal level

    input_data_raw['normed_consumption'] = np.nan

    for season_id in range(seasons.count.value):
        # normalize seasonally for input data

        season_data_consumption = input_data_raw[input_data_raw.season == season_id].non_pp_consumption

        min_season_val, max_season_val = season_data_consumption.quantile(0.02), season_data_consumption.quantile(0.98)

        season_data_consumption = (season_data_consumption - min_season_val) / (max_season_val - min_season_val)

        season_data_consumption[season_data_consumption < 0] = 0.

        season_data_consumption[season_data_consumption > 1] = 1.

        input_data_raw.loc[input_data_raw.season == season_id, 'normed_consumption'] = season_data_consumption

    # Get Weekday and Weekend Information

    SATURDAY_DAY_ID = raw_data_config.get('SATURDAY_DAY_ID')
    SUNDAY_DAY_ID = raw_data_config.get('SUNDAY_DAY_ID')

    input_data_raw['is_weekday'] = ~((input_data_raw['day_of_week'] == SATURDAY_DAY_ID) |
                                     (input_data_raw['day_of_week'] == SUNDAY_DAY_ID))

    input_data_raw['weekday_normed_consumption'] = input_data_raw['normed_consumption']

    input_data_raw.loc[~input_data_raw['is_weekday'], 'weekday_normed_consumption'] = 0.

    input_data_raw['weekend_normed_consumption'] = input_data_raw['normed_consumption']

    input_data_raw.loc[input_data_raw['is_weekday'], 'weekend_normed_consumption'] = 0.

    # Mark Season demarcations in normalized consumption for weekdays and weekends

    season_change_index = np.where(~(input_data_raw['season'].diff() == 0))

    season_change_dates = input_data_raw.date.values[season_change_index]

    input_data_raw.loc[input_data_raw.date.isin(season_change_dates), 'weekday_normed_consumption'] = 0.75

    input_data_raw.loc[input_data_raw.date.isin(season_change_dates), 'weekend_normed_consumption'] = 0.75

    # Get office goer start/end times

    office_goer_config = lifestyle_input_object.get('office_goer_config')

    input_data_raw['office_goer_info'] = get_office_goer_info(input_data_raw, lifestyle_output_object,
                                                              office_goer_config)

    # Get Wakeup/Sleep times information

    input_data_raw['wakeup_sleep_info'] = get_wakeup_sleep_info(input_data_raw, lifestyle_output_object,
                                                                global_config.get('sampling_rate'))

    # Merge office goer and wakeup/sleep info

    input_data_raw['office_wakeup_sleep_info'] = input_data_raw['office_goer_info']

    input_data_raw.loc[input_data_raw['wakeup_sleep_info'] < 0, 'office_wakeup_sleep_info'] = input_data_raw[
        'wakeup_sleep_info']

    # Get final heatmaps for all information

    energy_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='consumption', aggfunc=sum)
    energy_heatmap = energy_heatmap.fillna(0)
    energy_heatmap = energy_heatmap.astype(int)

    non_pp_energy_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='non_pp_consumption',
                                                       aggfunc=sum)
    non_pp_energy_heatmap = non_pp_energy_heatmap.fillna(0)
    non_pp_energy_heatmap = non_pp_energy_heatmap.astype(int)

    temperature_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='temperature', aggfunc=sum)
    temperature_heatmap = temperature_heatmap.fillna(0)
    temperature_heatmap = temperature_heatmap.astype(int)

    office_goer_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='office_wakeup_sleep_info',
                                                     aggfunc=sum)
    office_goer_heatmap = office_goer_heatmap.fillna(0)
    office_goer_heatmap = office_goer_heatmap.astype(float)

    weekday_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='weekday_normed_consumption',
                                                 aggfunc=sum)
    weekday_heatmap = weekday_heatmap.fillna(0)
    weekday_heatmap = weekday_heatmap.astype(float)

    weekend_heatmap = input_data_raw.pivot_table(index='date', columns=['time'], values='weekend_normed_consumption',
                                                 aggfunc=sum)
    weekend_heatmap = weekend_heatmap.fillna(0)
    weekend_heatmap = weekend_heatmap.astype(float)

    # -----Get probabilities for header information------

    # Get info for office goer, active user and weekend warrior

    office_goer_prob = round(float(annual_output.get('office_goer_prob', np.nan)), 2)

    fn_office_seasonal_prob = \
        lambda season_name: round(annual_output.get('office_goer_seasonal_prob').get(season_name), 2) if \
            season_exists[seasons[season_name].value] else 'N.A'

    active_user_prob = round(float(annual_output.get('active_user_prob', np.nan)), 2)

    fn_active_seasonal_prob = \
        lambda season_name: round(annual_output.get('active_user_seasonal_prob').get(season_name), 2) if \
            season_exists[seasons[season_name].value] else 'N.A'

    weekend_warrior_prob = round(float(annual_output.get('weekend_warrior_prob', np.nan)), 2)

    fn_weekend_seasonal_prob = \
        lambda season_name: round(annual_output.get('weekend_warrior_seasonal_prob').get(season_name), 2) if \
            season_exists[seasons[season_name].value] else 'N.A'

    # plot final combined heatmaps with header

    fig_heatmap, axn = plt.subplots(1, 6, sharey=True)
    fig_heatmap.set_size_inches(20, 10)

    suptitle_meta_info = 'Meta Info:  uuid: {}  |  Pilot id: {}  |  Sampling rate : {}'.format(
        global_config.get("uuid"),
        global_config.get("pilot_id"),
        global_config.get("sampling_rate"))

    suptitle_office_info = 'Office Goer Prob: {} | winter: {} | summer: {} | transition: {}'.format(office_goer_prob,
                                                                                                    fn_office_seasonal_prob(
                                                                                                        'winter'),
                                                                                                    fn_office_seasonal_prob(
                                                                                                        'summer'),
                                                                                                    fn_office_seasonal_prob(
                                                                                                        'transition'))

    suptitle_active_info = 'Active User Prob: {} | winter: {} | summer: {} | transition: {}'.format(active_user_prob,
                                                                                                    fn_active_seasonal_prob(
                                                                                                        'winter'),
                                                                                                    fn_active_seasonal_prob(
                                                                                                        'summer'),
                                                                                                    fn_active_seasonal_prob(
                                                                                                        'transition'))

    suptitle_weekend_info = 'Weekend Warrior Prob: {} | winter: {} | summer: {} | transition: {}'.format(
        weekend_warrior_prob,
        fn_weekend_seasonal_prob('winter'),
        fn_weekend_seasonal_prob('summer'),
        fn_weekend_seasonal_prob('transition'))

    fig_heatmap.suptitle('{}\n{}\n{}\n{}'.format(suptitle_meta_info, suptitle_office_info, suptitle_active_info,
                                                 suptitle_weekend_info), fontsize=12)

    # Get Max/Min limits for heatmaps

    e_max = np.max(energy_heatmap.max())
    t_max = np.max(temperature_heatmap.max())
    t_min = np.min(input_data_raw['temperature'])

    t_max = max(t_max, 110)

    if t_max != 110:
        t_max = min(t_max, 120)

    t_min = max(t_min, 10)

    # Add heatmaps to plot

    sns.heatmap(energy_heatmap, ax=axn.flat[0], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(non_pp_energy_heatmap, ax=axn.flat[1], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    sns.heatmap(temperature_heatmap, ax=axn.flat[2], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=t_min, vmax=t_max)
    sns.heatmap(office_goer_heatmap, ax=axn.flat[3], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=-1, vmax=1)
    sns.heatmap(weekday_heatmap, ax=axn.flat[4], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=0, vmax=1)
    sns.heatmap(weekend_heatmap, ax=axn.flat[5], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=0, vmax=1)

    axn.flat[0].set_title("Raw Energy")
    axn.flat[0].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[1].set_title("Raw Energy(-PP)")
    axn.flat[1].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[2].set_title("Temperature (F)")
    axn.flat[2].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[3].set_title("Office Goer + Wakeup/Sleep")
    axn.flat[3].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[4].set_title("Weekday Energy(Normed)")
    axn.flat[4].tick_params(axis='x', which='major', labelsize=7)
    axn.flat[5].set_title("Weekend Energy(Normed)")
    axn.flat[5].tick_params(axis='x', which='major', labelsize=7)

    for axis_idx in range(len(axn.flat)):
        x_axis = axn.flat[axis_idx].get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)

        y_axis = axn.flat[axis_idx].get_yaxis()
        y_label = y_axis.get_label()
        y_label.set_visible(False)

    plt.yticks(rotation=0)

    disagg_mode = global_config.get('disagg_mode')

    plot_dir = debug_config.get('plot_dir')

    plot_dir = plot_dir + '/' + str(disagg_mode)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = plot_dir + '/heatmap_lifestyle_' + disagg_input_object['config']['uuid'] + '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location, dpi=250)
    plt.close()
    del (fig_heatmap)

    # -----Dump lineplots for weekend warrior-----

    fig_lineplot, axn = plt.subplots(1, 3)
    fig_lineplot.set_size_inches(15, 6)

    season_output = lifestyle_output_object.get('season')

    fn_weekend_attrs = \
        lambda season_id, attribute: \
            np.round(season_output.get(seasons(season_id).name).get('weekend_warrior_debug', {}).get(attribute,
                                                                                                     np.array([])),
                     2) if \
                season_exists[season_id] else None

    samples_per_day = int(Cgbdisagg.SEC_IN_DAY / global_config.get('sampling_rate'))

    prepare_and_dump_plot(seasons, season_exists, axn, samples_per_day, fn_weekend_attrs, suptitle_meta_info,
                          suptitle_weekend_info, fig_lineplot, plot_dir, disagg_input_object)

    # Plot yearly load type plots for user

    fig_yearplot, axn = plt.subplots(1, 1)
    fig_yearplot.set_size_inches(22, 10)

    # Get input from annual data

    yearly_load_type_enum = lifestyle_input_object.get('yearly_load_type')

    yearly_load_debug = annual_output.get('yearly_load_debug')

    year_data_normed = yearly_load_debug.get('year_data_normed')

    year_data_idx = yearly_load_debug.get('year_data_idx')

    cluster_distances = yearly_load_debug.get('cluster_distances')

    # plot lineplot for yearly normalized load

    axn.plot(year_data_normed, color='b', label='Seasonal Load Curve')
    axn.set_ylim(0, 1)
    axn.legend()
    axn.grid()
    axn.set_ylabel("Normalized Load", fontsize=15)
    axn.set_xlabel('Day of Year', fontsize=15)

    # Set xticks
    tick_idx = list(range(0, year_data_normed.shape[0], max(int(year_data_normed.shape[0] / 30), 1)))
    tick_vals = year_data_idx[tick_idx]
    axn.set_xticks(tick_idx)
    axn.set_xticklabels(tick_vals, rotation=90)

    # create title for yearly plots

    suptitle_yearly_load_info = "Annual Load Type: {} | Consumption Min: {} Wh, Max: {} Wh | ".format(
        yearly_load_debug.get('yearly_load_type'), str(round(yearly_load_debug.get('min_cons_val', 2))),
        str(round(yearly_load_debug.get('max_cons_val', 2))))

    yearly_load_type_enum = lifestyle_input_object.get('yearly_profile_kmeans_model').get('cluster_labels')
    centers = lifestyle_input_object.get('yearly_profile_kmeans_model').get('cluster_centers')

    suptitle_cluster_dists = ["{} : {}".format(yearly_load_type_enum[i], round(cluster_distances[i], 2))
                              for i in range(len(yearly_load_type_enum))]

    suptitle_cluster_dist_info = "Cluster Distances:" + ' | '.join(suptitle_cluster_dists)

    axn.set_title('{}\n{}\n{}'.format(suptitle_meta_info, suptitle_yearly_load_info, suptitle_cluster_dist_info),
                  fontsize=12)

    image_location = plot_dir + '/yearplot_lifestyle_' + disagg_input_object['config']['uuid'] + '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location, dpi=250)
    plt.close()

    del (fig_yearplot)

    t_lifestyle_plots_end = datetime.now()

    logger.info("%s Plotted lifestyle debug plots  in | %.3f s", log_prefix('Generic'),
                get_time_diff(t_lifestyle_plots_start, t_lifestyle_plots_end))

    plt.figure(figsize=(14, 9 ))

    for i in range(len(yearly_load_type_enum)):
        plt.plot(centers[i], label=yearly_load_type_enum[i])
        plt.legend()

    plt.savefig( plot_dir + '/yearplot_centers_' + disagg_input_object['config']['uuid'] + '.png')
    plt.close()

    return None


def get_office_goer_info(input_data_raw, lifestyle_output_object, office_goer_config):
    """
    Parameters:
        input_data_raw(np.ndarray)              : Raw input data used for plotting
        lifestyle_output_object(dict)           : Dictionary containing all lifestyle outputs
        office_goer_config(dict)                : static config for office goers
    Returns:
        office_times_info(np.ndarray)           : 1-d array for shape raw_input_data.shape[0] identifying office times
    """

    # initilaize office time DF and get relevant inputs

    is_office_time_arr = np.zeros(input_data_raw.shape[0])

    # check if user is office goer or not. Only show office times if user is office goer

    is_office_goer = lifestyle_output_object.get('annual').get('is_office_goer', False)

    if not is_office_goer:
        return is_office_time_arr

    df_office_times = None

    season_output = lifestyle_output_object.get('season')

    office_goer_clusters = office_goer_config.get('OFFICE_CLUSTERS_PRIMARY')

    # Loop over all seasons to populate office times

    for season_name in season_output.keys():

        office_goer_debug_info = season_output.get(season_name).get('office_goer_debug')

        if office_goer_debug_info is None:
            continue

        for cluster in office_goer_clusters:

            if cluster.value in office_goer_debug_info.keys():

                cluster_day_idx = office_goer_debug_info.get(cluster.value).get('cluster_day_idx')

                office_time_mask = office_goer_debug_info.get(cluster.value).get('office_time_mask')

                df_cluster_office_times = pd.DataFrame(office_time_mask, index=cluster_day_idx)

                if df_office_times is None:

                    df_office_times = copy.deepcopy(df_cluster_office_times)

                else:

                    df_office_times = pd.concat([df_office_times, df_cluster_office_times])

    # Melt Dataframe based on day_idx

    df_office_time_raw = pd.melt(df_office_times.reset_index(), id_vars='index', var_name='hour_of_day',
                                 value_name='office_times')

    df_office_time_raw.index = df_office_time_raw.index.astype(int)

    df_office_time_raw.hour_of_day = df_office_time_raw.hour_of_day.astype(int)

    df_office_time_raw = pd.merge(input_data_raw, df_office_time_raw, left_on=['day', 'hour_of_day'],
                                  right_on=['index', 'hour_of_day'], how='left')

    df_office_time_raw.office_times = df_office_time_raw.office_times.fillna(False)

    is_office_time_arr = df_office_time_raw.office_times.values

    return is_office_time_arr


def get_wakeup_sleep_info(input_data_raw, lifestyle_output_object, sampling_rate):
    """
    Parameters:
        input_data_raw(np.ndarray)              : Raw input data used for plotting
        lifestyle_output_object(dict)           : Dictionary containing all lifestyle outputs
        sampling_rate(int)                      : Sampling rate for this user
    Returns:
        wakeup_sleep_info(np.ndarray)           : 1-d array for shape raw_input_data.shape[0] identifying wakeup/sleep times
    """

    df_wakeup_sleep_time = pd.pivot_table(input_data_raw[['billcycle', 'time', 'hour_of_day']].drop_duplicates(),
                                          index='billcycle', columns='time', values='hour_of_day')

    df_wakeup_sleep_time.loc[:, :] = 0.

    wakeup_sleep_columns = list(df_wakeup_sleep_time.columns)

    billcycle_output = lifestyle_output_object.get('billcycle')

    samples_per_hour = Cgbdisagg.SEC_IN_HOUR / sampling_rate

    # Get wakeup/sleep times for all bill cycles written

    for billcycle_start in billcycle_output.keys():

        wakeup = billcycle_output.get(billcycle_start).get('wakeup')

        if wakeup is not None:
            wakeup_time = wakeup.get('wakeup_time')

            wakeup_col_id = np.floor(wakeup_time * samples_per_hour)

            df_wakeup_sleep_time.loc[billcycle_start, wakeup_sleep_columns[int(wakeup_col_id)]] = -1

        sleep = billcycle_output.get(billcycle_start).get('sleep')

        if sleep is not None:
            sleep_time = sleep.get('sleep_time')

            sleep_col_id = np.ceil(sleep_time * samples_per_hour)

            df_wakeup_sleep_time.loc[billcycle_start, wakeup_sleep_columns[int(sleep_col_id)]] = -1

    # Melt Wakeup/Sleep DF into get column for raw input data

    df_wakeup_sleep_raw = pd.melt(df_wakeup_sleep_time.reset_index(), id_vars='billcycle', var_name='time',
                                  value_name='wakeup_sleep_times')

    df_wakeup_sleep_raw = pd.merge(input_data_raw, df_wakeup_sleep_raw, on=['billcycle', 'time'], how='left')

    df_wakeup_sleep_raw.wakeup_sleep_times = df_wakeup_sleep_raw.wakeup_sleep_times.fillna(0)

    is_wakeup_sleep_time_arr = df_wakeup_sleep_raw.wakeup_sleep_times.values

    return is_wakeup_sleep_time_arr
