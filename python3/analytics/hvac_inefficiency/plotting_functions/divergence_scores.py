"""
Author - Anand Kumar Singh
Date - 26th June 2021
Function to plot multiple  heatmaps
"""

# Import python packages

import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def plot_app_change(input_inefficiency_object, output_inefficiency_object, device, plot_loc):

    """
    Function to plot appliance change

    Parameters:
        input_inefficiency_object   (dict): Dictionary containing inputs
        output_inefficiency_object  (dict): Dictionary containing outputs
        device                      (str): HVAC identifier
        plot_loc                    (str): location

    Returns:
        None
    """

    timestamp_current = input_inefficiency_object.get('raw_input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX]
    timestamp_previous_year = output_inefficiency_object.get(device, {}).get('app_change', {}).get('previous_year')

    previous_year_dc = output_inefficiency_object.get(device, {}).get('app_change', {}).get('previous_dc_relation')
    current_dc_relation = output_inefficiency_object.get(device, {}).get('app_change', {}).get('current_dc_relations')

    prediction = output_inefficiency_object.get(device, {}).get('app_change', {}).get('app_change')
    probability = output_inefficiency_object.get(device, {}).get('app_change', {}).get('probability')
    direction = output_inefficiency_object.get(device, {}).get('app_change', {}).get('change_direction')

    plt.figure(figsize=(16, 8))

    if previous_year_dc is not None:
        date = datetime.datetime.fromtimestamp(timestamp_previous_year)
        plt.plot(previous_year_dc[:, 0], previous_year_dc[:, 1], color='red', label=date)

    if current_dc_relation is not None:
        date = datetime.datetime.fromtimestamp(timestamp_current)
        plt.plot(current_dc_relation[:, 0], current_dc_relation[:, 1], color='blue', label=date)

    current_year_fcc = output_inefficiency_object.get(device, {}).get('app_change', {}).get('current_fcc')
    previous_year_fcc = output_inefficiency_object.get(device, {}).get('app_change', {}).get('previous_fcc')

    plt.title('{} App Change | prediction={} | probability={} | FCCs = {}, {} | direction = {}'.format(device.upper(),
                                                                                                       prediction,
                                                                                                       probability,
                                                                                                       previous_year_fcc,
                                                                                                       current_year_fcc,
                                                                                                       direction))
    plt.legend()
    plt.savefig(plot_loc)
    plt.close()


def plot_app_degradation(input_inefficiency_object, output_inefficiency_object, device, plot_loc):

    """
    Function to plot app degradation

    Parameters:
        input_inefficiency_object   (dict): Dictionary containing inputs
        output_inefficiency_object  (dict): Dictionary containing outputs
        device                      (str): HVAC identifier
        plot_loc                    (str): location

    Returns:
        None
    """

    timestamp_current = input_inefficiency_object.get('raw_input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX]
    timestamp_two_year_old = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('two_year_old')
    timestamp_previous_year = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('previous_year')

    two_year_old_dc = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('two_year_old_dc')
    previous_year_dc = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('previous_dc_relation')
    current_dc_relation =\
        output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('current_dc_relations')

    prediction = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('degradation')
    probability = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('probability')

    plt.figure(figsize=(16, 8))

    if two_year_old_dc is not None:
        date = datetime.datetime.fromtimestamp(timestamp_two_year_old)
        plt.plot(two_year_old_dc[:, 0], two_year_old_dc[:, 1], color='black', label=date)

    if previous_year_dc is not None:
        date = datetime.datetime.fromtimestamp(timestamp_previous_year)
        plt.plot(previous_year_dc[:, 0], previous_year_dc[:, 1], color='red', label=date)

    if current_dc_relation is not None:
        date = datetime.datetime.fromtimestamp(timestamp_current)
        plt.plot(current_dc_relation[:, 0], current_dc_relation[:, 1], color='blue', label=date)

    current_year_fcc = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('current_fcc')
    previous_year_fcc = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('previous_fcc')
    two_year_old_fcc = output_inefficiency_object.get(device, {}).get('app_degradation', {}).get('two_year_old_fcc')

    plt.title('{} App Degradation | prediction={} | probability={} | FCCs = {}, {}, {}'.format(device.upper(),
                                                                                               probability, prediction,
                                                                                               two_year_old_fcc,
                                                                                               previous_year_fcc,
                                                                                               current_year_fcc))
    plt.legend()
    plt.savefig(plot_loc)
    plt.close()


def plot_behavior_change(input_inefficiency_object, output_inefficiency_object, device, plot_loc):

    """
    Function to plot behavior change

    Parameters:
        input_inefficiency_object   (dict): Dictionary containing inputs
        output_inefficiency_object  (dict): Dictionary containing outputs
        device                      (str): HVAC identifier
        plot_loc                    (str): location

    Returns:
        None
    """

    timezone = input_inefficiency_object.get('meta_data').get('timezone')
    uuid = input_inefficiency_object.get('uuid')
    pilot_id = input_inefficiency_object.get('pilot_id')

    divergence_array =\
        output_inefficiency_object.get(device, dict({})).get('behavior_change', dict({})).get('divergence_array')

    high_change_dates =\
        output_inefficiency_object.get(device, dict({})).get('behavior_change', dict({})).get('change_date_high', [])

    low_change_dates = \
        output_inefficiency_object.get(device, dict({})).get('behavior_change', dict({})).get('change_date_low', [])

    fig, ax = plt.subplots(figsize=(16, 8))

    if divergence_array is not None:
        df = pd.DataFrame(divergence_array, columns=['date', 'divergence', 'support'])

        df['change'] = 0

        if low_change_dates is not None:
            df.loc[df['date'].isin(low_change_dates), 'change'] =\
                output_inefficiency_object.get(device, {}).get('behavior_change', {}).get('lower_threshold', -2.5)

        if high_change_dates is not None:
            df.loc[df['date'].isin(high_change_dates), 'change'] =\
                output_inefficiency_object.get(device, {}).get('behavior_change', {}).get('upper_threshold', 2.5)

        df['date'] = pd.to_datetime(df['date'], unit='s').dt.tz_localize('UTC').dt.tz_convert(timezone)
        df['date'] = df['date'].dt.date

        ax.plot(df['date'], df['divergence'], label='divergence', color='b', linestyle='solid', marker='o')

        ax.plot(df[df['divergence'].notnull()]['date'], df[df['divergence'].notnull()]['change'], label='change',
                color='black', linestyle='dashed', marker='^')

        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Divergence", fontsize=14)

        ax2 = ax.twinx()
        ax2.plot(df['date'], df['support'], label='length', color='r', linestyle='dashed', marker='x')

        ax2.set_ylabel("Support Length", fontsize=14)
        fig.legend()
        fig.autofmt_xdate()

    title_string = """uuid: {} | pilot: {}""".format(uuid, pilot_id)
    fig.suptitle(title_string)
    fig.savefig(plot_loc)
    plt.close()
