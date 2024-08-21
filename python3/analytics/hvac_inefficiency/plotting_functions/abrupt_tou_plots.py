"""
Author - Anand Kumar Singh
Date - 12th March 2021
Function to plot multiple  heatmaps
"""

# Import python packages

import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.analytics.hvac_inefficiency.plotting_functions.side_by_side_heatmap import plot_columns_of_heatmap


def plot_abrupt_change_in_tou(input_inefficiency_object, output_inefficiency_object,  destination_prefix=None):
    """
    Function to plot abrupt change in tou

    Parameters:
        input_inefficiency_object   (dict)  : Dictionary containing inputs
        output_inefficiency_object  (dict)  : Dictionary containing outputs
        destination_prefix          (str)   : destination

    Returns:
        None
    """
    uuid = input_inefficiency_object.get('uuid')
    pilot_id = input_inefficiency_object.get('pilot_id')
    timezone = input_inefficiency_object.get('meta_data').get('timezone')
    dates = output_inefficiency_object.get('abrupt_tou_change').get('row')
    time = output_inefficiency_object.get('abrupt_tou_change').get('columns')
    final_outlier_days = output_inefficiency_object.get('abrupt_tou_change').get('final_outlier_days')
    total_consumption = output_inefficiency_object.get('abrupt_tou_change').get('total_consumption')
    hvac_outlier_hours_local = output_inefficiency_object.get('abrupt_tou_change').get('local_outliers')
    hvac_outlier_hours_global = output_inefficiency_object.get('abrupt_tou_change').get('global_outliers')
    net_consumption_outliers = output_inefficiency_object.get('abrupt_tou_change').get('consumption_outlier')
    cooling_consumption = output_inefficiency_object.get('abrupt_tou_change').get('ac')
    heating_consumption = output_inefficiency_object.get('abrupt_tou_change').get('sh')
    hvac_consumption = heating_consumption - cooling_consumption
    dates = pd.to_datetime(pd.Series(dates), unit='s', ).dt.tz_localize('UTC').dt.tz_convert(timezone)

    # Creating dataframes for plotting

    final_outlier_days = pd.DataFrame(final_outlier_days, index=dates.dt.date, columns=time)
    total_consumption = pd.DataFrame(total_consumption, index=dates.dt.date, columns=time)
    hvac_outlier_hours_local = pd.DataFrame(hvac_outlier_hours_local, index=dates.dt.date, columns=time)
    hvac_outlier_hours_global = pd.DataFrame(hvac_outlier_hours_global, index=dates.dt.date, columns=time)
    net_consumption_outliers = pd.DataFrame(net_consumption_outliers, index=dates.dt.date, columns=time)
    hvac_consumption = pd.DataFrame(hvac_consumption, index=dates.dt.date, columns=time)

    input_data = input_inefficiency_object.get('raw_input_data')

    # Prepare temperature array
    temperature = input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # epoch values
    epoch_values = input_data[:, Cgbdisagg.INPUT_EPOCH_IDX]

    temperature = np.c_[epoch_values, temperature]
    temperature = pd.DataFrame(temperature, columns=['epoch', 'temperature'])
    temperature['datetime'] = pd.to_datetime(temperature['epoch'],
                                             unit='s').dt.tz_localize('UTC').dt.tz_convert(timezone)
    temperature['date'] = temperature['datetime'].dt.date
    temperature['time'] = temperature['datetime'].dt.time
    temperature = temperature.pivot_table(index='date', columns='time', values='temperature', aggfunc=np.nansum)

    list_of_dataframes = [total_consumption, temperature, hvac_consumption, hvac_outlier_hours_local,
                          hvac_outlier_hours_global, net_consumption_outliers, final_outlier_days]
    cmaps = ['hot', 'jet', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'hot', 'RdBu_r']
    titles = ['Net', 'Temperature', 'HVAC', 'Local outliers', 'Global outliers', 'Net outlier', 'Final outlier']

    mins = [0, None, 0, 0, 0, 0, 0]

    main_title = """uuid: {} | pilot: {}| Abrupt TOU Change""".format(uuid, pilot_id)
    destination = '{}.png'.format(destination_prefix)
    plot_columns_of_heatmap(list_of_dataframes, cmaps=cmaps, titles=titles, sup_titles=main_title,
                            save_path=destination, mins=mins)

    # Memory Management: Removing large variables

    del input_data
    del temperature
    del epoch_values
    del hvac_consumption
    del total_consumption
    del final_outlier_days
    del list_of_dataframes
    del heating_consumption
    del cooling_consumption
    del hvac_outlier_hours_local
    del net_consumption_outliers
    del hvac_outlier_hours_global
