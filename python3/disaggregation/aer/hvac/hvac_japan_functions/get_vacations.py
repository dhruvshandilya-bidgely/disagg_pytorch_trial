"""
Author - Kris Duszak
Date - 3/25/2019
Module to return vacation related information
"""

# Import python packages

import numpy as np
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_vacation_info(hvac_input, vacation_periods, debug_detection, percent_vacation_daily_thresh=0.6):

    """
    Get the vacation days and indices which correspond to vacation days

    Parameters:
        hvac_input                      (np.ndarray)    : hvac features
        vacation_periods                (np.ndarray)    : vacation epoch periods
        debug_detection                 (dict)          : dict of debugging output
        percent_vacation_daily_thresh   (float)         : serves as a threshold for classifying a point as a vacation

    Returns:
        vacation_days_idx               (np.ndarray)    : index of vacation days
        vacation_days                   (np.ndarray)    : days which are predicted as vacation
        agg_vacation_month              (pd.DataFrame)  : aggregate of vacations at monthly level
    """

    agg_vacation_day = aggregate_vacation(hvac_input, vacation_periods, agg_type='daily')
    agg_vacation_month = aggregate_vacation(hvac_input, vacation_periods, agg_type='monthly')

    point_epoch = agg_vacation_day['epoch']
    vacation_days = point_epoch[agg_vacation_day['percent_vacation'] > percent_vacation_daily_thresh]
    vacation_days_idx = np.isin(hvac_input[:, Cgbdisagg.INPUT_DAY_IDX], vacation_days)

    debug_detection['vacation_days_idx'] = vacation_days_idx

    return vacation_days_idx, vacation_days, agg_vacation_month


def aggregate_vacation(hvac_input, vacation_periods, agg_type):

    """
    Function to aggregate predicted vacation points at a daily and monthly level

    Parameters:
        hvac_input                  (np.ndarray)        : Feature matrix for hvac_module
        vacation_periods            (np.ndarray)        : The predicted vacation epochs
        agg_type                    (str)               : The level at which to aggregate vacation points

    Returns:
        df_stats                    (pd.DataFrame)      : The aggregated vacation stats
    """

    point_vacation_idx = np.array([False] * len(hvac_input))
    epochs = hvac_input[:, Cgbdisagg.INPUT_EPOCH_IDX]
    if vacation_periods is None:
        return pd.DataFrame({'epoch': [0], 'agg_vac_points': [0], 'percent_vacation': [0]})
    else:
        n_rows = vacation_periods.shape[0]

    for i in range(n_rows):
        point_vacation_idx = (epochs >= vacation_periods[i, 0]) & (epochs <= vacation_periods[i, 1])

    # Aggregate the vacation periods by day

    if agg_type == 'daily':

        day_epoch = hvac_input[:, Cgbdisagg.INPUT_DAY_IDX]

        df = pd.DataFrame({'day_epoch': day_epoch, 'point_vacation_idx': point_vacation_idx})

        agg_vac_points = df.groupby('day_epoch')['point_vacation_idx'].sum()
        agg_num_points = df.groupby('day_epoch').size()

        percent_vacation = agg_vac_points / agg_num_points

        return pd.DataFrame({'epoch': agg_vac_points.index.values, 'agg_vac_points': agg_vac_points.values,
                             'percent_vacation': percent_vacation.values})

    # Aggregate the vacation periods by bill cycle

    elif agg_type == 'monthly':

        month_epoch = hvac_input[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]

        df = pd.DataFrame({'month_epoch': month_epoch, 'point_vacation_idx': point_vacation_idx})

        agg_vac_points = df.groupby('month_epoch')['point_vacation_idx'].sum()
        agg_num_points = df.groupby('month_epoch')['point_vacation_idx'].size()

        percent_vacation = agg_vac_points / agg_num_points

        return pd.DataFrame({'epoch': agg_vac_points.index.values, 'agg_vac_points': agg_vac_points.values,
                             'percent_vacation': percent_vacation.values})
