"""
Date Created - 13 Nov 2018
Author name - Pratap
Calculates various metrics of LAPs like distance etc.
"""
import logging
import numpy as np

from python3.config.Cgbdisagg import Cgbdisagg


def get_lap_distances(config, ref_detection, logger_base):
    """
    This function initiates all the variables to be updated by following functions

    Parameters:
        config (dict):
        ref_detection (dict):
        logger_base (logger):

    Returns:
        lap_distances:
        lowest_point:
        lowest_pt_cumulative_sum:
        perc_pts_lying_below_level:
        dist_from_transitions:
        num_transitions_above:
        num_transitions_below:

    """

    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_ref_estimation")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    laps = ref_detection['LAPs']
    input_data = ref_detection['input_data']
    lap_low_point = ref_detection['lapLowPoint']

    # Initializing the params as zeros
    lap_distances = np.ones((np.shape(laps)[0], 1))
    lowest_point = np.zeros((np.shape(laps)[0], 1))
    dist_from_transitions = np.zeros((np.shape(laps)[0], 1))
    num_transitions_above = np.zeros((np.shape(laps)[0], 1))
    num_transitions_below = np.zeros((np.shape(laps)[0], 1))
    perc_pts_lying_below_level = np.zeros((np.shape(laps)[0], 1))
    lowest_pt_cumulative_sum = np.zeros((np.shape(laps)[0], 1))

    # Calculating the metrics
    for i in range(np.shape(laps)[0]):
        lap_i = np.transpose((laps[i, :][:, np.newaxis]))
        if (np.shape(laps)[0] > 1) & (lap_i[0, 1] > lap_i[0, 0]):
            # noinspection PyBroadException
            try:
                lap_distances[i], lowest_point[i], lowest_pt_cumulative_sum[i], perc_pts_lying_below_level[i], \
                dist_from_transitions[i], num_transitions_above[i], num_transitions_below[i] = \
                    get_distances_for_lap(lap_i, input_data, lap_low_point, config)
            except Exception:
                logger.warning('NO metrics calculated |')

    return lap_distances, lowest_point, lowest_pt_cumulative_sum, perc_pts_lying_below_level, \
           dist_from_transitions, num_transitions_above, num_transitions_below


def get_distances_for_lap(lap_i, input_data, lap_low_point, config):
    """
    This function consolidates all the metrics for particular LAP
    """
    tstart = lap_i[0][0]
    tend = lap_i[0][1]
    input_data = input_data[(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= tstart) &
                            (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= tend), :]
    lap_low_point = lap_low_point[(lap_low_point[:, 1] >= tstart) & (lap_low_point[:, 1] <= tend) , :]

    # Get the lowest point metrics
    dist, lowest_point, lowest_pt_cumulative_sum, perc_pts_lying_below_level = get_dist_using_low_points(config, lap_low_point, input_data, lap_i)

    # Get the distance metrics
    dist_from_transitions, num_transitions_above, num_transitions_below,transitions = get_dist_using_transitions(config, input_data)

    return dist,lowest_point,lowest_pt_cumulative_sum,perc_pts_lying_below_level,dist_from_transitions,num_transitions_above,num_transitions_below


def get_dist_using_transitions(config, input_data):
    """
    This function calculates transitions, number of transitions above, below etc

    Parameters:
        config (dict):
        input_data (np.ndarray):

    Returns:
        dist_from_transitions:
        num_transitions_above:
        num_transitions_below:
        transitions:

    """
    dist_from_transitions = -1
    num_transitions_above = 1000
    num_transitions_below = -1
    transitions = 0

    if np.shape(input_data)[0] > 1:
        transitions = np.abs(np.diff(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]))
        if np.shape(transitions)[0] > 1:
            dist_from_transitions = np.nanmean(transitions[(transitions >=
                                                            config['LAPDetection']['MeanTransitionsBelowLimit']) &
                                                           (transitions <=
                                                            config['LAPDetection']['MeanTransitionsAboveLimit'])])
            num_transitions_above = np.nansum((transitions >= config['LAPDetection']['TransitionsAboveLimit']))
            num_transitions_below = np.nansum((transitions > config['LAPDetection']['TransitionsBelowLimit']) &
                                              (transitions < config['LAPDetection']['TransitionsAboveLimit']))

    return dist_from_transitions, num_transitions_above, num_transitions_below, transitions


def get_dist_using_low_points(config, lap_low_point, input_data, lap_i):
    """
    Calculates distance, lowestpointcumulativesum, percentagepointslying below level etc

    Parameters:
        config (dict):
        lap_low_point (np.ndarray):
        input_data (np.ndarray):
        lap_i (int)
    """
    # Initialize the default params
    dist = 1000
    lowest_point = 1000
    y_low_points = np.zeros((1, 1))
    lowest_pt_cumulative_sum = 1000
    perc_pts_lying_below_level = 0
    new_lowst_pnt = 0

    # Calculate the parameters if LAP has valid low points
    if (np.shape(input_data)[0] > 0) & (np.shape(lap_low_point[:, 2])[0] > 0) & (lap_i[0][1] > lap_i[0][0]):
        y_low_points = input_data[lap_low_point[:, 2] == 1, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        if np.shape(y_low_points)[0] > 1:
            lowest_point = np.nanmin(y_low_points)
            ylow_points_2 = y_low_points - lowest_point
            med_enr_low_pnts = np.nanmedian(ylow_points_2)
            std_enr_low_pnts = np.nanstd(ylow_points_2, ddof=1)
            min_enr_lmt = med_enr_low_pnts - (config['Estimation']['StdThrEst'] * std_enr_low_pnts)
            max_enr_lmt = med_enr_low_pnts + (config['Estimation']['StdThrEst'] * std_enr_low_pnts)

            if (min_enr_lmt < config['Estimation']['MinEnrThrLmt']):
                min_enr_lmt = 0
            fltrd_enr_pnt = ylow_points_2[(ylow_points_2 >= min_enr_lmt) & (ylow_points_2 <= max_enr_lmt)]
            if (np.shape(fltrd_enr_pnt)[0] > 0):
                srtd_fltrd_pnt = np.sort(fltrd_enr_pnt)
                top_srt_flt_pnt = srtd_fltrd_pnt[:3]
                new_lowst_pnt = np.nanmean(top_srt_flt_pnt)
            else:
                new_lowst_pnt = np.nanmin(ylow_points_2)

            new_lowst_pnt = new_lowst_pnt + lowest_point

            lwstpntrmd = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] - new_lowst_pnt
            lwstpntrmd[lwstpntrmd < 0] = 0
            lowest_pt_cumulative_sum = np.nanmean(lwstpntrmd)

            distances = np.abs(y_low_points - new_lowst_pnt)
            dist = np.nanmean(distances)
            level_filter = new_lowst_pnt + config['LAPDetection']['levelFilter']
            perc_pts_lying_below_level = np.sum(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] < level_filter) / \
                                         np.shape(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])[0]

    return dist, new_lowst_pnt, lowest_pt_cumulative_sum, perc_pts_lying_below_level
