"""
Date Created - 13 Nove 2018
Author name - Pratap
Major function in which all the sub functions are incorporated
"""

import copy
import logging
import numpy as np
from datetime import datetime

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.refrigerator.functions.get_lap_timestamps import get_lap_timestamps
from python3.disaggregation.aer.refrigerator.functions.get_consolidated_laps import get_consolidated_laps
from python3.disaggregation.aer.refrigerator.functions.get_lap_distances import get_lap_distances
from python3.disaggregation.aer.refrigerator.functions.get_ref_hourly_output import get_ref_hourly_output
from python3.disaggregation.aer.refrigerator.functions.get_seasonality_output import get_seasonality_output


def block_nan_output(ref_detection):

    """Utility to modify nan values in epoch level output to 0 so that we can post epoch level output correctly"""

    # In case we have some values as nan in timestamp level out

    ref_epoch = copy.deepcopy(ref_detection['refHourlyOutput'][:, 2])
    is_nan_epoch = np.isnan(ref_epoch)

    if np.sum(is_nan_epoch) < len(ref_epoch):
        ref_epoch[is_nan_epoch] = 0

    ref_detection['refHourlyOutput'][:, 2] = ref_epoch

    return ref_detection


def get_ref_estimation(input_data, config, make_hsm, hsm_in, bypass_hsm, logger_base):
    """
    This is final output script from the Ref module
    Computes monthly estimates by applying seasonality

    Parameters:
        input_data (np.ndarray):
        config (dict):
        make_hsm (bool):
        hsm_in (dict):
        bypass_hsm (bool):
        logger_base (logger):

    Returns:
        ref_detection (dict):
    """

    ref_detection = {
        'samplingRate': 0
    }

    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_ref_estimation")
    logger_pass = {"logger": logger_local,
                   "logging_dict": logger_base.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    ref_detection['input_data'] = input_data

    if make_hsm | bypass_hsm:
        logger.info('Running Mode without hsm for Ref Estimation |')
        logger.info('Total number of Available Billing Cycles - %d months |', np.shape(np.unique(input_data[:, 0]))[0] )

        # Get LAP Mid timestamps either 30 or below 6%
        ref_detection['lapMidTimestamps'] = get_lap_timestamps(input_data, config, logger_pass)

        # get Consolidated & merged LAPs with edge correction
        ref_detection['LAPs'], ref_detection['lapLowPoint'],ref_detection['trnctdLAPs'] = get_consolidated_laps(ref_detection, config, logger_pass)

        # obtain various LAP metrics like distance etc.
        lap_distances, lowest_point, lowest_point_cumulative_sum,perc_pts_lying_below_level, \
        dist_from_transitions, num_transitions_above, num_transitions_below = get_lap_distances(config, ref_detection, logger_pass)

        ref_detection['LAPs'] = np.hstack((ref_detection['LAPs'],lowest_point,lap_distances,
                                           lowest_point_cumulative_sum,perc_pts_lying_below_level,
                                           dist_from_transitions,num_transitions_above,num_transitions_below))

        ref_detection = get_ref_hourly_output(ref_detection, config, make_hsm, hsm_in, bypass_hsm, logger_pass)
        months = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])[:, np.newaxis]
        chk2_1 = np.zeros((np.shape(months)[0],3))
        chk2_2 = np.zeros((np.shape(months)[0],2))
        ref_hourly_output = ref_detection['refHourlyOutput']
        for mnth_idx in range(np.shape(months)[0]):
            mth_data = ref_hourly_output[ref_hourly_output[:,0]== months[mnth_idx,0] ,:]
            chk2_1[mnth_idx,0] = months[mnth_idx,0]
            chk2_1[mnth_idx,1] = np.sum(mth_data[:,2])
            chk2_1[mnth_idx,2] = np.mean(mth_data[:,2])

            chk2_2[mnth_idx,0] = months[mnth_idx,0]
            chk2_2[mnth_idx,1] = np.shape(mth_data[:,2])[0]

        avg_temp = np.zeros((1,1))
        for mnthind in range(np.shape(months)[0]):
            partic_month_data = \
                np.nanmean(input_data[(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == months[mnthind]), Cgbdisagg.INPUT_TEMPERATURE_IDX])
            avg_temp = np.vstack((avg_temp,partic_month_data))

        # Introduce Seasonality
        avg_temp = avg_temp[1:,:]
        nan_avg_temp =avg_temp[ ~np.isnan(avg_temp)][:,np.newaxis]
        if (np.shape(nan_avg_temp)[0] >=3):
            sorted_index = np.argsort(nan_avg_temp[:,0])
            coldest_month_temp = nan_avg_temp[sorted_index[0],0]
            second_hottest_month_temp = nan_avg_temp[sorted_index[np.shape(sorted_index)[0]-2],0]
            logger.info('Introducing seasonality ..., coldest Month Temp - %.3f  2nd Hottest Month Temp - %.3f  |',
                        coldest_month_temp,second_hottest_month_temp)
            scaling = 1 + (config['Seasonality']['secondHottestMonthSeasonalityScaling'] - 1) * \
                      (avg_temp-coldest_month_temp)/(second_hottest_month_temp-coldest_month_temp)
            scaling[np.isnan(scaling)] = 1
            scaling[scaling<1]=1
            scaling[scaling>config['Seasonality']['upperBoundSeasonalityScaling']] = config['Seasonality']['upperBoundSeasonalityScaling']
            ref_detection = get_seasonality_output(ref_detection, config, make_hsm, hsm_in, bypass_hsm, logger_pass)
            month_ref = np.copy(ref_detection['monthRef'])
        else:
            logger.warning('Skipping seasonality ... not enough billing cycles  |')
            coldest_month_temp = nan_avg_temp[0,0] - 0.5
            second_hottest_month_temp = nan_avg_temp[0,0] + 0.5
            scaling = 1
            unique_months = np.unique(ref_hourly_output[:,0])[:,np.newaxis]
            unique_months = np.hstack((unique_months,np.zeros((np.shape(unique_months)[0],1))))
            for month_index_to_fill in range(np.shape(unique_months)[0]):
                unique_months[month_index_to_fill,1] = np.nansum(ref_hourly_output[ref_hourly_output[:,0]==unique_months[month_index_to_fill,0],2])

            ref_detection['monthRef'] = np.copy(unique_months)
        month_net = np.copy(ref_detection['monthRef'])
        month_net[:, 1] = 0.0
        for month_net_index in range(np.shape(month_net)[0]):
            month_net[month_net_index, 1] = np.nansum(input_data[input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == month_net[month_net_index, 0], Cgbdisagg.INPUT_CONSUMPTION_IDX])
        ref_detection['monthNet'] = month_net
    else:
        logger.info('Running Mode with hsm for Ref Estimation |')
        ref_detection = get_ref_hourly_output(ref_detection, config, make_hsm, hsm_in, bypass_hsm, logger_pass)
        months = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])[:, np.newaxis]
        avg_temp = np.zeros((1,1))
        for mnthind in range(np.shape(months)[0]):
            partic_month_data = np.nanmean(input_data[(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == months[mnthind]), Cgbdisagg.INPUT_TEMPERATURE_IDX])
            avg_temp = np.vstack((avg_temp, partic_month_data))

    if ref_detection.get('monthRef') is None:
        zero_cons = np.zeros(shape=(months.shape[0],))
        month_ref = np.c_[months, zero_cons]
        ref_detection['monthRef'] = month_ref
    else:
        month_ref = np.copy(ref_detection['monthRef'])

    monthly_ref_log = [(datetime.utcfromtimestamp(month_ref[i, 0]).strftime('%b-%Y'),
                        np.round(month_ref[i, 1], 3))
                       for i in range(month_ref.shape[0])]

    logger.info('Monthly Estimates Ref: | {}'.format(str(monthly_ref_log).replace('\n', ' ')))

    ref_detection = block_nan_output(ref_detection)

    return ref_detection
