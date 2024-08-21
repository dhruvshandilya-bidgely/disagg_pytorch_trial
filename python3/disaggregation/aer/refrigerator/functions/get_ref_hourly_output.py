"""
Date Created - 13 Nov 2018
Author name - Pratap
Ref Hourly Output is crucial part in estimation in which estimates are calculated for all the seasons
"""
import os
import logging
import numpy as np
from datetime import datetime

from python3.utils.maths_utils.matlab_utils import percentile_1d

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.refrigerator.LapColumns import LapColumns
from python3.disaggregation.aer.refrigerator.functions.get_lowest_point_clustering import get_lowest_point_clustering
from python3.utils.time.time_utils import unix2mthdateint, unix2yeardateint


def season_temp_seg(first_temp_seg, second_temp_seg, summer_temp_seg, winter_temp_seg, inter_temp_seg):
    """ find temp seg """
    if (np.shape(first_temp_seg)[0] >= 3) & (np.shape(second_temp_seg)[0] >= 3) & (np.shape(inter_temp_seg)[0] >= 3):
        if np.median(first_temp_seg[:, 3]) > np.median(second_temp_seg[:, 3]):
            summer_temp_seg = first_temp_seg
            winter_temp_seg = second_temp_seg
        else:
            summer_temp_seg = second_temp_seg
            winter_temp_seg = first_temp_seg
    elif (np.shape(first_temp_seg)[0] >= 3) & (np.shape(inter_temp_seg)[0] >= 3):
        if np.median(first_temp_seg[:, 3]) > np.median(inter_temp_seg[:, 3]):
            summer_temp_seg = first_temp_seg
        else:
            winter_temp_seg = first_temp_seg
    elif (np.shape(second_temp_seg)[0] >= 3) & (np.shape(inter_temp_seg)[0] >= 3):
        if np.median(second_temp_seg[:, 3]) > np.median(inter_temp_seg[:, 3]):
            summer_temp_seg = second_temp_seg
        else:
            winter_temp_seg = second_temp_seg

    return summer_temp_seg, winter_temp_seg


def lap_energy_std(input_data, filtered_laps, ref_detection, energy_per_data_point, std_per_data_point,
                   enr_low_pnt_data, config):
    """ get energy and stdev of laps """
    for jj in range(np.shape(filtered_laps)[0]):
        lap_j = filtered_laps[jj, :]
        lap_j = np.transpose(lap_j[:, np.newaxis])
        idx_in_month_to_filter = (
            (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] >= lap_j[:, LapColumns.LAP_START_EPOCH_IDX]) &
            (input_data[:, Cgbdisagg.INPUT_EPOCH_IDX] <= lap_j[:, LapColumns.LAP_END_EPOCH_IDX]))

        lowest_pt = np.asscalar(lap_j[:, LapColumns.LAP_LOWEST_PT_IDX])
        test_chk = input_data[idx_in_month_to_filter, Cgbdisagg.INPUT_CONSUMPTION_IDX]
        # Calculate metrics for LAP
        abc_test = test_chk[1:(np.shape(test_chk)[0] - 1)]

        new_lowst_pnt, incrval, trncatedseries = get_lowest_point_clustering(enr_low_pnt_data, lap_j, config,
                                                                             abc_test, lowest_pt)

        ref_detection['trnctdsgnl'] = np.vstack((ref_detection['trnctdsgnl'], trncatedseries[:, np.newaxis]))
        ref_detection['sixtyprctle'][jj, 0] = np.max(trncatedseries)
        ref_detection['svntyprctle'][jj, 0] = lowest_pt + incrval
        ref_detection['egthyprctle'][jj, 0] = new_lowst_pnt

        updtd_trncd_srs = trncatedseries - new_lowst_pnt
        updtd_trncd_srs[updtd_trncd_srs < 0] = 0
        energy_per_data_point[jj, 0] = np.mean(updtd_trncd_srs)
        std_per_data_point_chk = trncatedseries - lowest_pt
        sum_vari = np.sum(np.multiply(std_per_data_point_chk - np.mean(std_per_data_point_chk),
                                      std_per_data_point_chk - np.mean(std_per_data_point_chk)))
        sum_vari_sd = np.sqrt(sum_vari / np.shape(std_per_data_point_chk)[0])
        std_per_data_point[jj, 0] = sum_vari_sd

    return energy_per_data_point, std_per_data_point, ref_detection


def get_seasonal_estimate(summer_laps, config, logger, season):
    """ get seasonal estimate """
    seasonal_estimated_ref = np.nan
    seasonal_std_ref = np.nan
    orig = np.nan
    seasonal_estmtd_mnth = np.nan

    if (np.shape(summer_laps)[0] >= 3):
        seasonal_top_ten_laps = summer_laps[:10, :]
        seasonal_estimated_ref, seasonal_std_ref, seasonal_estmtd_mnth = energy_per_data_pt(seasonal_top_ten_laps,
                                                                                            config, season, logger)
        seasonal_estimated_ref = post_process_ref_estimation(seasonal_estimated_ref, config, season, logger)
        orig = np.shape(summer_laps)[0]
        seasonal_top_ten_laps[:, 15] = (seasonal_top_ten_laps[:, 1] - seasonal_top_ten_laps[:, 0]) / Cgbdisagg.SEC_IN_HOUR
        logger.info(season + ' Monthly Ref Estimate is : %.3f kWh |',
                    seasonal_estimated_ref * 30 * 24 * 3.6 / config['samplingRate'])
    else:
        logger.info('Do not have enough 3 LAPs for ' + season + ' Estimation |')

    return seasonal_estimated_ref, seasonal_std_ref, seasonal_estmtd_mnth, orig


def write_csv_file(write_file_flag, orig, flt, season_std_ref, season_estimated_ref, config):
    """ write csv file """

    if write_file_flag:
        file_name = os.getcwd() + '/refDebug.csv'
        orig1 = orig.get("orig1")
        orig2 = orig.get("orig2")
        orig3 = orig.get("orig3")
        orig4 = orig.get("orig4")
        orig5 = orig.get("orig5")

        flt1 = flt.get("flt1")
        flt2 = flt.get("flt2")
        flt3 = flt.get("flt3")
        flt4 = flt.get("flt4")
        flt5 = flt.get("flt5")
        flt6 = flt.get("flt6")

        estimated_ref = season_estimated_ref.get("estimated_ref")
        smr_estimated_ref = season_estimated_ref.get("smr_estimated_ref")
        wtr_estimated_ref = season_estimated_ref.get("wtr_estimated_ref")
        itr_estimated_ref = season_estimated_ref.get("itr_estimated_ref")

        std_ref = season_std_ref.get("_std_ref")
        smr_std_ref = season_std_ref.get("_std_ref")
        wtr_std_ref = season_std_ref.get("_std_ref")
        itr_std_ref = season_std_ref.get("_std_ref")

        if not os.path.isfile(file_name):
            with open(file_name, 'w') as fid_w2:
                string_to_write = 'UUID,InitLAPs,FinalLAPs,estRef,stdRef,Filt1blckd,Filt2blckd,' \
                                  'Filt3blckd,Filt4blckd,Filt5blckd,Filt6blckd'
                fid_w2.write('%s\n' % string_to_write)
                fid_w2.close()
        fid_w2 = open(file_name, 'a')
        string_to_write = '%s,%d,%d,%f,%f,%d,%d,%d,%d,%d,%d' % (config['UUID'], orig1, orig2,
                                                                estimated_ref * 30 * 24 * 3.6 / config['samplingRate'],
                                                                std_ref, orig1 - flt1,
                                                                orig1 - flt2, orig1 - flt3, orig1 - flt4,
                                                                orig1 - flt5, orig1 - flt6)
        fid_w2.write('%s\n' % string_to_write)
        fid_w2.close()

        file_name2 = os.getcwd() + '/refEstimates.csv'
        if not os.path.isfile(file_name2):
            with open(file_name2, 'w') as fid_w3:
                string_to_write = 'UUID,InitLAPs,FinalLAPs,estRef,stdRef,smrFnlLAPs,smrEstRef,\
                smrStdRef,wtrFnlLAPs,wtrEstRef,wtrStdRef,itrFnlLAPs,itrEstRef,itrStdRef'
                fid_w3.write('%s\n' % string_to_write)
                fid_w3.close()

        fid_w3 = open(file_name2, 'a')
        string_to_write = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % (config['UUID'], orig1, orig2,
                                                                         estimated_ref * 30 * 24 * 3.6 / config[
                                                                             'samplingRate'],
                                                                         std_ref, orig3,
                                                                         smr_estimated_ref * 30 * 24 * 3.6 / config[
                                                                             'samplingRate'],
                                                                         smr_std_ref, orig4,
                                                                         wtr_estimated_ref * 30 * 24 * 3.6 / config[
                                                                             'samplingRate'],
                                                                         wtr_std_ref, orig5,
                                                                         itr_estimated_ref * 30 * 24 * 3.6 / config[
                                                                             'samplingRate'],
                                                                         itr_std_ref)
        fid_w3.write('%s\n' % string_to_write)
        fid_w3.close()

    return


def monthly_avg_temperature(input_data, months, avg_temp):
    """ get monthly average temp """
    for mnth_ind in range(np.shape(months)[0]):
        partic_mnth_dta = np.nanmean(
            input_data[(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == months[mnth_ind]), Cgbdisagg.INPUT_TEMPERATURE_IDX])
        avg_temp = np.vstack((avg_temp, partic_mnth_dta))

    return avg_temp


def get_month_year(avg_temp_chkmnth):
    """ get month and year number """
    for i in range(np.shape(avg_temp_chkmnth)[0]):
        avg_temp_chkmnth[i, 1] = unix2mthdateint(avg_temp_chkmnth[i, 0])
        avg_temp_chkmnth[i, 2] = unix2yeardateint(avg_temp_chkmnth[i, 0])

    return avg_temp_chkmnth


def filter_laps(sorted_filtered_laps, temp_seg):
    """ get laps in each temp seg """

    summer_temp_seg = temp_seg.get("summer_temp_seg")
    winter_temp_seg = temp_seg.get("winter_temp_seg")
    inter_temp_seg = temp_seg.get("inter_temp_seg")

    # noinspection PyBroadException
    try:
        summer_laps = sorted_filtered_laps[np.isin(sorted_filtered_laps[:, 2], summer_temp_seg[:, 0]), :]
    except Exception:
        summer_laps = np.zeros((1, 16))

    # noinspection PyBroadException
    try:
        winter_laps = sorted_filtered_laps[np.isin(sorted_filtered_laps[:, 2], winter_temp_seg[:, 0]), :]
    except Exception:
        winter_laps = np.zeros((1, 16))

    # noinspection PyBroadException
    try:
        inter_laps = sorted_filtered_laps[np.isin(sorted_filtered_laps[:, 2], inter_temp_seg[:, 0]), :]
    except Exception:
        inter_laps = np.zeros((1, 16))

    return summer_laps, winter_laps, inter_laps


def get_ref_hourly_output(ref_detection, config, hist_mode, hsm_in, bypass_hsm, logger_base):
    """
    Function for calculating Ref Estimation after applying
    filtering, scoring etc.
    """
    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_ref_hourly_output")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    input_data = np.copy(ref_detection['input_data'])
    threshold_perc_pts_lying_below_level = config['Estimation']['threshold_percPtsLyingBelowLevel']
    unique_month_ts = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])
    ref_hourly_output = np.hstack((input_data[:,[Cgbdisagg.INPUT_BILL_CYCLE_IDX, Cgbdisagg.INPUT_EPOCH_IDX]],
                                   np.zeros((np.shape(input_data)[0], 1)), input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX][:, np.newaxis]))

    amp_month = np.zeros((np.shape(unique_month_ts)[0], 1))
    period_month = np.zeros((np.shape(unique_month_ts)[0], 1))

    if hist_mode | bypass_hsm:
        lap_low_point = ref_detection['lapLowPoint']
        enr_low_pnt_data = np.hstack((lap_low_point, input_data[:, 6][:, np.newaxis]))

        laps = np.copy(ref_detection['LAPs'])
        orig1 = np.shape(laps)[0]
        filtered_laps = np.copy(laps)
        num_lap_hours = (filtered_laps[:, LapColumns.LAP_END_EPOCH_IDX] - filtered_laps[:, LapColumns.LAP_START_EPOCH_IDX]) / Cgbdisagg.SEC_IN_HOUR

        szchk1_alap = np.shape(laps)[0]
        logger.info('No.of LAPs before Filtering : %d |',szchk1_alap )

        # Applied 6 Filters on LAPs
        filtered_laps = filtered_laps[
            (num_lap_hours >= config['Estimation']['minNumHourForLAP']) &
            (filtered_laps[:, LapColumns.LAP_PERC_PTS_LYING_BELOW_IDX] >
             config['Estimation']['threshold_percPtsLyingBelowLevel']) &
            (filtered_laps[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX] > config['Estimation']['MinMeanAmpTransitions']) &
            (filtered_laps[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX] < config['Estimation']['MaxMeanAmpTransitions']) &
            (filtered_laps[:, LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX] >
             config['Estimation']['MinNumTransitionsBelow'] * num_lap_hours) &
            (filtered_laps[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX] <
             config['Estimation']['MaxNumTransitionsAbove'] * num_lap_hours), :]
        orig2 = np.shape(filtered_laps)[0]

        # Following LAP filters applied one by one in series at each stage to
        # show effectiveness of filtering process
        chk1_alap = np.copy(laps)
        chk1_alap =np.hstack((chk1_alap,num_lap_hours[:,np.newaxis]))
        chk2_alap = chk1_alap[num_lap_hours >= config['Estimation']['minNumHourForLAP'], :]
        chk3_alap = chk2_alap[(chk2_alap[:, LapColumns.LAP_PERC_PTS_LYING_BELOW_IDX] >
                               config['Estimation']['threshold_percPtsLyingBelowLevel']), :]
        chk4_alap = chk3_alap[(chk3_alap[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX] >
                               config['Estimation']['MinMeanAmpTransitions']), :]
        chk5_alap = chk4_alap[(chk4_alap[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX] <
                               config['Estimation']['MaxMeanAmpTransitions']), :]
        chk6_alap = chk5_alap[(chk5_alap[:, LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX] >
                               config['Estimation']['MinNumTransitionsBelow'] * chk5_alap[:, 10]), :]
        chk7_alap = chk6_alap[(chk6_alap[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX] <
                               config['Estimation']['MaxNumTransitionsAbove'] * chk6_alap[:, 10]), :]

        szchk2_alap = np.shape(chk2_alap)[0]
        szchk3_alap = np.shape(chk3_alap)[0]
        szchk4_alap = np.shape(chk4_alap)[0]
        szchk5_alap = np.shape(chk5_alap)[0]
        szchk6_alap = np.shape(chk6_alap)[0]
        szchk7_alap = np.shape(chk7_alap)[0]

        prchk2_alap = 0.0
        prchk3_alap = 0.0
        prchk4_alap = 0.0
        prchk5_alap = 0.0
        prchk6_alap = 0.0
        prchk7_alap = 0.0
        prchk8_alap = 0.0
        prchk9_alap = 0.0

        if (szchk1_alap > 0):
            prchk2_alap = (szchk2_alap / szchk1_alap) * 100
            prchk3_alap = (szchk3_alap / szchk1_alap) * 100
            prchk4_alap = (szchk4_alap / szchk1_alap) * 100
            prchk5_alap = (szchk5_alap / szchk1_alap) * 100
            prchk6_alap = (szchk6_alap / szchk1_alap) * 100
            prchk7_alap = (szchk7_alap / szchk1_alap) * 100

        logger.info('No.of LAPs Passed through Filter 1: %d which is %.3f %% |' ,szchk2_alap,prchk2_alap)
        logger.info('No.of LAPs Passed through Filter 2: %d which is %.3f %% |' ,szchk3_alap,prchk3_alap)
        logger.info('No.of LAPs Passed through Filter 3: %d which is %.3f %% |' ,szchk4_alap,prchk4_alap)
        logger.info('No.of LAPs Passed through Filter 4: %d which is %.3f %% |' ,szchk5_alap,prchk5_alap)
        logger.info('No.of LAPs Passed through Filter 5: %d which is %.3f %% |' ,szchk6_alap,prchk6_alap)
        logger.info('No.of LAPs Passed through Filter 6: %d which is %.3f %% |' ,szchk7_alap,prchk7_alap)

        # following filters are independently applied to check which filter is
        # causing issue the most at individual level
        chk1_ilap = np.copy(laps)
        chk2_ilap = chk1_ilap[(num_lap_hours >= config['Estimation']['minNumHourForLAP']), :]
        chk3_ilap = chk1_ilap[(chk1_ilap[:, LapColumns.LAP_PERC_PTS_LYING_BELOW_IDX] > config['Estimation']['threshold_percPtsLyingBelowLevel']), :]
        chk4_ilap = chk1_ilap[(chk1_ilap[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX] > config['Estimation']['MinMeanAmpTransitions']), :]
        chk5_ilap = chk1_ilap[(chk1_ilap[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX] < config['Estimation']['MaxMeanAmpTransitions']), :]
        chk6_ilap = chk1_ilap[(chk1_ilap[:, LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX] > config['Estimation']['MinNumTransitionsBelow'] * num_lap_hours), :]
        chk7_ilap = chk1_ilap[(chk1_ilap[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX] < config['Estimation']['MaxNumTransitionsAbove'] * num_lap_hours), :]

        flt1 = np.shape(chk2_ilap)[0]
        flt2 = np.shape(chk3_ilap)[0]
        flt3 = np.shape(chk4_ilap)[0]
        flt4 = np.shape(chk5_ilap)[0]
        flt5 = np.shape(chk6_ilap)[0]
        flt6 = np.shape(chk7_ilap)[0]

        num_lap_hours = (filtered_laps[:, LapColumns.LAP_END_EPOCH_IDX] -
                         filtered_laps[:, LapColumns.LAP_START_EPOCH_IDX]) / Cgbdisagg.SEC_IN_HOUR
        chk_laphours = num_lap_hours/np.sum(num_lap_hours)

        filtered_laps_normalized = np.copy(filtered_laps)

        # Normalizing LAP features to create same dimension for all metrics
        # between 0 and 1
        filtered_laps_normalized[:,LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX] = \
            filtered_laps_normalized[:, LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX] * 10.0 / num_lap_hours
        filtered_laps_normalized[:,LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX] = \
            filtered_laps_normalized[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX] * 10.0 / num_lap_hours


        sum_lap_dist = np.sum(filtered_laps_normalized[:, np.r_[LapColumns.LAP_LAP_DISTANCE_IDX: (LapColumns.LAP_LOWEST_PT_CUM_SUM_IDX + 1)]], axis=0)
        sum_lap_dist = np.transpose(sum_lap_dist[:, np.newaxis])

        filtered_laps_normalized[:, np.r_[LapColumns.LAP_LAP_DISTANCE_IDX: (LapColumns.LAP_LOWEST_PT_CUM_SUM_IDX + 1)]] = \
            filtered_laps_normalized[:, np.r_[LapColumns.LAP_LAP_DISTANCE_IDX: (LapColumns.LAP_LOWEST_PT_CUM_SUM_IDX + 1)]] / sum_lap_dist

        sm_lap_trns = np.sum(filtered_laps_normalized[:, np.r_[LapColumns.LAP_MEAN_TRANSITIONS_IDX:(LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX + 1)]], axis=0)
        sm_lap_trns = np.transpose(sm_lap_trns[:,np.newaxis])
        filtered_laps_normalized[:, np.r_[LapColumns.LAP_MEAN_TRANSITIONS_IDX:(LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX + 1)]] = \
            filtered_laps_normalized[:, np.r_[LapColumns.LAP_MEAN_TRANSITIONS_IDX:(LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX + 1)]] / sm_lap_trns

        filtered_laps_normalized = np.hstack((filtered_laps_normalized,chk_laphours[:,np.newaxis]))


        if np.sum(np.isnan(filtered_laps_normalized[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX])) > 0:
            filtered_laps_normalized[np.isnan(filtered_laps_normalized[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX]), LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX] = \
                1 / np.sum(np.isnan(filtered_laps_normalized[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX]))

        # Sorting the LAPs in ascending order based on overall score
        sort_idx = (0.4 * filtered_laps_normalized[:, LapColumns.LAP_LAP_DISTANCE_IDX]
                    +0.1 * filtered_laps_normalized[:,  LapColumns.LAP_LOWEST_PT_CUM_SUM_IDX]
                    +0.1 * filtered_laps_normalized[:, LapColumns.LAP_NUM_TRANSITIONS_ABOVE_IDX]
                    -0.3 * filtered_laps_normalized[:, LapColumns.LAP_NUM_TRANSITIONS_BELOW_IDX]
                    -0.1 * filtered_laps_normalized[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX])
        sort_idx_new = np.argsort(sort_idx)

        filtered_laps = np.array(filtered_laps)[sort_idx_new]
        energy_per_data_point = np.zeros((np.shape(filtered_laps)[0], 1))
        std_per_data_point = np.zeros((np.shape(filtered_laps)[0], 1))

        ref_detection['sixtyprctle'] = np.zeros((np.shape(filtered_laps)[0], 1))
        ref_detection['svntyprctle'] = np.zeros((np.shape(filtered_laps)[0], 1))
        ref_detection['egthyprctle'] = np.zeros((np.shape(filtered_laps)[0], 1))
        ref_detection['trnctdsgnl'] = np.zeros((1, 1))

        if (np.shape(filtered_laps)[0] >= config['Estimation']['Min_LAPs_Reqd_Estimation']):
            energy_per_data_point, std_per_data_point, ref_detection = lap_energy_std(input_data, filtered_laps, ref_detection,
                                                                                      energy_per_data_point, std_per_data_point,
                                                                                      enr_low_pnt_data, config)

        num_laps_used = np.shape(filtered_laps)[0]
        months = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])
        avg_temp = np.zeros((1,1))

        # Find average temperature of each month
        avg_temp = monthly_avg_temperature(input_data, months, avg_temp)

        avg_temp = avg_temp[1:, :]
        avg_temp_chkmnth = np.hstack((months[:, np.newaxis], np.zeros((np.shape(months)[0], 2)), avg_temp))

        avg_temp_chkmnth = get_month_year(avg_temp_chkmnth)

        avg_temp_chkmnth2 = avg_temp_chkmnth[~np.isnan(avg_temp_chkmnth[:,3]), :]

        monthly_temp_log = [(datetime.utcfromtimestamp(avg_temp_chkmnth2[i, 0]).strftime('%b-%Y'),
                             np.round(avg_temp_chkmnth2[i, 3], 3))
                            for i in range(avg_temp_chkmnth2.shape[0])]

        logger.info('Monthly Avg temperature: | {}'.format(str(monthly_temp_log).replace('\n', ' ')))

        # for checking laps months in avg temp list
        lapmembtopass = np.unique(filtered_laps[:,LapColumns.LAP_MONTH_EPOCH_IDX])
        avg_temp_chkmnth3 = avg_temp_chkmnth2[np.isin(avg_temp_chkmnth2[:,0],lapmembtopass),:]

        # Seasonality based on Jun-Jul-Aug & Dec-Jan-Feb combination to decide
        # summer and winter w.r.to temperature
        fmembtopass = np.array([[6, 7, 8]])
        smembtopass = np.array([1, 2, 12])
        imembtopass = np.array([3, 4, 5, 9, 10, 11])

        first_temp_seg = avg_temp_chkmnth3[np.isin(avg_temp_chkmnth3[:,1],fmembtopass),:]
        second_temp_seg = avg_temp_chkmnth3[np.isin(avg_temp_chkmnth3[:,1],smembtopass),:]
        inter_temp_seg = avg_temp_chkmnth3[np.isin(avg_temp_chkmnth3[:,1],imembtopass),:]

        summer_temp_seg = np.nan
        winter_temp_seg = np.nan

        summer_temp_seg, winter_temp_seg = season_temp_seg(first_temp_seg, second_temp_seg, summer_temp_seg,
                                                           winter_temp_seg, inter_temp_seg)

        ref_detection['summerTempSeg'] = summer_temp_seg
        ref_detection['winterTempSeg'] = winter_temp_seg
        ref_detection['interTempSeg'] = inter_temp_seg

        sorted_filtered_laps = np.hstack((filtered_laps, energy_per_data_point, std_per_data_point, ref_detection['sixtyprctle'],
                                          ref_detection['svntyprctle'], ref_detection['egthyprctle']))

        sorted_filtered_laps = np.hstack((sorted_filtered_laps,
                                          np.round(np.divide(np.round(sorted_filtered_laps[:,LapColumns.LAP_LAP_DISTANCE_IDX], 2),
                                                             np.round(sorted_filtered_laps[:, LapColumns.LAP_MEAN_TRANSITIONS_IDX], 2))[:, np.newaxis], 3)))

        # Remove uneven LAPs in case if both median energy and
        # median ratio is greater than 10%
        if np.shape(sorted_filtered_laps)[0] >= 10:
            topt_med_est = 1.1 * np.median(sorted_filtered_laps[:10, 10])
            topt_rto_est = 1.1 * np.median(sorted_filtered_laps[:10, 15])
            sorted_filtered_laps = sorted_filtered_laps[ ~((sorted_filtered_laps[:, 10] > topt_med_est ) &
                                                           (sorted_filtered_laps[:, 15] > topt_rto_est)), :]

        szchk8_alap = np.shape(sorted_filtered_laps)[0]
        sorted_filtered_laps = sorted_filtered_laps[sorted_filtered_laps[:, 10] > 3.0, :]
        szchk9_alap = np.shape(sorted_filtered_laps)[0]

        if (szchk1_alap > 0):
            prchk8_alap = (szchk8_alap / szchk1_alap) * 100
            prchk9_alap = (szchk9_alap / szchk1_alap) * 100

        logger.info('No.of LAPs Passed through Filter 1.1* Median & 1.1*Ratio of Dist/Mean Dist: %d which is %.3f %% |', szchk8_alap, prchk8_alap)
        logger.info('No.of LAPs Passed through Filter  > 3.0: %d which is %.3f %% |', szchk9_alap, prchk9_alap)

        temp_seg = {
            "summer_temp_seg": summer_temp_seg,
            "winter_temp_seg": winter_temp_seg,
            "inter_temp_seg": inter_temp_seg
        }

        # Segregate LAPs for each season
        summer_laps, winter_laps, inter_laps = filter_laps(sorted_filtered_laps, temp_seg)

        estimated_ref = np.nan
        std_ref = np.nan
        estmtd_mnth = np.nan

        # Overall Estimation
        if (np.shape(sorted_filtered_laps)[0] >= 3):
            total_top_ten_laps = sorted_filtered_laps[:10, :]
            estimated_ref,std_ref,estmtd_mnth = energy_per_data_pt(total_top_ten_laps, config, 'Total', logger)
            estimated_ref = post_process_ref_estimation(estimated_ref, config, 'Total', logger)

            logger.info('Monthly Ref Estimate is : %.3f kWh |', estimated_ref * 30 * 24 * 3.6 / config['samplingRate'])
        else:
            logger.info('Do not have enough 3 LAPs for Total Estimation |')

        # Summer Estimate
        smr_estimated_ref, smr_std_ref, smr_estmtd_mnth, orig3 = get_seasonal_estimate(summer_laps, config, logger,
                                                                                       "Summer")

        # Winter Estimate
        wtr_estimated_ref, wtr_std_ref, wtr_estmtd_mnth, orig4 = get_seasonal_estimate(winter_laps, config, logger,
                                                                                       "Winter")

        # Intermediate Estimate
        itr_estimated_ref, itr_std_ref, itr_estmtd_mnth, orig5 = get_seasonal_estimate(inter_laps, config, logger,
                                                                                       "Inter")

        ref_hourly_output[(ref_hourly_output[:,3] >= 2), 2] = estimated_ref
        amp_month[:, 0] = estimated_ref
        period_month[:, 0] = estimated_ref
        ref_detection['numLapsUsed'] = num_laps_used
        # Total, Summer, Winter & Intermediate months estimates
        ref_detection['estimatedRef'] = estimated_ref
        ref_detection['estmtdMnth'] = estmtd_mnth
        ref_detection['smrEstimatedRef'] = smr_estimated_ref
        ref_detection['smrEstmtdMnth'] = smr_estmtd_mnth
        ref_detection['wtrEstimatedRef'] = wtr_estimated_ref
        ref_detection['wtrEstmtdMnth']=wtr_estmtd_mnth
        ref_detection['itrEstimatedRef'] = itr_estimated_ref
        ref_detection['itrEstmtdMnth'] = itr_estmtd_mnth
        orig3 = np.shape(sorted_filtered_laps)[0]

        # following code is used for writing various metrics into sheet
        write_file_flag = 0

        season_estimated_ref = {
            "estimated_ref": estimated_ref,
            "smr_estimated_ref": smr_estimated_ref,
            "wtr_estimated_ref": wtr_estimated_ref,
            "itr_estimated_ref": itr_estimated_ref
        }

        orig = {
            "orig1": orig1,
            "orig2": orig2,
            "orig3": orig3,
            "orig4": orig4,
            "orig5": orig5
        }

        flt = {
            "flt1": flt1,
            "flt2": flt2,
            "flt3": flt3,
            "flt4": flt4,
            "flt5": flt5,
            "flt6": flt6
        }

        season_std_ref = {
            "std_ref": std_ref,
            "smr_std_ref": smr_std_ref,
            "wtr_std_ref": wtr_std_ref,
            "itr_std_ref": itr_std_ref
        }

        write_csv_file(write_file_flag, orig, flt, season_std_ref, season_estimated_ref, config)

    else:
        hsm = hsm_in
        ref_from_hsm = get_ref_from_hsm(hsm)

        estimated_ref_hsm = post_process_ref_estimation(ref_from_hsm, config, 'Total', logger)
        ref_hourly_output[(ref_hourly_output[:, 3] >= 2), 2] = estimated_ref_hsm
        logger.info('HSM Total Monthly Ref Estimate is : | %.3f kWh', estimated_ref_hsm*30*24*3.6/config['samplingRate'])
        ref_detection['numLapsUsed'] = -1

    ref_hourly_output = ref_hourly_output[:,:3]
    ref_detection['refHourlyOutput'] = ref_hourly_output
    ref_detection['ampMonth'] = amp_month
    ref_detection['periodMonth'] = period_month
    ref_detection['threshold_percPtsLyingBelowLevel'] = threshold_perc_pts_lying_below_level

    return ref_detection


def energy_per_data_pt(energy_laps, refconfig, season, logger):
    """
    Following function is used for checking 80th/20th percentile ratio
    """
    eng_per_data_point = energy_laps[:, 10]
    # if Max Est is less than Max Limit dont perform 80/20 ratio test
    if np.max(eng_per_data_point) <\
            (refconfig['Estimation']['MaxMonthRefEstimate'] /
             (30 * 24 * Cgbdisagg.SEC_IN_HOUR / refconfig['samplingRate'])) or\
            ((percentile_1d(eng_per_data_point, 80) / percentile_1d(eng_per_data_point, 20)) <
             refconfig['Estimation']['MaxRatioLAPEnergyEstimates']):

        estimated_ref = np.nanmedian(eng_per_data_point)
        std_ref = np.nanstd(eng_per_data_point, ddof=1)
        energy_laps = np.hstack((energy_laps, np.abs(eng_per_data_point - estimated_ref)[:, np.newaxis]))
        energy_laps[:, 16] = np.round(energy_laps[:, 16], 3)
        energy_laps_2 = energy_laps[np.lexsort((energy_laps[:, 2], energy_laps[:, 16]))]
        estimated_ref_mnth = energy_laps_2[0, 2]
    else:
        estimated_ref = np.nan
        std_ref = np.nan
        estimated_ref_mnth = np.nan
        logger.info(season + ' Ref Estimate is not holding 80/20th percentile  < 2.25 |')

    return estimated_ref, std_ref, estimated_ref_mnth


def post_process_ref_estimation(estimated_ref, config, season, logger):
    """
    Function used for checking the energy bounds
    """
    lower_bound = config['Estimation']['MinMonthRefEstimate']/(30 * 24 * Cgbdisagg.SEC_IN_HOUR/config['samplingRate'])
    upper_bound = config['Estimation']['MaxMonthRefEstimate']/(30 * 24 * Cgbdisagg.SEC_IN_HOUR/config['samplingRate'])

    if ((estimated_ref > upper_bound) | (estimated_ref < lower_bound)):
        estimated_ref = np.nan
        logger.info(season + ' Ref Estimate is out of bounds %.3f kWh and %.3f kWh ',
                    config['Estimation']['MinMonthRefEstimate']/1000,
                    config['Estimation']['MaxMonthRefEstimate']/1000)

    return estimated_ref


def get_ref_from_hsm(hsm):

    """Utility to return ref from hsm"""

    ref_from_hsm = hsm['attributes'].get('Ref_Energy_Per_DataPoint')

    # Put in to handle a crash

    if ref_from_hsm is None or type(ref_from_hsm) == str:
        ref_from_hsm = 0
    elif type(ref_from_hsm) == list or type(ref_from_hsm) == np.ndarray:
        ref_from_hsm = ref_from_hsm[0]

    return ref_from_hsm


def nan_helper(y):
    """ nan helper function is used for enabling interpolation  """
    return np.isnan(y), lambda z: z.nonzero()[0]
