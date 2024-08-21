"""
Date Created - 13 Nov 2018
Author name - Pratap
Edge correction procedure applied on consolidated LAPs
"""
import logging
import numpy as np

from python3.config.Cgbdisagg import Cgbdisagg


def forward_sign(lap_data_vals, config):
    """ forward combine """

    for iter_combined_fwd in range(1, int(Cgbdisagg.SEC_IN_HOUR * 1.5 / config['samplingRate'])):
        if (iter_combined_fwd < len(lap_data_vals)) and (np.sign(lap_data_vals[iter_combined_fwd, 3]) == np.sign(lap_data_vals[0, 3])):
            lap_data_vals[iter_combined_fwd, 4] = 1
        else:
            break
    lap_data_vals[:, 2] = lap_data_vals[:, 2] * -1
    lap_data_vals[:, 2] = np.append(0, lap_data_vals[:(np.shape(lap_data_vals)[0] - 1), 2])
    lap_data_vals[:, 3] = np.sign(lap_data_vals[:, 2])

    return lap_data_vals


def backward_sign(lap_data_vals, config):
    """ backward combine """

    for iter_comb_bkwd in range((np.shape(lap_data_vals)[0] - 2),
                                (np.shape(lap_data_vals)[0] - int(Cgbdisagg.SEC_IN_HOUR * 1.5 /
                                                                  config['samplingRate'])) - 1, -1):
        if (iter_comb_bkwd >= 1) and np.sign(lap_data_vals[iter_comb_bkwd, 3]) == np.sign(lap_data_vals[np.shape(lap_data_vals)[0] - 1, 3]):
            lap_data_vals[iter_comb_bkwd, 4] = 1
        else:
            break

    return lap_data_vals


def forward_transition(lap_data_vals, stdv_list, med_val, config):
    """ forward transition """
    n_points_to_check = int(np.min([(Cgbdisagg.SEC_IN_HOUR * 1.5) / config['samplingRate'], len(stdv_list)]))

    for iter_med_fwd in range(n_points_to_check):
        interlap_vals = lap_data_vals[iter_med_fwd:int(np.ceil(np.shape(lap_data_vals)[0] / 2))]
        interlap_vals_2 = interlap_vals[interlap_vals[:, 4] == 0, 1]
        stdvl = std_median(interlap_vals_2, med_val)
        stdv_list[iter_med_fwd] = stdvl
    stdev_list2 = np.zeros((np.shape(lap_data_vals)[0], 1))

    return stdev_list2


def backward_transition(lap_data_vals, med_val, stdev_list2, config):
    """ backward transition """

    indices_to_check = range((np.shape(lap_data_vals)[0] - 1), (np.shape(lap_data_vals)[0] -
                                                                int(Cgbdisagg.SEC_IN_HOUR * 1.5 /
                                                                    config['samplingRate'])) - 1, -1)

    if len(stdev_list2) <= 2:
        indices_to_check = range(len(stdev_list2) - 1, -len(stdev_list2) - 1)

    for iter_med_bkwd in indices_to_check:
        interlap_vals = lap_data_vals[(int(np.ceil(np.shape(lap_data_vals)[0] / 2)) - 1):(iter_med_bkwd + 1)]
        interlap_vals_2 = interlap_vals[interlap_vals[:, 4] == 0, 1]
        stdvl = std_median(interlap_vals_2, med_val)
        stdev_list2[iter_med_bkwd] = stdvl

    return stdev_list2


def combine_for_back(lap_data_vals_4):
    """ combine forward and backward """
    for iter_cmb_points in range(np.shape(lap_data_vals_4)[0]):
        if lap_data_vals_4[iter_cmb_points, 5] != 0:
            lap_data_vals_4[iter_cmb_points, 9] = lap_data_vals_4[iter_cmb_points, 5]
            lap_data_vals_4[iter_cmb_points, 10] = lap_data_vals_4[iter_cmb_points, 7]
        else:
            lap_data_vals_4[iter_cmb_points, 9] = lap_data_vals_4[iter_cmb_points, 6]
            lap_data_vals_4[iter_cmb_points, 10] = lap_data_vals_4[iter_cmb_points, 8]

    return lap_data_vals_4


def forward_statistical(dtp_chk_trnc, new_std_series, lowest_point_edge, med_val_edge, config):
    """ forward statistical """
    for itr_ec_fwd in range(dtp_chk_trnc):
        if (new_std_series[itr_ec_fwd, 11] > config['LAPDetection']['PerStdTrnc']) &\
                ((new_std_series[itr_ec_fwd, 1] - lowest_point_edge) >
                 config['LAPDetection']['EdgCrcTrncMed'] * med_val_edge) |\
                (new_std_series[itr_ec_fwd, 11] < -config['LAPDetection']['PerStdTrnc']) &\
                ((new_std_series[itr_ec_fwd, 1] - lowest_point_edge) >
                 config['LAPDetection']['EdgCrcTrncMed'] * med_val_edge):

            new_std_series[itr_ec_fwd, 12] = 1
        else:
            break

    return new_std_series


def backward_statistical(dtp_chk_trnc, new_std_series, lowest_point_edge, med_val_edge, config):
    """ bakcward statistical"""
    for itr_ec_bkwd in range((np.shape(new_std_series)[0] - 1), (np.shape(new_std_series)[0] - dtp_chk_trnc) - 1, -1):
        if (new_std_series[itr_ec_bkwd, 11] > config['LAPDetection']['PerStdTrnc']) &\
                ((new_std_series[itr_ec_bkwd, 1] - lowest_point_edge) >
                 config['LAPDetection']['EdgCrcTrncMed'] * med_val_edge) | \
                (new_std_series[itr_ec_bkwd, 11] < -config['LAPDetection']['PerStdTrnc']) &\
                ((new_std_series[itr_ec_bkwd, 1] - lowest_point_edge) >
                 config['LAPDetection']['EdgCrcTrncMed'] * med_val_edge):

            new_std_series[itr_ec_bkwd, 12] = 1
        else:
            break

    return new_std_series


def forward_median(new_std_series_2, med_val, config):
    """ forward median """
    for itr_mc_fwd in range(int(np.shape(new_std_series_2)[0] / 2) - 1):
        if new_std_series_2[itr_mc_fwd, 1] > config['LAPDetection']['EdgCrcTrncMed'] * med_val:
            new_std_series_2[itr_mc_fwd, 12] = 1
        else:
            break

    return new_std_series_2


def backward_median(new_std_series_2, med_val, config):
    """ backward median """
    for itr_mc_bkwd in range(int(np.shape(new_std_series_2)[0]) - 1, int(np.shape(new_std_series_2)[0] / 2) + 1, -1):
        if new_std_series_2[itr_mc_bkwd, 1] > config['LAPDetection']['EdgCrcTrncMed'] * med_val:
            new_std_series_2[itr_mc_bkwd, 12] = 1
        else:
            break

    return new_std_series_2


def check_truncated_laps(this_both_trnctd_lap, chk_extrctd_in, crrctd_ext_pnts, num_d_pnts_insd, config):
    """ check truncated laps """
    for trncindx in range(np.shape(this_both_trnctd_lap)[0]):
        extrctd_in = crrctd_ext_pnts[((this_both_trnctd_lap[trncindx, 0] -
                                       (num_d_pnts_insd * config['samplingRate'])) < crrctd_ext_pnts[:, 0]) &
                                     ((this_both_trnctd_lap[trncindx, 0] + (num_d_pnts_insd * config['samplingRate'])) >
                                      crrctd_ext_pnts[:, 0]), :]
        chk_extrctd_in = np.vstack((chk_extrctd_in, extrctd_in))

    return chk_extrctd_in


def idx_truncated_laps(trnctd_laps_2, data_lines_in_lp_2):
    """ idx of truncated laps """
    for rm_tr_idx in range(np.shape(trnctd_laps_2)[0]):
        indx = np.where(data_lines_in_lp_2[:, 1] == trnctd_laps_2[rm_tr_idx, 0])
        data_lines_in_lp_2[indx[0][0], 0] = 0

    return data_lines_in_lp_2


def edge_correction_laps(ref_detection, config, dllps, logger_base):
    """
     This function is used for correction of edges both sides for 1.5 hour if
    1. Combine points in same direction
    2. Apply statistical edge correction
    3. Apply correction to points with transition value greater than 1.5 *
    median transitions value
    """
    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("edge_correction_laps")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    data_lies_in_lap = np.copy(dllps)

    diff_x = np.diff(np.hstack((0, data_lies_in_lap[:, 0], 0)))
    start_idx = np.where(diff_x > 0)
    end_idx = np.where(diff_x < 0)
    laps_start_end_pnts = np.hstack((start_idx[0][:, np.newaxis], end_idx[0][:, np.newaxis]))

    input_data = np.copy(ref_detection['input_data'])
    truncated_laps = np.zeros((1, 4))
    comb_crrctd_pnts = np.zeros((1, 7))

    # Following loop is to iterate over all LAPs in which each LAP
    # separately checks for edge correction in each LAP
    for i in range(laps_start_end_pnts.shape[0]):
        lap_data_vals = input_data[(laps_start_end_pnts[i, 0]):(laps_start_end_pnts[i, 1]),
                                   [Cgbdisagg.INPUT_EPOCH_IDX, Cgbdisagg.INPUT_CONSUMPTION_IDX]]
        lap_data_vals = lap_data_vals[~np.isnan(lap_data_vals[:, 1]), :]
        if len(lap_data_vals) <= 1:
            continue
        lap_data_vals = np.hstack((lap_data_vals, np.hstack((np.diff(lap_data_vals[:, 1]), 0))[:, np.newaxis]))
        lap_data_vals = np.hstack((lap_data_vals, np.sign(lap_data_vals[:, 2])[:, np.newaxis]))
        lap_data_vals = np.hstack((lap_data_vals, np.zeros((np.shape(lap_data_vals)[0], 1))))

        # To combine data points which are in similar direction in forward direction
        lap_data_vals = forward_sign(lap_data_vals, config)

        # To combine data points which are in similar direction
        # check only upto 1.5 hours equivalent data points from backwards
        lap_data_vals = backward_sign(lap_data_vals, config)

        stdv_list = np.zeros((np.shape(lap_data_vals)[0], 1))
        med_val = np.nanmedian(lap_data_vals[:, 1])
        med_val_edge = np.nanmedian(lap_data_vals[:, 1] - np.nanmin(lap_data_vals[:, 1]))
        lowest_point_edge = np.nanmin(lap_data_vals[:, 1])

        # check for transition value is > 1.5 * median value - forward
        stdev_list2 = forward_transition(lap_data_vals, stdv_list, med_val, config)

        # check only upto 1.5 hours equivalent data points backwards
        # check for transition value is > 1.5 * median value - backward
        stdev_list2 = backward_transition(lap_data_vals, med_val, stdev_list2, config)

        lap_data_vals_2 = np.hstack((lap_data_vals, stdv_list, stdev_list2))
        lap_data_vals_3 = lap_data_vals_2[lap_data_vals_2[:, 4] == 0, :]
        crrctd_ext_pnts = lap_data_vals_2[lap_data_vals_2[:, 4] == 1, :]
        comb_crrctd_pnts = np.vstack((comb_crrctd_pnts, crrctd_ext_pnts))
        d_stdv_list = np.hstack((np.diff(lap_data_vals_3[:, 5]), 0))[:, np.newaxis]

        stdvlist2_upsd = np.flipud(lap_data_vals_3[:, 6])
        d_stdvlist2 = np.hstack((np.diff(stdvlist2_upsd), 0))[:, np.newaxis]
        d_stdvlist2_btnrml = np.flipud(d_stdvlist2)
        lap_data_vals_4 = np.hstack((lap_data_vals_3, d_stdv_list, d_stdvlist2_btnrml,
                                     np.zeros((lap_data_vals_3.shape[0], 2))))

        # Combine forward and backward columns
        lap_data_vals_4 = combine_for_back(lap_data_vals_4)

        new_std_series = np.copy(lap_data_vals_4)
        series_div = np.zeros((new_std_series.shape[0], 1))
        for srsdividx in range(np.shape(series_div)[0]):
            if (new_std_series[srsdividx, 10] == 0) | (new_std_series[srsdividx, 9] == 0):
                series_div[srsdividx, 0] = 0
            else:
                series_div[srsdividx, 0] = new_std_series[srsdividx, 10] / new_std_series[srsdividx, 9]

        new_std_series = np.hstack((new_std_series, series_div, np.zeros((new_std_series.shape[0], 1))))
        dtp_chk_trnc = int(Cgbdisagg.SEC_IN_HOUR * 1.5 / config['samplingRate'])

        # Truncate points if % stdev greater than threshold value
        # statistical edge correction - forward
        new_std_series = forward_statistical(dtp_chk_trnc, new_std_series, lowest_point_edge, med_val_edge, config)

        # statistical edge correction - backward
        new_std_series = backward_statistical(dtp_chk_trnc, new_std_series, lowest_point_edge, med_val_edge, config)

        new_std_series_2 = np.copy(new_std_series[new_std_series[:, 12] != 1, :])
        # Correction of transition greater than 1.5 * Median transition-
        # forward 1.5 hours equivalent
        new_std_series_2 = forward_median(new_std_series_2, med_val, config)

        # Correction of transition greater than 1.5 * Median transition-
        # backward 1.5 hours equivalent
        new_std_series_2 = backward_median(new_std_series_2, med_val, config)

        this_trnctd_lap = new_std_series[new_std_series[:, 12] == 1, :]
        this_scnd_trnctd_lap = new_std_series_2[new_std_series_2[:, 12] == 1, :]
        this_trnctd_lap = this_trnctd_lap[:, :2]
        this_scnd_trnctd_lap = this_scnd_trnctd_lap[:, :2]
        this_trnctd_lap = np.hstack((this_trnctd_lap, np.full((np.shape(this_trnctd_lap)[0], 1), 1)))
        this_scnd_trnctd_lap = np.hstack((this_scnd_trnctd_lap, np.full((np.shape(this_scnd_trnctd_lap)[0], 1), 2)))
        this_both_trnctd_lap = np.vstack((this_trnctd_lap, this_scnd_trnctd_lap))

        num_d_pnts_insd = int((Cgbdisagg.SEC_IN_HOUR * 1.5 / config['samplingRate']) - 1)

        chk_extrctd_in = np.zeros((1, 7))

        # Check for truncated LAPs
        if np.shape(this_both_trnctd_lap)[0] > 0:
            chk_extrctd_in = check_truncated_laps(this_both_trnctd_lap, chk_extrctd_in, crrctd_ext_pnts,
                                                  num_d_pnts_insd, config)

        if np.shape(chk_extrctd_in)[0] > 0:
            chk_extrctd_in = chk_extrctd_in[1:, :2]
        chk_extrctd_in = np.hstack((chk_extrctd_in, np.full((np.shape(chk_extrctd_in)[0], 1), 3)))

        all_trnctd_laps = np.vstack((this_trnctd_lap, this_scnd_trnctd_lap, chk_extrctd_in))
        all_trnctd_laps = np.hstack((all_trnctd_laps, np.zeros((np.shape(all_trnctd_laps)[0], 1))))
        all_trnctd_laps[:, 3] = i

        if np.shape(all_trnctd_laps)[0] > 0:
            truncated_laps = np.vstack((truncated_laps, all_trnctd_laps))

    trnctd_laps_2 = truncated_laps[1:, :]
    data_lines_in_lp_2 = np.copy(data_lies_in_lap)

    logger.info('Number of Truncated LAPs: {} |'.format(trnctd_laps_2.shape[0]))

    # Find index of truncated LAPs
    if np.shape(trnctd_laps_2)[0] > 0:
        data_lines_in_lp_2 = idx_truncated_laps(trnctd_laps_2, data_lines_in_lp_2)

    return data_lines_in_lp_2, trnctd_laps_2


def std_median(seriesval, mdin):
    """
    Customized function for calculating standard deviation w.r.to median
    rather than regular mean
    """
    calc_std_wmd = np.multiply(seriesval - mdin, seriesval - mdin)
    stdval_med = np.sqrt(np.nanmean(calc_std_wmd))
    return stdval_med
