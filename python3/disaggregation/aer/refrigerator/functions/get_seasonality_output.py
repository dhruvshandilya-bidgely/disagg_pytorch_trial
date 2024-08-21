"""
Date Created - 13 Nove 2018
Author name - Pratap
Seasonality function
"""

import logging
import numpy as np
from scipy import interpolate

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.time_utils import unix2mthdateint, unix2yeardateint


def one_est(avg_temp_mnth, estmtd_mnth):
    """ Monthly estimates from One available estimate  """
    if ~np.isnan(estmtd_mnth):
        norm_total_est = avg_temp_mnth[avg_temp_mnth[:, 0] == estmtd_mnth, 4] / \
                         avg_temp_mnth[avg_temp_mnth[:, 0] == estmtd_mnth, 8]
        month_est = norm_total_est * avg_temp_mnth[:, 8]
        month_est = month_est[:, np.newaxis]
    else:
        month_est = np.empty((np.shape(avg_temp_mnth)[0], 1))
        month_est[:] = np.nan

    if np.ndim(month_est) == 1:
        month_est = month_est[:, np.newaxis]
    return month_est


def two_est(avg_temp_mnth, first_est, first_est_mnth, second_est, second_est_mnth, estmtd_mnth, combo):
    """
     Monthly estimates from two available estimates
     input - 2 estimates with its month
     output - Monthly estimates
    """
    if ((first_est / second_est) > 0.9) & ((first_est / second_est) < 1.2):
        xone = np.asscalar(avg_temp_mnth[avg_temp_mnth[:, 0] == first_est_mnth, 1])
        xtwo = np.asscalar(avg_temp_mnth[avg_temp_mnth[:, 0] == second_est_mnth, 1])
        yone = first_est
        ytwo = second_est
        aone = (ytwo - yone) / (xtwo - xone)
        azero = yone - (aone * xone)
        month_est = azero + aone * avg_temp_mnth[:, 1]
        if combo == 'iw':
            if first_est > second_est:
                month_est[month_est > (1.2 * first_est)] = (1.2 * first_est)
            else:
                month_est[month_est < (0.9 * second_est)] = (0.9 * second_est)
        elif combo == 'si':
            if first_est > second_est:
                month_est[month_est < (0.8 * second_est)] = (0.8 * second_est)
            else:
                month_est[month_est > (1.1 * first_est)] = (1.1 * first_est)
    else:
        month_est = one_est(avg_temp_mnth, estmtd_mnth)

    if (np.ndim(month_est) == 1):
        month_est = month_est[:, np.newaxis]

    # adding perturbation for capped values
    min_perturbation = -0.05
    max_perturbation = 0.05
    perturbation = min_perturbation + (max_perturbation - min_perturbation) * np.random.random(
        (np.shape(month_est)[0], 1))
    month_est = month_est + perturbation
    return month_est


def three_est(avg_temp_mnth, first_est, first_est_mnth, second_est, second_est_mnth, third_est, third_est_mnth):
    """
    Monthly estimates from threee seasonal available estimates
    Inputs - 3 estimates
    Outputs - Monthly estimates
    """
    xone = np.asscalar(avg_temp_mnth[avg_temp_mnth[:, 0] == first_est_mnth, 1])
    xtwo = np.asscalar(avg_temp_mnth[avg_temp_mnth[:, 0] == second_est_mnth, 1])
    xthree = np.asscalar(avg_temp_mnth[avg_temp_mnth[:, 0] == third_est_mnth, 1])
    yone = first_est
    ytwo = second_est
    ythree = third_est
    x = np.array([xone, xtwo, xthree])
    y = np.array([yone, ytwo, ythree])

    f = interpolate.interp1d(x, y)
    month_est = f(avg_temp_mnth[:, 1])
    if (np.ndim(month_est) == 1):
        month_est = month_est[:, np.newaxis]

    return month_est


def nan_helper(y):
    """ nan helper function used before interpolation function"""
    return np.isnan(y), lambda z: z.nonzero()[0]


def find_shoulder_months(avg_temp_mnth, itr_estimated_ref, itr_estmtd_mnth, min_month_idx, max_month_idx):
    """ find shoulder months """
    if ~np.isnan(itr_estimated_ref) & ~np.isnan(itr_estmtd_mnth):
        indx_intr = np.where(avg_temp_mnth[:, 0] == itr_estmtd_mnth)[0][0]
        if (indx_intr != max_month_idx) & (indx_intr != min_month_idx):
            avg_temp_mnth[indx_intr, 7] = itr_estimated_ref
        else:
            itr_temp_val = np.copy(avg_temp_mnth[indx_intr, 1])
            allowed_temp_array = np.copy(avg_temp_mnth[:, 1])
            allowed_temp_array[min_month_idx] = np.nan
            allowed_temp_array[max_month_idx] = np.nan
            indx_intr = np.nanargmin(np.abs(allowed_temp_array - itr_temp_val))
            avg_temp_mnth[indx_intr, 7] = itr_estimated_ref
            itr_estmtd_mnth = avg_temp_mnth[indx_intr, 0]

    return avg_temp_mnth, itr_estmtd_mnth


def seasonal_estimate(allchk, season_chk, avg_temp_mnth, season_estimated_ref, season_estmtd_mnth):
    """

    :param allchk:
    :param season_chk:
    :param avg_temp_mnth:
    :param season_estimated_ref:
    :param season_estmtd_mnth:
    :return:
    """
    smrchk = season_chk.get("smrchk")
    wtrchk = season_chk.get("wtrchk")
    itrchk = season_chk.get("itrchk")

    smr_estimated_ref = season_estimated_ref.get("smr_estimated_ref")
    wtr_estimated_ref = season_estimated_ref.get("wtr_estimated_ref")
    itr_estimated_ref = season_estimated_ref.get("itr_estimated_ref")

    estmtd_mnth = season_estmtd_mnth.get("estmtd_mnth")
    smr_estmtd_mnth = season_estmtd_mnth.get("smr_estmtd_mnth")
    wtr_estmtd_mnth = season_estmtd_mnth.get("wtr_estmtd_mnth")
    itr_estmtd_mnth = season_estmtd_mnth.get("itr_estmtd_mnth")

    if (allchk == 0):
        s2iratio = smr_estimated_ref / itr_estimated_ref
        i2wratio = itr_estimated_ref / wtr_estimated_ref
        s2wratio = smr_estimated_ref / wtr_estimated_ref
        # Following snippet check the validity of ratios for estimates
        # between Summer,Intermediate & Winter
        if (((s2iratio > 0.9) & (s2iratio < 1.2)) & ((i2wratio > 0.9) & (i2wratio < 1.2))):
            avg_temp_mnth = np.hstack((avg_temp_mnth, three_est(avg_temp_mnth, smr_estimated_ref, smr_estmtd_mnth,
                                                                wtr_estimated_ref, wtr_estmtd_mnth, itr_estimated_ref,
                                                                itr_estmtd_mnth)))

        elif ((s2iratio > 0.9) & (s2iratio < 1.2)):
            avg_temp_mnth = np.hstack((avg_temp_mnth, two_est(avg_temp_mnth, smr_estimated_ref, smr_estmtd_mnth,
                                                              itr_estimated_ref, itr_estmtd_mnth, estmtd_mnth, 'si')))

        elif ((i2wratio > 0.9) & (i2wratio < 1.2)):
            avg_temp_mnth = np.hstack((avg_temp_mnth, two_est(avg_temp_mnth, itr_estimated_ref, itr_estmtd_mnth,
                                                              wtr_estimated_ref, wtr_estmtd_mnth, estmtd_mnth, 'iw')))

        elif ((s2wratio > 0.9) & (s2wratio < 1.2)):
            avg_temp_mnth = np.hstack((avg_temp_mnth, two_est(avg_temp_mnth, smr_estimated_ref, smr_estmtd_mnth,
                                                              wtr_estimated_ref, wtr_estmtd_mnth, estmtd_mnth, 'sw')))
        else:
            avg_temp_mnth = np.hstack((avg_temp_mnth, one_est(avg_temp_mnth, estmtd_mnth)))
    elif (allchk == 1):
        if (smrchk == 1):
            first_est_ref = itr_estimated_ref
            first_est_mnth = itr_estmtd_mnth
            second_est_ref = wtr_estimated_ref
            second_est_mnth = wtr_estmtd_mnth
            combi = 'iw'
        elif (wtrchk == 1):
            first_est_ref = smr_estimated_ref
            first_est_mnth = smr_estmtd_mnth
            second_est_ref = itr_estimated_ref
            second_est_mnth = itr_estmtd_mnth
            combi = 'si'
        elif (itrchk == 1):
            first_est_ref = smr_estimated_ref
            first_est_mnth = smr_estmtd_mnth
            second_est_ref = wtr_estimated_ref
            second_est_mnth = wtr_estmtd_mnth
            combi = 'sw'

        avg_temp_mnth = np.hstack((avg_temp_mnth, two_est(avg_temp_mnth, first_est_ref, first_est_mnth,
                                                          second_est_ref, second_est_mnth, estmtd_mnth,
                                                          combi)))
    elif (allchk >= 2):
        avg_temp_mnth = np.hstack((avg_temp_mnth, one_est(avg_temp_mnth, estmtd_mnth)))

    return avg_temp_mnth


def fill_temperature(avg_temp_mnth):
    """ fill existing temp """
    for act_tmp in range(np.shape(avg_temp_mnth)[0]):
        if (np.isnan(avg_temp_mnth[act_tmp, 1])):
            alltemps = avg_temp_mnth[avg_temp_mnth[:, 3] == avg_temp_mnth[act_tmp, 3], 1]
            avg_temp_mnth[act_tmp, 1] = np.nanmean(alltemps)

    return avg_temp_mnth


def extract_values_from_hsm(hsm_in):

    """Extract values from hsm using"""

    # Handle different forms of hsm accordingly

    coldest_month_temp = hsm_in['attributes']['coldestMonthTemp']
    second_hottest_month_temp = hsm_in['attributes']['secondHottestMonthTemp']

    if type(coldest_month_temp) == list:
        coldest_month_temp = coldest_month_temp[0]

    if type(second_hottest_month_temp) == list:
        second_hottest_month_temp = coldest_month_temp[0]

    return coldest_month_temp, second_hottest_month_temp


def get_seasonality_output(ref_detection, config, hist_mode, hsm_in, bypass_hsm, logger_base):
    """
    Calculate monthly estimates based on the available scenarios
    3-cases possible
    Case1: Only 1 estimate is available
    Case2: Two estimates available (Summer & Winter or Winter & Inter etc.)
    Case3: All 3 estimates available
    input - Estimates- Total, Summer, Winter & Intermediate,Monthly avg.
    temperatures
    """
    # Taking new logger base for this module
    logger_local = logger_base.get("logger").getChild("get_seasonality_output")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    logger.info('Processing seasonality output for ref |')

    input_data = ref_detection['input_data']
    months = np.unique(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX])[:, np.newaxis]
    avg_temp = np.zeros((1, 1))

    for mnth_ind in range(np.shape(months)[0]):
        partic_mnth_dta = np.nanmean(
            input_data[(input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == months[mnth_ind]), Cgbdisagg.INPUT_TEMPERATURE_IDX])
        avg_temp = np.vstack((avg_temp, partic_mnth_dta))

    avg_temp = avg_temp[1:, :]
    ref_hourly_output = np.hstack((input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX][:, np.newaxis],
                                   input_data[:, Cgbdisagg.INPUT_EPOCH_IDX][:, np.newaxis],
                                   0 * input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX][:, np.newaxis],
                                   input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX][:, np.newaxis]))
    estimated_ref = ref_detection['estimatedRef']
    avg_temp_mnth = np.hstack((months, avg_temp, np.zeros((np.shape(months)[0], 2))))

    for i in range(np.shape(avg_temp_mnth)[0]):
        avg_temp_mnth[i, 2] = unix2yeardateint(avg_temp_mnth[i, 0])
        avg_temp_mnth[i, 3] = unix2mthdateint(avg_temp_mnth[i, 0])

    # Replacing temp with existing month temps
    avg_temp_mnth = fill_temperature(avg_temp_mnth)

    # Interpolation of missing temp in between
    y = avg_temp_mnth[:, 1]
    nans, x = nan_helper(y)
    if np.shape(y)[0] > 1:
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        avg_temp_mnth[:, 1] = y
    avg_temp_mnth = np.hstack((avg_temp_mnth, np.zeros((np.shape(avg_temp_mnth)[0], 4))))
    # following section used to calcluate scaling based on average temp.
    sorted_idx = np.argsort(avg_temp_mnth[:, 1])
    # if config.MONTHLY_FLAG | np.isnan(config['hsm']['timestamp']):
    if hist_mode | bypass_hsm:
        coldest_month_temp = avg_temp_mnth[sorted_idx[0], 1]
        second_hottest_month_temp = avg_temp_mnth[sorted_idx[np.shape(avg_temp_mnth)[0] - 2], 1]
        hsm = dict({
            'timestamp': np.nan,
            'attributes': {
                'coldestMonthTemp': np.nan,
                'secondHottestMonthTemp': np.nan,
                'Ref_Energy_Per_DataPoint': np.nan
            }
        })

        hsm['timestamp'] = input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX]
        hsm['attributes']['coldestMonthTemp'] = coldest_month_temp
        hsm['attributes']['secondHottestMonthTemp'] = second_hottest_month_temp
        hsm['attributes']['Ref_Energy_Per_DataPoint'] = estimated_ref
        ref_detection['hsm'] = hsm
        # make sure to send hsm to api back
    else:
        coldest_month_temp, second_hottest_month_temp = extract_values_from_hsm(hsm_in)

    # smr = Summer, wtr = Winter, itr = Intermediate, Pratap wrote this code. He had this policy of removing vowels
    # from all variable names. This will be fixed in the refactoring.

    try:
        estmtd_mnth = ref_detection['estmtdMnth']
        avg_temp_mnth[avg_temp_mnth[:, 0] == estmtd_mnth, 4] = estimated_ref
        smr_estimated_ref = ref_detection['smrEstimatedRef']
        max_month_idx = np.argmax(avg_temp_mnth[:, 1])
        smr_estmtd_mnth = int(avg_temp_mnth[max_month_idx, 0])
        avg_temp_mnth[avg_temp_mnth[:, 0] == smr_estmtd_mnth, 5] = smr_estimated_ref

        wtr_estimated_ref = ref_detection['wtrEstimatedRef']
        min_month_idx = np.argmin(avg_temp_mnth[:, 1])
        wtr_estmtd_mnth = int(avg_temp_mnth[min_month_idx, 0])
        avg_temp_mnth[avg_temp_mnth[:, 0] == wtr_estmtd_mnth, 6] = wtr_estimated_ref

        itr_estimated_ref = ref_detection['itrEstimatedRef']
        itr_estmtd_mnth = ref_detection['itrEstmtdMnth']

        avg_temp_mnth, itr_estmtd_mnth = find_shoulder_months(avg_temp_mnth, itr_estimated_ref,
                                                              itr_estmtd_mnth, min_month_idx, max_month_idx)

        scaling = 1 + (config['Seasonality']['secondHottestMonthSeasonalityScaling'] - 1) * \
                  (avg_temp_mnth[:, 1] - coldest_month_temp) / (second_hottest_month_temp - coldest_month_temp)
        scaling[scaling < 1] = 1
        scaling[scaling > config['Seasonality']['upperBoundSeasonalityScaling']] = config['Seasonality'][
            'upperBoundSeasonalityScaling']
        avg_temp_mnth = np.hstack((avg_temp_mnth, scaling[:, np.newaxis]))
        smrchk = np.isnan(smr_estimated_ref) + 0
        wtrchk = np.isnan(wtr_estimated_ref) + 0
        itrchk = np.isnan(itr_estimated_ref) + 0
        allchk = smrchk + wtrchk + itrchk

        season_chk = {
            "smrchk": smrchk,
            "wtrchk": wtrchk,
            "itrchk": itrchk
        }

        season_estimated_ref = {
            "smr_estimated_ref": smr_estimated_ref,
            "wtr_estimated_ref": wtr_estimated_ref,
            "itr_estimated_ref": itr_estimated_ref
        }

        season_estmtd_mnth = {
            "estmtd_mnth": estmtd_mnth,
            "smr_estmtd_mnth": smr_estmtd_mnth,
            "wtr_estmtd_mnth": wtr_estmtd_mnth,
            "itr_estmtd_mnth": itr_estmtd_mnth
        }

        # Get average temp month and estimate for each season
        avg_temp_mnth = seasonal_estimate(allchk, season_chk, avg_temp_mnth, season_estimated_ref, season_estmtd_mnth)

        ssnlty_values_one = avg_temp_mnth[:, [0, 9]]
        ssnlty_values = np.copy(ssnlty_values_one)

        for mnth_idx_fill in range(np.shape(ssnlty_values_one)[0]):
            ref_hourly_output[(ref_hourly_output[:, 0] == ssnlty_values_one[mnth_idx_fill, 0]) &
                              (ref_hourly_output[:, 3] >= 2), 2] = ssnlty_values_one[mnth_idx_fill, 1]
            ssnlty_values[mnth_idx_fill, 1] = np.sum(
                ref_hourly_output[(ref_hourly_output[:, 0] == ssnlty_values_one[mnth_idx_fill, 0]) &
                                  (ref_hourly_output[:, 3] >= 2), 2])
    except (KeyError, IndexError) as e:
        ssnlty_values = np.hstack((months, np.zeros((np.shape(months)[0], 1))))
        ssnlty_values[:, 1] = np.nan
        ref_hourly_output = ref_detection['refHourlyOutput']

    ref_hourly_output = ref_hourly_output[:, :3]
    ref_detection['monthRef'] = ssnlty_values
    ref_detection['refHourlyOpSsntly'] = ref_hourly_output

    return ref_detection
