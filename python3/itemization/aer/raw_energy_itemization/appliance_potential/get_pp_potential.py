
"""
Author - Nisha Agarwal
Date - 10th Mar 2021
Calculate pp consumption from positive residual data
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.utils import find_overlap_days
from python3.itemization.aer.raw_energy_itemization.utils import postprocess_conf_arr

from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.functions.itemization_utils import resample_day_data

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_pp_potential(app_index, item_input_object, item_output_object, sampling_rate, logger_pass):

    """
    Calculate PP confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object          (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_pp_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs

    season_potential = item_output_object.get("season_potential")
    pp_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    season_potential = resample_day_data(season_potential, pp_disagg.shape[1])
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    # Initializing default values

    pp_confidence = np.ones(original_input_data.shape)
    pp_potential = np.zeros(original_input_data.shape)

    # No potential calculation since zero usage

    if np.all(pp_disagg == 0):
        return item_output_object

    # Calculating parameters required for potential calculation

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    config = get_pot_conf().get('pp')

    pp_confidence_score_offset = config.get('pp_confidence_score_offset')
    perenial_load_weight_in_pp_conf = config.get('perenial_load_weight_in_pp_conf')
    seasonal_potential_weight_in_pp_conf = config.get('seasonal_potential_weight_in_pp_conf')
    disagg_conf = config.get('disagg_conf')
    min_hyrbid_conf = config.get('min_hyrbid_conf')
    wind_size_for_season_wise_score = config.get('wind_size_for_season_wise_score')
    max_cons_perc = config.get('max_cons_perc')
    default_disagg_confidence = config.get('default_disagg_confidence')
    pp_conf_buckets = config.get('pp_conf_buckets')
    pp_perenial_score_offset = config.get('pp_perenial_score_offset')
    season_based_score_offset = config.get('season_based_score_offset')

    pp_tou = pp_disagg > 1

    pp_yearly_tou = np.sum(pp_tou, axis=0)
    pp_yearly_tou = np.fmin(1, pp_yearly_tou / np.sum(np.sum(pp_tou, axis=1) > 0) + pp_perenial_score_offset)

    pp_potential, pp_confidence = \
        initialize_pp_potential(pp_potential, pp_confidence, app_index, item_input_object, item_output_object, pp_tou, samples_per_hour)

    # Calculate season potential to be used in pp confidence calculation

    season_potential = np.fmin(1, 1 - season_potential - season_based_score_offset)

    season_potential[season_potential > np.percentile(season_potential, max_cons_perc)] = np.percentile(season_potential, max_cons_perc)

    days_in_month = Cgbdisagg.DAYS_IN_MONTH

    for index in range(0, wind_size_for_season_wise_score, max(1, len(season_potential)-days_in_month)):
        season_potential[index: index + days_in_month] = np.fmin(season_potential[index: index + days_in_month],
                                                                 np.percentile(season_potential[index: index + days_in_month], days_in_month))


    pp_confidence = np.fmin(1, np.fmax(pp_confidence, 0) + pp_confidence_score_offset)

    pp_confidence = perenial_load_weight_in_pp_conf*np.multiply(pp_confidence, pp_yearly_tou[None, :] + pp_confidence_score_offset) + \
                    seasonal_potential_weight_in_pp_conf*season_potential

    pp_confidence[pp_disagg == 0] = 0

    # updating PP confidence for each timeslot based on the distribution of confidence scores of the given time slot

    pp_confidence = np.fmax(pp_confidence, 0)

    pp_conf_present = np.any(pp_confidence, axis=0)

    pp_confidence[pp_confidence == 0] = np.nan
    pp_conf_hod_wise_perc = np.nanpercentile(pp_confidence, q=70, axis=0)
    pp_confidence = np.nan_to_num(pp_confidence)

    for sample in range(Cgbdisagg.HRS_IN_DAY * samples_per_hour):

        for val in pp_conf_buckets:
            if pp_conf_present[sample] and pp_conf_hod_wise_perc[sample] > val:
                pp_confidence[:, sample][np.logical_and(pp_confidence[:, sample] > 0, pp_confidence[:, sample] < val)] = val

    disagg_confidence, valid_score = fetch_disagg_conf_score(item_input_object, default_disagg_confidence)

    logger.info('PP Disagg confidence used during PP ts level potential calculation | %s', disagg_confidence)
    logger.info('Whether valid PP confidence is present  | %s', valid_score)

    if disagg_confidence > disagg_conf[0] and valid_score:
        pp_confidence = np.fmax(pp_confidence, disagg_confidence + 0.1)
        pp_confidence = np.fmax(min_hyrbid_conf[0], pp_confidence)
    if disagg_confidence < disagg_conf[1] and valid_score:
        pp_confidence = np.fmin(pp_confidence, disagg_confidence)
        pp_confidence = np.fmax(min_hyrbid_conf[1], pp_confidence)

    if disagg_confidence > disagg_conf[2] and valid_score:
        pp_confidence = np.fmax(min_hyrbid_conf[2], pp_confidence)
        pp_potential = pp_disagg / np.percentile(pp_disagg[pp_disagg > 0], max_cons_perc)

        for sample in range(samples_per_hour * Cgbdisagg.HRS_IN_DAY):
            pp_potential[:, sample][pp_potential[:, sample] > 0] = \
                np.max(pp_potential[:, get_index_array(sample-samples_per_hour, sample+samples_per_hour,
                                                       samples_per_hour * Cgbdisagg.HRS_IN_DAY)], axis=1)[pp_potential[:, sample] > 0]

        pp_potential[pp_disagg == 0] = 0

    pp_confidence = np.fmin(1, pp_confidence)

    item_input_object['disagg_special_outputs']['pp_hybrid_confidence'] = disagg_confidence * 100

    logger.info("PP hybrid confidence score is | %s", disagg_confidence)

    # Final sanity checks

    pp_confidence, pp_potential = postprocess_conf_arr(pp_confidence, pp_potential, pp_disagg, vacation=np.zeros(pp_disagg.shape).astype(bool))

    pp_confidence[pp_disagg == 0] = 0
    pp_potential[pp_disagg == 0] = 0

    pp_potential = pp_potential / np.max(pp_potential)

    item_output_object = update_pp_hsm(item_input_object, item_output_object, pp_disagg, disagg_confidence)

    # Heatmap dumping section

    item_output_object["app_confidence"][app_index, :, :] = pp_confidence
    item_output_object["app_potential"][app_index, :, :] = pp_potential

    t_end = datetime.now()

    logger.debug("PP potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def initialize_pp_potential(pp_potential, pp_confidence, app_index, item_input_object, item_output_object, pp_tou, samples_per_hour):

    """
    Calculate PP confidence and potential values based on distribution of consumption of each PP run

    Parameters:
        pp_potential                (np.ndarray)    : pp ts level potential
        pp_confidence               (np.ndarray)    : pp ts level confidence
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        pp_tou                      (np.ndarray)    : PP ts level tou
        samples_per_hour            (int)           : samples per hour

    Returns:
        pp_potential                (np.ndarray)    : updated pp ts level potential
        pp_confidence               (np.ndarray)    : updated pp ts level confidence
    """

    pp_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    config = get_pot_conf().get("pp")

    pp_capacity = np.max(pp_disagg)

    no_of_days = len(original_input_data)

    pp_unique_days, days_indices, days_count = np.unique(pp_tou, return_counts=True, return_inverse=True, axis=0)

    for index, pp_run in enumerate(pp_unique_days):

        if np.all(np.logical_not(pp_run)):
            continue

        if days_count[index] < config.get("min_days_in_run_type"):

            # get list of days for the particular run type and nearby days

            days_index_array = get_index_array((np.where(days_indices == index)[0][0] - config.get("run_type_days_win")) % no_of_days,
                                               (np.where(days_indices == index)[0][-1] + config.get("run_type_days_win")) % no_of_days, no_of_days)

            pp_unique_days_in_win, indices_in_win, days_count_in_win = np.unique(pp_tou[days_index_array], return_counts=True, return_inverse=True, axis=0)

            # Find overlapping runs in the nearby days window

            overlap_day, overlap_day_bool, overlap_count = \
                find_overlap_days(pp_run, pp_unique_days_in_win, days_count_in_win, days_count[index], samples_per_hour)

            # If no overlapping run is found nearby or total number of days for the given run is less,
            # less confidence is assigned to the days with the given

            # Else the confidence is calculated using the variation within the given run

            if (not overlap_day_bool) and overlap_count < config.get("min_days_in_run_type"):
                pp_confidence[days_indices == index] = pp_confidence[days_indices == index] - min(1, 1/days_count[index] * 10)
            else:
                overlapping_day_idx = \
                    np.where(np.sum(pp_unique_days_in_win == overlap_day, axis=1) == len(original_input_data[0]))[0][0]
                run_conf = pp_confidence[days_indices == index, :]
                run_conf[:, np.abs(overlap_day.astype(int) - pp_run.astype(int)) > 0] = \
                    run_conf[:, np.abs(overlap_day.astype(int) - pp_run.astype(int)) > 0] - \
                    (np.abs(overlap_day.astype(int) - pp_run.astype(int)).sum() / samples_per_hour) / 2

                pp_confidence[days_indices == index, :] = run_conf

                pp_confidence[days_indices == index, :] = np.minimum(pp_confidence[days_indices == index, :],
                                                                     np.max(pp_tou[days_index_array][indices_in_win == overlapping_day_idx, :], axis=0))

        run_variation = pp_disagg[days_indices == index]
        run_variation = run_variation[run_variation > 0]
        mean_variation = np.mean(run_variation)

        # Similar potential is calculated for days for a given run in order to maintain consistency

        pp_confidence[days_indices == index] = pp_confidence[days_indices == index] - \
                                               np.divide(np.abs(pp_disagg[days_indices == index] - mean_variation), pp_disagg[days_indices == index])

        pp_potential[days_indices == index] = np.mean(pp_disagg[days_indices == index][pp_disagg[days_indices == index] > 0]) / pp_capacity

    return pp_potential, pp_confidence


def fetch_disagg_conf_score(item_input_object, default_disagg_confidence):

    """
    fetch PP detection confidence

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        default_disagg_confidence   (float)         : default pp detection conf

    Returns:
        disagg_confidence           (np.ndarray)    : hybrid pp confidence score
        valid_score                 (int)           : flag that denotes whether pp hsm conf score can be used further
    """

    valid_pp_hsm = item_input_object.get("item_input_params").get('valid_pp_hsm')

    conf = 0

    valid_score = 0

    disagg_confidence = default_disagg_confidence

    # fetching confidence score from HSM in scenarios where PP HSM is available

    if valid_pp_hsm and item_input_object.get("item_input_params").get('pp_hsm') is not None:
        pp_hsm = item_input_object.get("item_input_params").get('pp_hsm')

        if pp_hsm.get('item_conf') is None:
            # hybrid pp detection conf is absent
            conf = 0
        elif isinstance(pp_hsm.get('item_conf'), list):
            # hybrid pp detection conf is present
            conf = pp_hsm.get('item_conf')[0]
        else:
            conf = pp_hsm.get('item_conf')

    if conf > 0:
        disagg_confidence = conf
        valid_score = 1

    # fetching confidence score from disagg module

    elif item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None:
        disagg_confidence = (item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100)
        valid_score = 1

    return disagg_confidence, valid_score


def update_pp_hsm(item_input_object, item_output_object, pp_disagg, disagg_confidence):

    """
    Update PP hsm

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        pp_disagg                   (np.ndarray)    : pp disagg output
        disagg_confidence           (float)         : pp detection confidence

    Returns:
        item_output_object          (dict)          : Dict containing all hybrid outputs
    """

    created_hsm = dict({
        'item_tou': np.zeros(len(pp_disagg[0])),
        'item_hld': 0,
        'item_conf': 0,
        'item_amp': 0,
        'final_item_amp': 0
    })

    samples = int(pp_disagg.shape[1]/Cgbdisagg.HRS_IN_DAY)

    amplitude_perc_thres = 90

    # updating PP HSM itemization attributes
    # these include - PP tou, amplitude, hld and confidence

    if not np.sum(pp_disagg) == 0:
        created_hsm['item_tou'] = np.sum(pp_disagg, axis=0) > 0
        created_hsm['item_amp'] = np.percentile(pp_disagg[pp_disagg > 0], amplitude_perc_thres) * samples
        created_hsm['final_item_amp'] = np.percentile(pp_disagg[pp_disagg > 0], amplitude_perc_thres) * samples
        created_hsm['item_hld'] = 1
        created_hsm['item_conf'] = disagg_confidence

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    if post_hsm_flag and item_output_object.get('created_hsm').get('pp') is None:
        item_output_object['created_hsm']['pp'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    pp_hsm_key_present =  (item_output_object.get('created_hsm') is not None) and \
        (item_output_object.get('created_hsm').get('pp') is not None) and \
        (item_output_object.get('created_hsm').get('pp').get('attributes') is not None) and \
        (not ((item_output_object.get('created_hsm').get('pp').get('attributes') is not None) and
              (item_output_object.get('created_hsm').get('pp').get('attributes').get('item_hld') == 1)))

    if post_hsm_flag and pp_hsm_key_present:
        item_output_object['created_hsm']['pp']['attributes'].update(created_hsm)

    return item_output_object
