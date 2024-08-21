
"""
Author - Nisha Agarwal
Date - 10th Mar 2021
Calculate twh ts level confidence and potential values
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.raw_energy_itemization.utils import find_overlap_days
from python3.itemization.aer.raw_energy_itemization.utils import postprocess_conf_arr

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.appliance_potential.config.get_pot_conf import get_pot_conf


def get_twh_potential(app_index, item_input_object, item_output_object, sampling_rate, logger_pass):

    """
    Calculate TWH confidence and potential values

    Parameters:
        app_index                   (int)           : Index of app in the appliance list
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        sampling_rate               (int)           : sampling rate
        logger_pass                 (dict)          : Contains the logger and the logging dictionary to be passed on

    Returns:
        item_output_object        (dict)          : updated Dict containing all hybrid outputs
    """

    t_start = datetime.now()

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_twh_potential')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    # Fetching required inputs
    twh_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    # Initializing default values

    twh_confidence = np.ones(original_input_data.shape)
    twh_potential = np.zeros(original_input_data.shape)

    config = get_pot_conf().get('wh')

    disagg_conf = config.get('disagg_conf')
    min_hyrbid_conf = config.get('min_hyrbid_conf')
    potential_score_offset = config.get('potential_score_offset')
    twh_conf_offset = config.get('twh_conf_offset')
    twh_perenial_score_offset = config.get('twh_perenial_score_offset')
    twh_disagg_conf_offset = config.get('twh_disagg_conf_offset')
    min_frac_for_high_twh_days_count = config.get('min_frac_for_high_twh_days_count')
    conf_offset_high_twh_days_count = config.get('conf_offset_high_twh_days_count')

    # No potential calculation since zero usage

    if np.all(twh_disagg == 0):
        return item_output_object

    # Calculating parameters required for potential calculation

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')

    twh_tou = twh_disagg > 1

    # preparing score based on consistency in TWH usage

    twh_yearly_tou = np.sum(twh_tou, axis=0)
    twh_yearly_tou = np.fmin(1, twh_yearly_tou / np.sum(twh_tou, axis=0) + twh_perenial_score_offset)

    twh_confidence, twh_potential = \
        initialize_twh_potential(twh_confidence, twh_potential, twh_tou, app_index, item_input_object, item_output_object, samples_per_hour)


    twh_confidence[twh_disagg == 0] = 0
    twh_potential[twh_disagg == 0] = 0

    twh_confidence = np.fmax(0, np.fmin(1, twh_confidence + twh_conf_offset))

    twh_confidence = np.fmax(0, np.multiply(twh_confidence, twh_yearly_tou[None, :]))

    vali_twh_conf_present_flag = item_input_object.get("disagg_special_outputs") is not None and \
                                 item_input_object.get("disagg_special_outputs").get("timed_wh_confidence") is not None

    if vali_twh_conf_present_flag:

        disagg_confidence2 = copy.deepcopy(item_input_object.get("disagg_special_outputs").get("timed_wh_confidence"))

        disagg_confidence = item_input_object.get("disagg_special_outputs").get("timed_wh_confidence") + twh_disagg_conf_offset

        if disagg_confidence > disagg_conf[0]:
            twh_confidence = np.fmax(twh_confidence, disagg_confidence + potential_score_offset[0])
            twh_confidence = np.fmax(min_hyrbid_conf[0], twh_confidence)

        if disagg_confidence < disagg_conf[1]:
            twh_confidence = np.fmin(twh_confidence, disagg_confidence + potential_score_offset[1])
            twh_confidence = np.fmax(min_hyrbid_conf[1], twh_confidence)

        if disagg_confidence > disagg_conf[2]:

            twh_confidence = np.fmax(min_hyrbid_conf[2], twh_confidence+potential_score_offset[2])
            twh_potential = twh_disagg / np.percentile(twh_disagg[twh_disagg > 0], 90)

            for sample in range(samples_per_hour * Cgbdisagg.HRS_IN_DAY):
                twh_potential[:, sample][twh_potential[:, sample] > 0] = \
                    np.max(twh_potential[:, get_index_array(sample - samples_per_hour, sample + samples_per_hour,
                                                            samples_per_hour * Cgbdisagg.HRS_IN_DAY)],
                           axis=1)[twh_potential[:, sample] > 0]

            twh_potential[twh_disagg == 0] = 0

        twh_confidence = np.fmin(1, twh_confidence)

        logger.info("Timed Water heater confidence score is | %s", disagg_confidence2)

    # Final sanity checks

    twh_confidence, twh_potential = postprocess_conf_arr(twh_confidence, twh_potential, twh_disagg, vacation=np.zeros_like(twh_disagg).astype(bool))

    twh_confidence[twh_disagg > 0] = np.fmax(0.5, twh_confidence[twh_disagg > 0])

    higher_twh_days_present = ((np.sum(twh_confidence, axis=1) > 0).sum() / len(twh_confidence)) > min_frac_for_high_twh_days_count

    if higher_twh_days_present:
        twh_confidence = twh_confidence + conf_offset_high_twh_days_count

    twh_confidence[twh_disagg == 0] = 0
    twh_potential[twh_disagg == 0] = 0

    item_output_object["app_confidence"][app_index, :, :] = twh_confidence
    item_output_object["app_potential"][app_index, :, :] = twh_potential

    t_end = datetime.now()

    logger.debug("TWH potential calculation took | %.3f ", get_time_diff(t_start, t_end))

    return item_output_object


def initialize_twh_potential(twh_confidence, twh_potential, twh_tou, app_index, item_input_object, item_output_object, samples_per_hour):

    """
    Calculate TWH confidence and potential values

    Parameters:
        twh_confidence              (np.ndarray)    : ts level confidence
        twh_potential               (np.ndarray)    : ts level potential
        twh_tou                     (np.ndarray)    : ts level tou
        app_index                   (int)           : Index of app in the appliance list
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        samples_per_hour            (int)           : samples per hour

    Returns:
        twh_confidence              (np.ndarray)    : updated ts level confidence
        twh_potential               (np.ndarray)    : updated ts level potential
    """

    config = get_pot_conf().get("twh")
    twh_disagg = item_output_object.get("updated_output_data")[app_index, :, :]
    original_input_data = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    twh_capacity = np.max(twh_disagg.shape)

    no_of_days = len(original_input_data)

    twh_unique_days, days_indices, days_count = np.unique(twh_tou, return_counts=True, return_inverse=True, axis=0)

    for index, twh_run in enumerate(twh_unique_days):

        if np.all(np.logical_not(twh_run)):
            continue

        if days_count[index] < config.get("min_days_in_run_type"):

            # get list of days for the particular run type and nearby days

            days_index_array = get_index_array((np.where(days_indices == index)[0][0] - config.get("run_type_days_win")) % no_of_days,
                                               (np.where(days_indices == index)[0][-1] + config.get("run_type_days_win")) % no_of_days, no_of_days)

            twh_unique_days_in_win, indices_in_win, days_count_in_win = np.unique(twh_tou[days_index_array], return_counts=True, return_inverse=True, axis=0)

            # Find overlapping runs in the nearby days window

            overlap_day, bool1, overlap_count = find_overlap_days(twh_run, twh_unique_days_in_win, days_count_in_win, days_count[index], samples_per_hour)

            # If no overlapping run is found nearby or total number of days for the given run is less,
            # less confidence is assigned to the days with the given

            # Else the confidence is calculated using the variation within the given run

            if (not bool1) and overlap_count < config.get("min_days_in_run_type"):
                twh_confidence[days_indices == index] = twh_confidence[days_indices == index] - min(0.5, 1/days_count[index] * 10)
            else:

                overlapping_day_idx = \
                np.where(np.sum(twh_unique_days_in_win == overlap_day, axis=1) == len(original_input_data[0]))[0][0]
                run_conf = twh_confidence[days_indices == index, :]
                run_conf[:, np.abs(overlap_day.astype(int) - twh_run.astype(int)) > 0] = \
                    run_conf[:, np.abs(overlap_day.astype(int) - twh_run.astype(int)) > 0] - \
                    (np.abs(overlap_day.astype(int) - twh_run.astype(int)).sum() / samples_per_hour) / 2
                twh_confidence[days_indices == index, :] = run_conf

                twh_confidence[days_indices == index, :] = np.minimum(twh_confidence[days_indices == index, :],
                                                                      np.max(twh_tou[days_index_array][indices_in_win == overlapping_day_idx, :], axis=0))

        run_variation = twh_disagg[days_indices == index]
        run_variation = run_variation[run_variation > 0]
        mean_variation = np.mean(run_variation)

        # Similar potential is calculated for days for a given run in order to maintain consistency

        twh_confidence[days_indices == index] = twh_confidence[days_indices == index] - \
                                               np.divide(np.abs(twh_disagg[days_indices == index] - mean_variation), twh_disagg[days_indices == index])

        twh_potential[days_indices == index] = np.mean(twh_disagg[days_indices == index][twh_disagg[days_indices == index] > 0]) / twh_capacity

    return twh_confidence, twh_potential
