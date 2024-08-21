
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
Update ev consumption ranges using inference rules
"""

# Import python packages

import copy
import numpy as np
from numpy.random import RandomState

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.functions.itemization_utils import rolling_func

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def ev_l1_postprocessing(l1_tag, ev_freq, item_input_object, app_index, min_cons, mid_cons, max_cons, item_output_object, logger):

    """
    Fill EV l1_bool for missing days

    Parameters:
        l1_tag                      (int)        : ev charger type
        ev_freq                     (int)        : freq of ev charging
        item_input_object           (dict)       : Dict containing all hybrid inputs
        app_index                   (int)        : index mapping of ev
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_output_object          (dict)       : Dict containing all hybrid outputs
        logger                      (logger)     : logger object

    Returns:
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_output_object          (dict)       : Dict containing all hybrid outputs
    """

    ev_config = get_inf_config().get('ev')

    if item_input_object["item_input_params"]["backup_ev"]:
        return min_cons, mid_cons, max_cons, item_output_object

    input_data = item_output_object.get("inference_engine_dict").get("input_data")[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :]

    appliance_list = item_input_object.get("item_input_params").get("app_list")

    samples_per_hour = int(mid_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    # calculating freq of ev l1 usage

    ev_days_seq = find_seq((mid_cons > 0).flatten(), np.zeros(mid_cons.size), np.zeros(mid_cons.size))
    ev_days_seq = ev_days_seq[ev_days_seq[:, 0] == 0]
    ev_usage_freq = np.diff(ev_days_seq[:, 1]) / mid_cons.shape[1]

    if np.any(ev_usage_freq < Cgbdisagg.DAYS_IN_MONTH):
        ev_usage_freq = ev_usage_freq[ev_usage_freq < Cgbdisagg.DAYS_IN_MONTH]
        ev_usage_freq = max(0, int(np.ceil(np.median(ev_usage_freq))))
    else:
        ev_usage_freq = 1

    logger.info("EV usage frequency | %s", ev_usage_freq)

    post_process_bool = 1

    len_factor = ev_config.get('l1_len_factor')

    input_data_for_ev_addition = copy.deepcopy(input_data)

    # removing pp and wh output, since wh and pp shouldnt be picked in wh

    output_data = item_output_object.get("hybrid_input_data").get("output_data")

    # preparing consumption that can be added into ev l1 output

    pp_index = np.where(np.array(appliance_list) == 'pp')[0][0] + 1
    pp_disagg = output_data[pp_index]
    input_data_for_ev_addition = input_data_for_ev_addition - pp_disagg

    wh_index = np.where(np.array(appliance_list) == 'wh')[0][0] + 1
    wh_disagg = output_data[wh_index]
    input_data_for_ev_addition = input_data_for_ev_addition - wh_disagg

    if item_input_object.get("config").get('disagg_mode') == 'mtd':
        len_factor = ev_config.get('mtd_l1_len_factor')

    # if there is significant gap in ev l1 output, it is not extended in hybrid v2

    high_gaps_in_ev_l1_output =  \
        np.sum(mid_cons) > 0 and \
        (np.max((mid_cons[:, :5 * samples_per_hour] > 0).sum(axis=0)) < 25 and
         (np.where((mid_cons.sum(axis=1) > 0))[0][-1] - np.where((mid_cons.sum(axis=1) > 0))[0][0]) > 200)

    if high_gaps_in_ev_l1_output:
        post_process_bool = 0

    ev_hsm_charger_type = 0

    ev_hsm_present = \
        (item_input_object.get('created_hsm') is not None) and \
        (item_input_object.get('created_hsm').get('ev') is not None) and \
        (item_input_object.get('created_hsm').get('ev').get('attributes') is not None)  and \
         (item_input_object.get('created_hsm').get('ev').get('attributes').get("charger_type") is not None)

    # checking ev charger type from previous disagg runs,
    # if ev l1 is present, it is extended in current run as well

    if ev_hsm_present:
        ev_hsm_charger_type = item_input_object.get('created_hsm').get('ev').get('attributes').get("charger_type")

    if ev_hsm_charger_type != 1 and ev_hsm_charger_type != 2:
        ev_hsm_charger_type = (np.median(mid_cons[mid_cons > 0]) < ev_config.get('l1_box_thres') / samples_per_hour)

    # fill ev only if ev l1_bool type is detected

    postprocess_ev_l1_output = (post_process_bool and ev_hsm_charger_type == 1) or (l1_tag == 1 and ev_freq > 0)

    if postprocess_ev_l1_output:

        # chunk of days, where ev will be added

        logger.info("postprocessing to fill remaining ev boxes  | ")

        min_cons, mid_cons, max_cons, item_output_object = \
            fill_ev_l1_missing_days([l1_tag, ev_freq], item_input_object, input_data_for_ev_addition, app_index,
                                    min_cons, mid_cons, max_cons, item_output_object, len_factor, ev_usage_freq)

    return min_cons, mid_cons, max_cons, item_output_object


def fill_ev_l1_missing_days(l1_params, item_input_object, input_data, app_index,
                            min_cons, mid_cons, max_cons, item_output_object, len_factor, ev_usage_freq):

    """
    Fill EV l1_bool for missing days

    Parameters:
        item_input_object           (dict)       : Dict containing all hybrid inputs
        input_data                  (np.ndarray) : input data
        app_index                   (int)        : index mapping of ev
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_output_object          (dict)       : Dict containing all hybrid outputs
        len_factor                  (int)        : min frac of ev boxes len required compared to original length
        ev_usage_freq               (int)        : ev usage frequency

    Returns:
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_output_object          (dict)       : Dict containing all hybrid outputs
    """

    l1_tag = l1_params[0]
    l1_freq = l1_params[1]

    if np.sum(mid_cons) > 0:
        min_cons, mid_cons, max_cons, item_output_object = \
            add_hybrid_ev_l1_consumption_for_disagg_users(min_cons, mid_cons, max_cons, item_input_object,
                                                          item_output_object, input_data, app_index, len_factor, ev_usage_freq)

    # fetching EV hsm information inorder to case ev l1 consumption where ev is not added in current disagg run
    # but was detected as ev l1 in previous run

    else:
        min_cons, mid_cons, max_cons, item_output_object = \
            add_hybrid_ev_l1_based_on_last_run_info(min_cons, mid_cons, max_cons, item_input_object, item_output_object,
                                                    input_data, app_index, len_factor, l1_tag, l1_freq)

    return min_cons, mid_cons, max_cons, item_output_object


def add_hybrid_ev_l1_consumption_for_disagg_users(min_cons, mid_cons, max_cons, item_input_object, item_output_object,
                                                  input_data, app_index, len_factor, ev_usage_freq):
    """
    Fill EV l1_bool for missing days for users where l1 is detected from disagg

    Parameters:
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_input_object           (dict)       : Dict containing all hybrid inputs
        item_output_object          (dict)       : Dict containing all hybrid outputs
        input_data                  (np.ndarray) : input data
        app_index                   (int)        : index mapping of ev
        len_factor                  (int)        : min frac of ev boxes len required compared to original length
        ev_usage_freq               (int)        : ev usage frequency

    Returns:
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_output_object          (dict)       : Dict containing all hybrid outputs
    """

    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index, :, :]

    samples_per_hour = int(mid_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    timed_app_output = item_output_object.get("timed_app_dict").get("timed_output")

    if timed_app_output is None:
        timed_app_output = np.zeros_like(mid_cons)

    # calculating ev l1 usage parameters

    ev_config = get_inf_config(samples_per_hour).get('ev')

    ev_days = np.sum(mid_cons, axis=1) > 0

    ev_days_seq = find_seq(ev_days, np.zeros_like(ev_days), np.zeros_like(ev_days), overnight=0)

    ev_usage_seq = find_seq((mid_cons > 0).flatten(), np.zeros_like((mid_cons > 0).flatten()),
                            np.zeros_like((mid_cons > 0).flatten()))

    ev_avg_length = np.mean(ev_usage_seq[ev_usage_seq[:, 0] > 0, 3])

    ev_amp = np.median(mid_cons[mid_cons > 0])

    ev_tou = np.sum(mid_cons > 0, axis=0)

    if ev_avg_length < 2:
        return min_cons, mid_cons, max_cons, item_output_object

    start = np.argmax(rolling_func(ev_tou, max(1, int(ev_avg_length/2))))

    # if start time is during day time, ev l1 consumption will not be added

    day_time_ev_usage = int(start - int(ev_avg_length / 2)) in np.arange(4.5 * samples_per_hour, 17 * samples_per_hour)

    if day_time_ev_usage:
        return min_cons, mid_cons, max_cons, item_output_object

    ev_tou = np.zeros(len(mid_cons[0]))

    ev_tou[get_index_array(start - int(ev_avg_length / 2), start + int(ev_avg_length / 2), samples_per_hour * 24).astype(int)] = 1

    non_ev_l1_hours = ev_config.get('non_ev_l1_hours')

    ev_tou[non_ev_l1_hours] = 0

    recent_ev_flag_present = \
        (item_input_object.get('created_hsm') is not None) and \
        (item_input_object.get('created_hsm').get('ev') is not None) and \
        (item_input_object.get('created_hsm').get('ev').get('attributes') is not None) and \
        (item_input_object.get('created_hsm').get('ev').get('attributes').get("recent_ev") is not None) and \
        item_input_object.get('created_hsm').get('ev').get('attributes').get("recent_ev") > 0

    # extending EV l1 consumption for each chunk of already detected ev l1 consumption in true disagg
    for i in range(len(ev_days_seq)):

        probable_ev_points = np.zeros_like(mid_cons)

        seq_start = ev_days_seq[i, 1]
        seq_end = ev_days_seq[i, 2]

        larger_chunk_for_l1_boxes_missing = \
            ev_days_seq[i, 0] == 0 and ev_days_seq[i, 3] < max(ev_config.get('max_l1_missing_days'), 400) and ev_days_seq[i, 3] > 2

        if (not larger_chunk_for_l1_boxes_missing) or recent_ev_flag_present:
            continue

        # chunk of days, where ev will be added

        # ev l1 points are only added at points with certain consumption level
        # and close to hours of the day where original ev l1 was estimated

        probable_ev_points[input_data > ev_amp * 0.8] = 1
        probable_ev_points[:, np.logical_not(ev_tou)] = 0

        probable_ev_points[timed_app_output > 0] = 0

        probable_ev_points[np.sum(probable_ev_points, axis=1) < max(3 * samples_per_hour,
                                                                    min(ev_tou.sum() * len_factor,
                                                                        ev_avg_length * len_factor) * 0.9)] = 0
        probable_ev_points[probable_ev_points > 0] = input_data[probable_ev_points > 0]
        probable_ev_points = np.fmax(0, probable_ev_points)
        probable_ev_points = np.fmin(probable_ev_points, ev_amp)

        seed = RandomState(random_gen_config.seed_value)

        # maintaining original ev freq while adding extra consumption

        chunk_ev_freq = (np.sum(probable_ev_points[seq_start: seq_end].sum(axis=1) > 0)) / ev_days_seq[i, 3]

        chunk_ev_freq = 1 / chunk_ev_freq

        if (np.sum(probable_ev_points[seq_start: seq_end].sum(axis=1) > 0)) == 0:
            continue

        freq_of_ev_addition = int((1 - (chunk_ev_freq / (1 + ev_usage_freq))) * (ev_days_seq[i, 3]))

        l1_ev_addition_flag = not (item_input_object.get("config").get('disagg_mode') == 'mtd') and freq_of_ev_addition > 0

        # removing extra ev l1 box from hybrid, if freq is different from expected frequency

        if l1_ev_addition_flag:
            remove_days = seed.choice(np.arange(ev_days_seq[i, 3]), min(ev_days_seq[i, 3], freq_of_ev_addition),
                                      replace=False)
            probable_ev_points[remove_days + ev_days_seq[i, 1]] = 0

        if np.sum(probable_ev_points[seq_start: seq_end].sum(axis=1) > 0) < ev_days_seq[i, 3] * 0.1:
            continue

        mid_cons[seq_start: seq_end] = mid_cons[seq_start: seq_end] + probable_ev_points[seq_start: seq_end]
        min_cons[seq_start: seq_end] = min_cons[seq_start: seq_end] + probable_ev_points[seq_start: seq_end]
        max_cons[seq_start: seq_end] = max_cons[seq_start: seq_end] + probable_ev_points[seq_start: seq_end]

        app_conf[np.logical_and(mid_cons > 0, app_conf == 0)] = 1

        item_output_object["inference_engine_dict"]["appliance_conf"][app_index] = app_conf

    return min_cons, mid_cons, max_cons, item_output_object


def add_hybrid_ev_l1_based_on_last_run_info(min_cons, mid_cons, max_cons, item_input_object, item_output_object,
                                            input_data, app_index, len_factor, l1_tag, l1_freq):

    """
    Fill EV l1_bool for missing days

    Parameters:
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_input_object           (dict)       : Dict containing all hybrid inputs
        item_output_object          (dict)       : Dict containing all hybrid outputs
        input_data                  (np.ndarray) : input data
        app_index                   (int)        : index mapping of ev
        len_factor                  (int)        : min frac of ev boxes len required compared to original length
        l1_tag                      (int)        : ev charger type
        l1_freq                     (int)        : freq of ev charging based on hsm data
    Returns:
        min_cons                    (np.ndarray) : min ts level cons
        mid_cons                    (np.ndarray) : mid ts level cons
        max_cons                    (np.ndarray) : max ts level cons
        item_output_object          (dict)       : Dict containing all hybrid outputs
    """

    app_conf = item_output_object.get("inference_engine_dict").get("appliance_conf")[app_index, :, :]

    timed_app_output = item_output_object.get("timed_app_dict").get("timed_output")

    if timed_app_output is None:
        timed_app_output = np.zeros_like(mid_cons)

    # calculating ev l1 usage parameters

    samples_per_hour = int(mid_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    ev_amp, ev_tou, add_ev_l1 = fetch_ev_hsm_info(item_input_object, samples_per_hour)

    if l1_tag == 1 and l1_freq > 0 and (add_ev_l1):

        probable_ev_points = np.zeros_like(mid_cons)

        # chunk of days, where ev will be added

        probable_ev_points[input_data > (ev_amp / samples_per_hour) * 0.8] = 1
        probable_ev_points[:, np.logical_not(ev_tou)] = 0

        probable_ev_points[timed_app_output > 0] = 0

        # ev l1 points are only added at points with certain consumption level
        # and close to hours of the day where original ev l1 was estimated

        probable_ev_points[np.sum(probable_ev_points, axis=1) < (ev_tou.sum() * len_factor) * 0.9] = 0
        probable_ev_points[probable_ev_points > 0] = input_data[probable_ev_points > 0]
        probable_ev_points = np.fmax(0, probable_ev_points)
        probable_ev_points = np.fmin(probable_ev_points, (ev_amp / samples_per_hour))

        seed = RandomState(random_gen_config.seed_value)

        # maintaining original ev freq while adding extra consumption

        chunk_ev_freq = (np.sum(probable_ev_points[:].sum(axis=1) > 0)) / len(probable_ev_points)

        if (chunk_ev_freq == 0) or (np.sum(probable_ev_points[:].sum(axis=1) > 0) < (len(probable_ev_points) * 0.1)):
            return min_cons, mid_cons, max_cons, item_output_object

        chunk_ev_freq = 1 / chunk_ev_freq

        freq_of_ev_addition = int((1 - (chunk_ev_freq / (1 + max(l1_freq, 2)))) * (len(mid_cons)))

        cons_days = np.where(probable_ev_points[:].sum(axis=1) > 0)[0]

        l1_ev_addition_flag = not (item_input_object.get("config").get('disagg_mode') == 'mtd') and freq_of_ev_addition > 0

        # removing extra ev l1 box from hybrid, if freq is different from expected frequency

        if l1_ev_addition_flag:
            remove_days = seed.choice(np.arange(len(cons_days)), min(len(cons_days), freq_of_ev_addition), replace=False)
            probable_ev_points[remove_days + 0] = 0

        if np.sum(probable_ev_points[:].sum(axis=1) > 0) < (len(probable_ev_points) * 0.1):
            return min_cons, mid_cons, max_cons, item_output_object

        mid_cons[:] = mid_cons[:] + probable_ev_points[:]
        min_cons[:] = min_cons[:] + probable_ev_points[:]
        max_cons[:] = max_cons[:] + probable_ev_points[:]

        app_conf[np.logical_and(mid_cons > 0, app_conf == 0)] = 1

        item_output_object["inference_engine_dict"]["appliance_conf"][app_index] = app_conf

    return min_cons, mid_cons, max_cons, item_output_object


def fetch_ev_hsm_info(item_input_object, samples_per_hour):

    """
    Fetch EV HSM info

    Parameters:
        item_input_object           (dict)       : Dict containing all hybrid inputs
        samples_per_hour            (int)        : samples in an hour
    Returns:
        ev_amp                      (int)        : EV amp of last run
        ev_tou                      (np.ndarray) : EV HSM of last run
        add_ev_l1                   (int)        : flag that represents whether EV will be added based on last run
    """

    add_ev_l1 = 1
    ev_amp = 0
    ev_tou = np.zeros((samples_per_hour*Cgbdisagg.HRS_IN_DAY))

    ev_hsm_present = item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None

    if ev_hsm_present and item_input_object.get("item_input_params").get('ev_hsm').get('item_tou') is not None:
        ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_tou')

        ev_tou = ev_hsm
        ev_tou = np.array(ev_tou)
        ev_tou[int(8 * samples_per_hour):int(17 * samples_per_hour + 1)] = 0
    else:
        add_ev_l1 = 0

        # Fetching EV amplitude from previous run

    if ev_hsm_present and item_input_object.get("item_input_params").get('ev_hsm').get('item_amp') is not None:
        ev_amp = item_input_object.get("item_input_params").get('ev_hsm').get('item_amp')

        if isinstance(ev_amp, list):
            ev_amp = ev_amp[0]
    else:
        add_ev_l1 = 0

    return ev_amp, ev_tou, add_ev_l1
