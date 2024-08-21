
"""
Author - Nisha Agarwal
Date - 10th Mar 2020
utils function for ev box addition
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.init_itemization_config import seq_config
from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config


def get_hsm_of_incremental_mode(item_input_object):

    """
    fetch ev hsm params

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs

    Returns:
        hsm_ev_type                 (int)        : hsm ev charger type
    """

    ev_config = get_inf_config().get('ev')

    hsm_ev_type = ev_config.get('default_type')

    valid_hsm_present = check_validity_of_hsm(item_input_object.get("config").get('disagg_mode') == 'incremental',
                                              item_input_object.get("item_input_params").get('ev_hsm'), 'item_type')

    if valid_hsm_present:
        ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(ev_hsm, list):
            hsm_ev_type = ev_hsm[0]
        else:
            hsm_ev_type = ev_hsm

    return hsm_ev_type


def filter_valid_ev_boxes(ev_box_count, l1_bool, valid_idx, box_seq, timed_output, valid_seq, samples_per_hour, non_ev_hours):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        ev_box_count                (int)         : count of valid ev box
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type
        valid_idx                   (np.ndarray)  : new ev time of use
        box_seq                     (np.ndarray)  : seq with params of all detected boxes
        timed_output                (np.ndarray)  : output of timed signature detction output
        valid_seq                   (np.ndarray)  : array containing information about which all boxes can be ev
        samples_per_hour            (int)         : number of samples in an hour
        non_ev_hours                 (np.ndarray)  : hours which are neglected while detecting ev usage (usually day time)

    Returns:
        ev_box_count                (int)         : count of valid ev box
        valid_idx                   (np.ndarray)  : new ev time of use
        box_seq                     (np.ndarray)  : seq with params of all detected boxes
    """

    # for each detected box
    # we check :
    # chances of the box being an ev usage
    # tou of the box
    # whether the box was lying in the timed signature

    total_samples = samples_per_hour * Cgbdisagg.HRS_IN_DAY

    for i in range(len(valid_seq)):

        seq_strt = box_seq[i, 1]
        seq_end = box_seq[i, 2]
        seq_len = box_seq[i, 3]
        seq_val = box_seq[i, 0]

        valid_ev_box = (seq_strt % total_samples not in non_ev_hours) or\
                       (l1_bool and seq_len > 8*samples_per_hour and
                        (seq_strt % (samples_per_hour * 24) not in np.arange(1*samples_per_hour, 17*samples_per_hour + 1))) or \
                       (l1_bool and seq_len > 10*samples_per_hour and
                        (seq_strt % total_samples not in np.arange(1*samples_per_hour, 15*samples_per_hour + 1)))

        if seq_val and valid_seq[i] and valid_ev_box and (not timed_output[seq_strt: seq_end + 1].sum() > 2 * samples_per_hour):
            valid_idx[seq_strt: seq_end + 1] = 1
            box_seq[i, 0] = 0
            ev_box_count = ev_box_count + 1

    return ev_box_count, valid_idx, box_seq


def fetch_ev_hsm_params(item_input_object):

    """
    fetch ev hsm params

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs

    Returns:
        hsm_ev_type                 (int)        : hsm ev charger type
        hsm_ev_amp                  (float)      : hsm ev charger amplitude
    """

    ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

    if ev_hsm is None:
        hsm_ev_type = '0'
    elif isinstance(ev_hsm, list):
        hsm_ev_type = ev_hsm[0]
    else:
        hsm_ev_type = ev_hsm

    ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_amp')

    if ev_hsm is None:
        hsm_ev_amp = 0
    elif isinstance(ev_hsm, list):
        hsm_ev_amp = ev_hsm[0]
    else:
        hsm_ev_amp = ev_hsm

    return hsm_ev_type, hsm_ev_amp


def get_low_ev_usage_hours(l1_bool, samples_per_hour, ev_disagg):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type
        samples_per_hour            (int)         : number of samples in an hour
        ev_disagg                   (np.ndarray)  : EV TS level disagg output

    Returns:
        non_ev_hours                (np.ndarray)  : hours with very low probability of ev being used
    """

    ev_config = get_inf_config(samples_per_hour).get('ev')

    if l1_bool:
        non_ev_hours = ev_config.get('l1_non_ev_hours')
    else:
        non_ev_hours = ev_config.get('l2_non_ev_hours')

    if np.sum(ev_disagg) and l1_bool:
        non_ev_hours = ev_config.get('disagg_l1_non_ev_hours')
    elif np.sum(ev_disagg):
        non_ev_hours = ev_config.get('disagg_l2_non_ev_hours')

    return non_ev_hours


def calculate_ev_box_count_threshold(item_input_object, valid_idx, ev_disagg, amplitude, l1_bool, input_data, pilot, ev_box_count, logger):

    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs
        valid_idx                   (np.ndarray)  : new ev time of use
        ev_disagg                   (np.ndarray)  : ev disagg outout
        amplitude                   (float)       : amplitude of detected ev box
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type
        input_data                  (np.ndarray)  : raw data
        pilot                       (int)         : pilot id
        ev_box_count                (int)         ; valid ev boxes
        logger                      (logger)      : logger object

    Returns:
        ev_box_threshold            (int)         : min ev box required
    """

    # preparing min number of boxes needed for ev addition

    ev_config = get_inf_config().get('ev')

    days_frac_for_recent_days = ev_config.get('days_frac_for_recent_days')
    max_days = ev_config.get('max_days')

    ev_box_threshold = ev_config.get('ev_box_threshold')

    ev_box_threshold = ev_config.get('eu_ev_box_threshold') * (pilot in ev_config.get('eu_pilots')) + \
                       ev_box_threshold * (pilot not in ev_config.get('eu_pilots'))

    logger.info("Number of EV strikes found in residual is %d | ", ev_box_count)

    if not l1_bool:
        ev_box_threshold = max(ev_config.get('min_ev_l2_box_thres'), int(ev_box_threshold * (len(input_data) / max_days)))
    else:
        ev_box_threshold = max(ev_config.get('min_ev_l1_box_thres'), int(ev_box_threshold * (len(input_data) / max_days)))

    if l1_bool:
        temp = np.reshape(valid_idx, input_data.shape)

        ev_days = (np.sum(temp, axis=1) > 0).astype(int)

        recent_days_flag = ev_days[-ev_config.get('recent_ev_days_thres'):].sum() / ev_days.sum() > days_frac_for_recent_days

        ev_box_threshold = recent_days_flag * ev_config.get('recent_ev_box_thres') + (not recent_days_flag) * ev_box_threshold

    # if ev is already detected in previous run, less number of boxes are required in present run

    hsm_ev_type = 0

    valid_hsm_present = check_validity_of_hsm(item_input_object.get("config").get('disagg_mode') == 'incremental',
                                              item_input_object.get("item_input_params").get('ev_hsm'), 'item_type')

    if valid_hsm_present:
        hsm_ev_type = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(hsm_ev_type, list):
            hsm_ev_type = hsm_ev_type[0]

    inc_run_ev_type_matching_with_current_run = (hsm_ev_type == 1 and l1_bool) or (hsm_ev_type == 2 and not l1_bool)

    if inc_run_ev_type_matching_with_current_run:
        ev_box_threshold = ev_config.get('inc_ev_box_thres')

    # less number of boxes are required if already detected in ev disagg mode

    if np.sum(ev_disagg):
        ev_box_threshold = ev_config.get('disagg_ev_box_thres')

    # less number of boxes are required incase of mtd mode

    if (item_input_object.get("config").get('disagg_mode') == 'mtd') and amplitude > 0:
        ev_box_threshold = ev_config.get('mtd_ev_box_thres')

    logger.info('ev box count threshold | %s ', ev_box_threshold)

    return ev_box_threshold


def get_required_ev_box_params(item_input_object, l1_bool, samples_per_hour, ev_disagg, pilot):
    """
    Calculate EV consumption in the leftover residual data

    Parameters:
        item_input_object           (dict)        : Dict containing all hybrid outputs
        l1_bool                     (bool)        : true if ev l1_bool output has to be estimated, false in case of L2 charger type
        samples_per_hour            (int)         : number of samples in an hour
        ev_disagg                   (np.ndarray)  : EV TS level disagg output
        pilot                       (int)         : pilot id

    Returns:
        max_cap                     (np.ndarray)  : required max amplitude for ev detection
        min_cap                     (np.ndarray)  : required min amplitude for ev detection
        min_len                     (np.ndarray)  : required max box length for ev detection
        max_len                     (np.ndarray)  : required min box length for ev detection
        low_ev_usage_hours          (np.ndarray)  : hours with very low probability of ev being used
    """

    max_amp = np.max(ev_disagg)

    ev_config = get_inf_config(samples_per_hour).get('ev')

    amp_buffer_for_eu_pilots = ev_config.get('amp_buffer_for_eu_pilots')
    min_amp_l2_charging_boxes = ev_config.get('min_amp_l2_charging_boxes')
    min_amp_l2_charging_boxes_for_disagg_user = ev_config.get('min_amp_l2_charging_boxes_for_disagg_user')
    buffer_amp_from_disagg_cons = ev_config.get('buffer_amp_from_disagg_cons')
    max_buffer_amp = ev_config.get('max_buffer_amp')

    valid_hsm_present = \
        (item_input_object.get('created_hsm') is not None) and \
        (item_input_object.get('created_hsm').get('ev') is not None) and \
        (item_input_object.get('created_hsm').get('ev').get('attributes') is not None) and \
        (item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude") is not None)

    if valid_hsm_present:
        amp = item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude")
        max_amp = amp * ev_config.get('amp_multiplier')

    # Calculating EV max/min box capacity, and max/min box length

    l2_type_boxes_detected_in_disagg = np.sum(ev_disagg) > 0 and \
                                       np.median(ev_disagg[ev_disagg > 0]) > min_amp_l2_charging_boxes_for_disagg_user

    #  if ev l2 is detected in disagg
    if l2_type_boxes_detected_in_disagg:
        min_cap = [max(min_amp_l2_charging_boxes, np.median(ev_disagg[ev_disagg > 0]) - buffer_amp_from_disagg_cons),
                   max(min_amp_l2_charging_boxes, np.median(ev_disagg[ev_disagg > 0]) / 2 - buffer_amp_from_disagg_cons),
                   max(min_amp_l2_charging_boxes, np.median(ev_disagg[ev_disagg > 0]) - buffer_amp_from_disagg_cons)]
        max_cap = [max_amp + max_buffer_amp, max_amp + max_buffer_amp, max_amp + max_buffer_amp]

        min_len = ev_config.get('disagg_l2_min_len')
        max_len = ev_config.get('disagg_l2_max_len')

    else:
        #  if ev not detected in disagg but looking for ev l2 boxes

        if not l1_bool:
            max_cap = ev_config.get('l2_max_cap')
            min_cap = ev_config.get('l2_min_cap')
            min_len = ev_config.get('l2_min_len')
            max_len = ev_config.get('l2_max_len')

            if pilot in ev_config.get('eu_pilots'):
                min_cap[0] = np.array(min_cap[0]) - amp_buffer_for_eu_pilots
                min_cap[2] = np.array(min_cap[2]) - amp_buffer_for_eu_pilots
                max_len = ev_config.get('eu_l2_max_len')

        #  if looking for ev l1 boxes

        else:
            max_cap = ev_config.get('l1_max_cap')
            min_cap = ev_config.get('l1_min_cap')
            min_len = ev_config.get('l1_min_len')
            max_len = ev_config.get('l1_max_len')

            if np.sum(ev_disagg):
                min_len = np.minimum(4 * samples_per_hour, np.array(min_len))

    low_ev_usage_hours = get_low_ev_usage_hours(l1_bool, samples_per_hour, ev_disagg)

    return max_cap, min_cap, min_len, max_len, low_ev_usage_hours



def check_ev_output(item_output_object, mid_cons, samples):

    """
    Modify appliance ev cases by removing low duration boxes

    Parameters:
        item_output_object         (dict)          : Dict containing all hybrid outputs
        mid_cons                   (np.ndarray)    : TS level ev cons
        samples                    (int)           : samples in an hour

    Returns:
        mid_cons                   (np.ndarray)    : TS level ev cons
    """

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    if np.sum(mid_cons) > 0 and samples > 1:

        ev_usage_arr = mid_cons.flatten()

        thres = min(1.5*samples, 4)

        seq = find_seq(ev_usage_arr > 0, np.zeros_like(ev_usage_arr), np.zeros_like(ev_usage_arr))

        for i in range(len(seq)):

            if (seq[i, seq_len] < thres) and seq[i, seq_label]:
                ev_usage_arr[int(seq[i, seq_start]): int(seq[i, seq_end]) + 1] = 0

        ev_usage_arr = ev_usage_arr.reshape(mid_cons.shape)

        ev_res = item_output_object.get("ev_residual")

        ev_usage_arr[ev_res > 0] = 1

        mid_cons[ev_usage_arr == 0] = 0

    if np.sum(mid_cons) > 0 and samples > 1:
        ev_usage_arr = mid_cons.flatten()

        thres = int(0.5 * samples) + 1

        seq = find_seq(ev_usage_arr > 0, np.zeros_like(ev_usage_arr), np.zeros_like(ev_usage_arr))

        for i in range(len(seq)):

            if (seq[i, seq_len] < thres) and seq[i, seq_label]:
                ev_usage_arr[int(seq[i, seq_start]): int(seq[i, seq_end]) + 1] = 0

        ev_usage_arr = ev_usage_arr.reshape(mid_cons.shape)

        mid_cons[ev_usage_arr == 0] = 0

    return mid_cons


def get_freq(valid_idx):
    """
    determine whether EV can be added in hybrid, based on distribution of high consumption boxes

    Parameters:
        valid_idx                 (np.ndarray)    : initial ev detection points

    Returns:
        avg_ev_chunck_len         (int)           : len of ev
    """

    seq_label = seq_config.SEQ_LABEL

    shape = valid_idx.shape

    samples = shape[1]

    valid_idx = (valid_idx > 0).flatten()

    seq = find_seq(valid_idx, np.zeros_like(valid_idx), np.zeros_like(valid_idx))

    if np.all(seq[:, seq_label] == 0):
        return -1

    seq = seq[seq[:, seq_label] == 0, 3] / (samples)

    avg_ev_chunck_len = np.percentile(seq, 30)

    return avg_ev_chunck_len


def check_seasonal_check_flag(item_input_object, residual_cons):
    """
    check whether ev seasonal checks is required based on previous run data

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        residual_cons             (np.ndarray)    : Residual data
    Returns:
        run_seasonal_check        (bool)          : check whether ev seasonal checks is required
    """

    run_seasonal_check = 1

    if item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_type') is not None:
        ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(ev_hsm, list):
            ev_type = ev_hsm[0]
        else:
            ev_type = ev_hsm

        if ev_type == 2 and len(residual_cons) > Cgbdisagg.DAYS_IN_MONTH * 3:
            run_seasonal_check = 0

    return run_seasonal_check


def get_recent_ev_flag(item_input_object, recent_ev):

    """
    check whether detected ev is recent

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
        recent_ev                 (bool)          : recent ev flag
    Returns:
        recent_ev                 (bool)          : recent ev flag
    """

    recent_ev_flag = (item_input_object.get('created_hsm') is not None) and \
    (item_input_object.get('created_hsm').get('ev') is not None) and \
    (item_input_object.get('created_hsm').get('ev').get('attributes') is not None) and \
    (item_input_object.get('created_hsm').get('ev').get('attributes').get("recent_ev") is not None)

    if recent_ev_flag:
        recent_ev = recent_ev or item_input_object.get('created_hsm').get('ev').get('attributes').get("recent_ev")

    return recent_ev


def get_charger_type(item_input_object):

    """
    check charger type of ev disagg

    Parameters:
        item_input_object         (dict)          : Dict containing all inputs
    Returns:
        charger_type              (int)           : charger type of ev disagg
    """

    charger_type = 2

    charger_type_present = (item_input_object.get('created_hsm') is not None) and \
                           (item_input_object.get('created_hsm').get('ev') is not None) and \
                           (item_input_object.get('created_hsm').get('ev').get('attributes') is not None) and \
                           (item_input_object.get('created_hsm').get('ev').get('attributes').get("charger_type") is not None)

    if charger_type_present:
        charger_type = item_input_object.get('created_hsm').get('ev').get('attributes').get("charger_type")

    return charger_type


def fetch_ev_l1_hsm_params(item_input_object, input_data):

    """
    fetch ev l1 usage freq and tag from previous run info

    Parameters:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        input_data                  (np.ndarray): input data

    Returns:
        freq_of_ev_l1_usage         (int)       : freq of l1 ev usage
        l1_tag                      (int)       : flag to show whether ev was detected in previous run
    """

    freq_of_ev_l1_usage = -1
    l1_tag = -1

    # fetch hsm info for ev l1 parameters

    if item_input_object.get("config").get('disagg_mode') in ['incremental', 'mtd'] and \
            item_input_object.get("item_input_params").get('ev_hsm') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_type') is not None \
            and item_input_object.get("item_input_params").get('ev_hsm').get('item_freq') is not None:
        ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_type')

        if isinstance(ev_hsm, list):
            ev_type = ev_hsm[0]
        else:
            ev_type = ev_hsm

        if (ev_type == 1 and len(input_data) > 100) or (ev_type == 1 and item_input_object.get("config").get('disagg_mode') in ['mtd']):
            l1_tag = 1

        ev_hsm = item_input_object.get("item_input_params").get('ev_hsm').get('item_freq')

        if isinstance(ev_hsm, list):
            ev_freq = ev_hsm[0]
        else:
            ev_freq = ev_hsm

        if (ev_freq > 0 and len(input_data) > 100) or (ev_freq > 0 and item_input_object.get("config").get('disagg_mode') in ['mtd']):
            freq_of_ev_l1_usage = ev_freq

    return freq_of_ev_l1_usage, l1_tag


def update_ev_hsm_with_l1_charger_params(l1_tag, freq_of_ev_l1_usage, residual_cons, mid_cons, item_input_object, item_output_object, disagg_cons):

    """
    Update ev hsm with hybrid attributes for ev l1 user

    Parameters:
        freq_of_ev_l1_usage         (int)       : freq of l1 ev usage
        l1_tag                      (int)       : flag to show whether ev was detected in previous run
        residual_cons               (np.ndarray): hybrid ev cons
        mid_cons                    (np.ndarray): ev output
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        disagg_cons                 (np.ndarray): ev disagg output

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    samples_per_hour = int(mid_cons.shape[1] / Cgbdisagg.HRS_IN_DAY)

    # preparing inputs to be added into HSM data

    if (mid_cons > 0).sum() > 0:
        ev_days_seq = find_seq((mid_cons > 0).flatten(), np.zeros(mid_cons.size), np.zeros(mid_cons.size))
        ev_days_seq = ev_days_seq[ev_days_seq[:, 0] == 0]
        ev_usage_freq = np.diff(ev_days_seq[:, 1]) / mid_cons.shape[1]

        if np.any(ev_usage_freq < Cgbdisagg.DAYS_IN_MONTH):
            ev_usage_freq = ev_usage_freq[ev_usage_freq < Cgbdisagg.DAYS_IN_MONTH]
            ev_usage_freq = max(0, int(np.ceil(np.median(ev_usage_freq))))
        else:
            ev_usage_freq = 0
    else:
        ev_usage_freq = 0

    created_hsm = dict({
        'item_tou': (np.sum(mid_cons>0, axis=0) > 10),
        'item_freq': ev_usage_freq
    })

    # adding params into hsm dictionary

    if post_hsm_flag and (item_output_object.get('created_hsm').get('ev') is None):
        item_output_object['created_hsm']['ev'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    if post_hsm_flag and \
            (item_output_object.get('created_hsm').get('ev') is not None) and \
            (item_output_object.get('created_hsm').get('ev').get('attributes') is not None):
        item_output_object['created_hsm']['ev']['attributes'].update(created_hsm)

    if (np.sum(disagg_cons) == 0 and np.sum(residual_cons) == 0 and l1_tag == 1 and freq_of_ev_l1_usage > 0) and np.sum(mid_cons) > 0:

        created_hsm = dict({
            'item_type': 1,
            'item_amp': np.percentile(mid_cons[mid_cons > 0], 90) * samples_per_hour
        })

        if post_hsm_flag and (item_output_object.get('created_hsm').get('ev') is None):
            item_output_object['created_hsm']['ev'] = {
                'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
                'attributes': dict()
            }

        if post_hsm_flag and \
                (item_output_object.get('created_hsm').get('ev') is not None) and \
                (item_output_object.get('created_hsm').get('ev').get('attributes') is not None):
            item_output_object['created_hsm']['ev']['attributes'].update(created_hsm)

    return item_output_object


def update_ev_hsm(item_input_object, item_output_object, disagg_cons):

    """
    Update ev hsm with hybrid attributes

    Parameters:
        item_input_object           (dict)      : Dict containing all hybrid inputs
        item_output_object          (dict)      : Dict containing all hybrid outputs
        disagg_cons                 (np.ndarray): ev disagg output

    Returns:
        item_output_object        (dict)      : updated Dict containing all hybrid outputs
    """

    created_hsm = dict({
        'item_type': 0,
        'item_amp': 0
    })

    post_hsm_flag = item_input_object.get('item_input_params').get('post_hsm_flag')

    if post_hsm_flag and (item_output_object.get('created_hsm').get('ev') is None):
        item_output_object['created_hsm']['ev'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    if post_hsm_flag and \
            (item_output_object.get('created_hsm').get('ev') is not None) and \
            (item_output_object.get('created_hsm').get('ev').get('attributes') is not None):
        item_output_object['created_hsm']['ev']['attributes'].update(created_hsm)

    return item_output_object
