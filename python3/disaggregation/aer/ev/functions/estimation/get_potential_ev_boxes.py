"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to find Potential EV boxes for estimation
"""

# Import python packages

import logging
import numpy as np
from scipy import stats
from copy import deepcopy

# Import within modules

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.ev.functions.detection.dynamic_box_fitting import remove_timed_block_boxes
from python3.disaggregation.aer.ev.functions.get_ev_boxes import get_ev_boxes

from python3.disaggregation.aer.ev.functions.detection.get_boxes_features import boxes_features
from python3.disaggregation.aer.ev.functions.get_season_categorisation import add_season_month_tag
from python3.disaggregation.aer.ev.functions.estimation.get_cleanest_boxes import get_cleanest_boxes
from python3.disaggregation.aer.ev.functions.estimation.estimation_post_processing import post_processing


def get_potential_ev_boxes(debug, ev_config, logger_base, hsm_in):
    """
    Function to remove noise boxes from EV estimation

        Parameters:
            debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
            ev_config                  (dict)              : Module config dict
            logger_base               (logger)            : logger base
            hsm_in                    (dict)              : Input hsm object

        Returns:
            new_box_data              (np.ndarray)        : New box data
            debug                     (dict)              : Object containing all important data/values as well as HSM

    """
    logger_local = logger_base.get('logger').getChild('get_box_clusters')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    #  Getting parameter dicts to be used in this function
    est_config = ev_config.get('estimation')
    columns_dict = ev_config.get('box_features_dict')

    sampling_rate = ev_config.get('sampling_rate')

    region = ev_config.get('region')

    factor = debug.get('factor')

    minimum_duration = ev_config.get('minimum_duration').get(sampling_rate)

    if ev_config['disagg_mode'] == 'mtd':
        lower_amp = hsm_in['lower_amplitude'][0]
        ev_duration_mean = hsm_in['mean_duration'][0]

        debug['clean_box_tou'] = hsm_in['clean_box_tou'][0]
        debug['clean_box_amp'] = hsm_in['clean_box_amp'][0]
        debug['clean_box_auc'] = hsm_in['clean_box_auc'][0]
        debug['clean_box_dur'] = hsm_in['clean_box_dur'][0]

        recent_ev = hsm_in['recent_ev'][0]

        debug['mean_duration'] = deepcopy(ev_duration_mean)

        debug['lower_amplitude'] = lower_amp
    else:
        # Get the most relevant box data and the box features

        target_key = get_target_key(debug)

        # Get possible EV usage boxes according to L1/L2 chargers

        if debug['charger_type'] == 'L1':
            box_data = debug['l1']['updated_box_data_' + target_key]
            box_features = debug['l1']['updated_box_features_' + target_key]
            lower_amp = debug['l1']['amplitude_' + target_key]
        else:
            box_data = debug['updated_box_data_' + target_key]
            box_features = debug['updated_box_features_' + target_key]
            lower_amp = debug['amplitude_' + target_key]

        # Recent EV

        recent_ev = debug['recent_ev']

        # Save boxes data & Create box features

        debug['estimation_box_output'] = deepcopy(box_data)

        debug['estimation_box_features'] = deepcopy(box_features)

        # Find average EV boxes duration

        ev_duration_mean = np.mean(box_features[:, columns_dict['boxes_duration_column']])

        debug['mean_duration'] = deepcopy(ev_duration_mean)

        logger.info('Mean EV duration (in hours) | {}'.format(ev_duration_mean))

        # Get an early estimate of the amplitude

        cleanest_boxes_idx = get_cleanest_boxes(box_features, column_id=columns_dict['boxes_energy_std_column'],
                                                percentile=est_config['clean_boxes_std_percentile'], greater_than=False,
                                                ev_config=ev_config)
        debug['clean_boxes_idx'] = cleanest_boxes_idx

        amp_col = columns_dict['boxes_energy_per_point_column']

        debug = get_clean_box_attributes(box_features, cleanest_boxes_idx, columns_dict, debug)

        initial_ev_amplitude = np.mean(box_features[cleanest_boxes_idx, amp_col])

        if region == 'NA':
            lower_amp = min(lower_amp, est_config['na_lower_amp_ratio'] * initial_ev_amplitude)
        elif region == 'EU':
            lower_amp = min(lower_amp, est_config['eu_lower_amp_ratio'] * initial_ev_amplitude)

        debug['lower_amplitude'] = lower_amp

    min_energy = lower_amp / factor

    logger.info("lower amplitude for EV estimation: | {}".format(lower_amp))
    # Get new set of boxes using the determined amplitude

    amp_col = columns_dict['boxes_energy_per_point_column']
    input_data = debug.get('input_after_baseload')
    new_box_data = get_ev_boxes(input_data, minimum_duration, min_energy, factor, logger_pass)
    new_box_data = remove_timed_block_boxes(new_box_data, debug, ev_config, logger_base)

    new_box_data = add_season_month_tag(new_box_data)

    box_features = boxes_features(new_box_data, factor, ev_config)

    # Identify cleanest boxes for calculating ideal metrics (if not MTD model)

    if ev_config['disagg_mode'] != 'mtd':

        new_box_data, box_features = post_processing(debug, new_box_data, box_features, ev_config, logger)

        cleanest_boxes_idx = get_cleanest_boxes(box_features, column_id=columns_dict['boxes_energy_std_column'],
                                                percentile=est_config['clean_boxes_std_percentile'], greater_than=False,
                                                ev_config=ev_config)
        debug['clean_boxes_idx'] = cleanest_boxes_idx

        final_ev_amplitude = int(np.mean(box_features[cleanest_boxes_idx, amp_col])) if len(cleanest_boxes_idx) > 0 else 0

        debug['ev_amplitude'] = final_ev_amplitude

    elif ev_config.get('disagg_mode') == 'mtd' and str(ev_config.get('pilot_id')) in ev_config.get('est_boxes_refine_pilots'):

        # apply post-processing if disagg mode is mtd but pilot-id is in the list of pilots
        # where estimation boxes have to be refined aggressively to remove doubtful boxes
        new_box_data, box_features = post_processing(debug, new_box_data, box_features, ev_config, logger)

        final_ev_amplitude = hsm_in['ev_amplitude'][0]

        debug['ev_amplitude'] = final_ev_amplitude

    else:
        final_ev_amplitude = hsm_in['ev_amplitude'][0]

        debug['ev_amplitude'] = final_ev_amplitude

    logger.info('EV amplitude (in Watts) | {}'.format(final_ev_amplitude))

    # Create the features of these new boxes

    final_boxes = np.array([1] * box_features.shape[0])

    labeled_features = np.c_[box_features, final_boxes]

    # Correcting boxes for the partial EV start / stop in the middle of sampling rate

    if len(labeled_features) > 0:
        labeled_features = edge_correction(labeled_features, debug['input_after_baseload'], final_ev_amplitude, factor,
                                           ev_config)

    debug['labeled_features'] = deepcopy(labeled_features)

    # Defining the type of EV detected (High energy / High duration)

    debug['type'] = 'energy'

    if (ev_config['disagg_mode'] != 'mtd') and (recent_ev == 1):
        # Filter out the non-recent data in case of recent EV
        recent_cutoff_idx = debug['recent_data_cutoff_idx']

        new_box_data[:recent_cutoff_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

    return debug, new_box_data


def edge_correction(box_features, baseload, ev_amplitude, factor, ev_config):
    """
    Functions to find cleanest boxes for estimation

        Parameters:
            box_features             (np.ndarray)            : Features of the final boxes
            baseload                 (int)                   : Input 13 column matrix with baseload removed
            ev_amplitude             (float)                  : Amplitude of EV boxes
            factor                   (int)                   : Number of data points in an hour
            ev_config                (dict)                : Module config dict

        Returns:
            labeled_features         (np.ndarray)            : Labeled features with edge corrected
    """
    labeled_features = deepcopy(box_features)
    baseload_data = deepcopy(baseload)

    # Get the data shape

    data_size = baseload_data.shape[0]

    # Get the start and end idx of the boxes

    start_idx = labeled_features[:, 0].astype(int)
    end_idx = labeled_features[:, 1].astype(int)

    # Minimum acceptable energy

    min_energy = ev_config['estimation']['edge_energy_ratio'] * ev_amplitude / factor

    # Get the indices before start and after end

    pre_start_idx = start_idx - 1
    post_end_idx = end_idx + 1

    if pre_start_idx[0] < 0:
        pre_start_idx[0] = 0
        pre_start_issue = True
    else:
        pre_start_issue = False

    if post_end_idx[-1] == data_size:
        post_end_idx[-1] = data_size - 1
        post_end_issue = True
    else:
        post_end_issue = False

    # Get the values before start and after end

    pre_start_values = baseload_data[pre_start_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    post_end_values = baseload_data[post_end_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    # Check if any of the value in the acceptable range

    valid_pre_start_idx = np.where(pre_start_values > min_energy)[0]
    valid_post_end_idx = np.where(post_end_values > min_energy)[0]

    if pre_start_issue:
        valid_pre_start_idx = valid_pre_start_idx[1:]

    if post_end_issue:
        valid_post_end_idx = valid_post_end_idx[:-1]

    # Update the start and end indices

    labeled_features[valid_pre_start_idx, 0] -= 1
    labeled_features[valid_post_end_idx, 1] += 1

    return labeled_features


def get_target_key(debug):
    """
    This function is used to get the target key
    Parameters:
        debug               (dict)          : Debug dictionary
    Returns:
        target_key          (string)        : Target box key number
    """

    if debug['charger_type'] == 'L1':
        target_key = int(debug['l1']['final_box_index'])
    else:
        target_key = int(debug['final_box_index'])

    if target_key > 0:
        target_key = str(target_key)
    else:
        target_key = '1'

    return target_key


def get_clean_box_attributes(box_features, cleanest_boxes_idx, columns_dict, debug):
    """
    Function to store the attributes of clean boxes
    Parameters:
        box_features                (np.ndarray)            : Box features array
        cleanest_boxes_idx          (np.ndarray)            : Indexes of the cleanest boxes
        columns_dict                (dict)                  : Mapping of columns dictionary
        debug                       (dict)                  : Debug dictionary
    Returns:
         debug                       (dict)                  : Debug dictionary
    """

    box_season_col = columns_dict['boxes_start_season']
    dur_col = columns_dict['boxes_duration_column']
    auc_col = columns_dict['boxes_areas_column']
    amp_col = columns_dict['boxes_energy_per_point_column']

    if len(cleanest_boxes_idx):
        debug['clean_box_tou'] = int(stats.mode(box_features[cleanest_boxes_idx, box_season_col])[0])
        debug['clean_box_dur'] = np.mean(box_features[cleanest_boxes_idx, dur_col])
        debug['clean_box_auc'] = np.mean(box_features[cleanest_boxes_idx, auc_col])
        debug['clean_box_amp'] = np.mean(box_features[cleanest_boxes_idx, amp_col])
    else:
        debug['clean_box_tou'] = 0
        debug['clean_box_dur'] = 0
        debug['clean_box_auc'] = 0
        debug['clean_box_amp'] = 0

    return debug
