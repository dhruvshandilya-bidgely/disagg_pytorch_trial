"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to filter noise boxes from estimation
"""

# Import python packages
from copy import deepcopy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.ev.functions.detection.dynamic_box_fitting import get_ev_boxes


def filter_noise_boxes(updated_box_data, debug, ev_config, logger_base, raw_data, hsm_in):
    """
    Function to remove noise boxes from EV estimation

    Parameters:
        updated_box_data          (np.ndarray)        : Box data
        debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
        ev_config                  (dict)              : Module config dict
        logger_base               (logger)            : logger base
        raw_data                  (np.ndarray)        : Input 13 columns matrix
        hsm_in                    (dict)              : Input hsm object

    Returns:
        box_data                  (np.ndarray)        : New box data
        debug                     (dict)              : Object containing all important data/values as well as HSM

    """
    logger_local = logger_base.get('logger').getChild('filter_noise_boxes')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    #  Getting parameter dicts to be used in this function
    est_config = ev_config['estimation']
    columns_dict = ev_config['box_features_dict']

    # Extracting useful parameters to be used in the function
    neighborhood_points_hrs = est_config['neighborhood_points_hrs']
    box_end_idx_col = columns_dict['end_index_column']

    # Specific logger fo this function

    box_data = deepcopy(updated_box_data)

    input_data = deepcopy(raw_data)

    # Extract required values from debug

    labeled_boxes = debug['labeled_features']

    # Check if MTD mode

    if ev_config['disagg_mode'] == 'mtd':
        max_energy_allowed = hsm_in['max_energy_allowed'][0]
        max_deviation_allowed = hsm_in['max_deviation_allowed'][0]
    else:

        clean_box_idx = debug['clean_boxes_idx']

        if len(clean_box_idx) > 0:
            max_deviation_allowed = maximum_deviation_allowed(labeled_boxes, clean_box_idx, ev_config)

            max_energy_allowed = maximum_allowed_energy(labeled_boxes, clean_box_idx, debug['type'], ev_config)
        else:
            max_deviation_allowed, max_energy_allowed = 0, 0

        debug['max_energy_allowed'] = deepcopy(max_energy_allowed)
        debug['max_deviation_allowed'] = deepcopy(max_deviation_allowed)

    # Fill values for each valid box

    if debug['type'] == 'energy':

        for _, row in enumerate(labeled_boxes):
            start_idx, end_idx = row[: box_end_idx_col + 1].astype(int)

            if row[-1] == 0:
                box_data[start_idx: (end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0
            else:
                n_neighbor_points = int(neighborhood_points_hrs * debug['factor'])

                box_energy = input_data[start_idx: (end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX]

                minimum_energy = np.min(box_energy)

                before_energy = input_data[(start_idx - n_neighbor_points): start_idx, Cgbdisagg.INPUT_CONSUMPTION_IDX]

                after_energy = input_data[(end_idx + 1): (end_idx + 1 + n_neighbor_points),
                                          Cgbdisagg.INPUT_CONSUMPTION_IDX]

                box_energy -= np.min(np.r_[before_energy, after_energy])

                base_energy = np.percentile(box_energy, 50)

                capped_box_energy = np.fmin(box_energy, base_energy + max_deviation_allowed)
                capped_box_energy = np.fmax(capped_box_energy, minimum_energy)

                box_data[start_idx: (end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = capped_box_energy
    else:
        for _, row in enumerate(labeled_boxes):
            start_idx, end_idx = row[:box_end_idx_col + 1].astype(int)

            if row[-1] == 0:
                box_data[start_idx: (end_idx + 1), Cgbdisagg.INPUT_CONSUMPTION_IDX] = 0

        box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = np.fmin(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX],
                                                               max_energy_allowed)

    sampling_rate = ev_config.get('sampling_rate')

    region = ev_config['region']

    factor = debug['factor']

    minimum_duration = ev_config['minimum_duration'][sampling_rate]

    # Start box amplitude varies for L1 and L2/L3, so select accordingly

    start_box_amplitude = get_start_box_amp(debug, ev_config, region)

    final_boxes_amp_ratio = est_config['final_boxes_amp_ratio']

    final_box_amplitude = np.fmax(start_box_amplitude, final_boxes_amp_ratio * debug['ev_amplitude'])

    final_minimum_energy = final_box_amplitude / factor

    box_data = get_ev_boxes(box_data, minimum_duration, final_minimum_energy, factor, logger_pass)

    return box_data, debug


def get_start_box_amp(debug, ev_config, region):
    """
    Function to get the start box amplitude based on the EV charger type
    Parameters:
        debug                   (dict)          : Debug dictionary
        ev_config               (dict)          : EV configurations dictionary
        region                  (string)        : Region
    Returns:
        start_box_amplitude     (float)         : Start box amplitude
    """

    if debug['charger_type'] == 'L1':
        start_box_amplitude = debug['l1_config']['detection'][region + '_start_box_amplitude']
    else:
        start_box_amplitude = ev_config['detection'][region + '_start_box_amplitude']

    return start_box_amplitude


def maximum_allowed_energy(labeled_boxes, clean_box_idx, gmm_type, ev_config):
    """
    Calculation of maximum allowed energy as a percentile of clean boxes

    Parameters:
        labeled_boxes             (np.ndarray)        : Features of labeled boxes
        clean_box_idx             (np.array)          : Indices of clean boxes
        gmm_type                  (str)               : Type of gmm cluster
        ev_config                  (dict)              : Module config dict

    Returns:
        max_energy_allowed        (float)              : maximum allowed energy of the boxes
    """

    #  Getting parameter dicts to be used in this function
    est_config = ev_config.get('estimation')
    columns_dict = ev_config.get('box_features_dict')

    # Getting metrics from clean boxes
    clean_boxes = labeled_boxes[clean_box_idx, :]

    energy_gmm_percentile = est_config.get('energy_gmm_percentile')
    duration_gmm_percentile = est_config.get('duration_gmm_percentile')

    if gmm_type == 'energy':
        max_energy_allowed = np.percentile(clean_boxes[:, columns_dict['boxes_energy_per_point_column']],
                                           energy_gmm_percentile)
    else:
        max_energy_allowed = np.percentile(clean_boxes[:, columns_dict['boxes_energy_per_point_column']],
                                           duration_gmm_percentile)

    return max_energy_allowed


def maximum_deviation_allowed(labeled_boxes, clean_box_idx, ev_config):
    """
    Calculation of maximum allowed deviation on the basis of clean boxes

    Parameters:
        labeled_boxes             (np.ndarray)        : Features of labeled boxes
        clean_box_idx             (np.array)          : Indices of clean boxes
        ev_config                  (dict)              : Module config dict

    Returns:
        max_deviation_allowed     (float)              : maximum allowed energy of the boxes
    """
    #  Getting parameter dicts to be used in this function
    columns_dict = ev_config.get('box_features_dict')

    clean_boxes = labeled_boxes[clean_box_idx, :]

    max_deviation_allowed = np.max(clean_boxes[:, columns_dict['boxes_energy_std_column']])

    return max_deviation_allowed
