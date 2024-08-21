"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to find cleanest boxes for estimation
"""

# Import python packages

import numpy as np
from copy import deepcopy


def get_cleanest_boxes(box_feats, column_id, percentile, greater_than, ev_config):
    """
    Functions to find cleanest boxes for estimation

        Parameters:
            box_feats             (np.ndarray)            : Features of the final boxes
            column_id             (int)                   : Column id of feature being used to identify clean boxes
            percentile            (int)                   : Percentile threshold for clean boxes
            greater_than          (bool)                  : Whether to take high percentile or low percentile boxes
            ev_config              (dict)                  : EV module config parameters

        Returns:
            cleanest_boxes_idx    (np.ndarray)            : Indices of the cleanest boxes
    """

    if len(box_feats) <= 0:
        return []

    #  Getting parameter dicts to be used in this function
    est_config = ev_config['estimation']
    columns_dict = ev_config['box_features_dict']

    # Fetching useful parameters
    clean_boxes_auc_percentile = est_config['clean_boxes_auc_percentile']

    box_features = deepcopy(box_feats)

    ev_label_1_idx = np.where(box_features[:, -1])[0]

    # Valid boxes on the basis of AUC

    auc_threshold = np.percentile(box_features[:, columns_dict['boxes_areas_column']], clean_boxes_auc_percentile)

    valid_auc_boxes_idx = np.where(box_features[:, columns_dict['boxes_areas_column']] >= auc_threshold)[0]

    column_values = box_features[:, column_id]

    energy_threshold = np.percentile(box_features[:, column_id], percentile)

    if greater_than:
        cleanest_boxes_idx = np.where(column_values > energy_threshold)[0]
    else:
        cleanest_boxes_idx = np.where(column_values <= energy_threshold)[0]

    cleanest_boxes_idx_copy = deepcopy(cleanest_boxes_idx)

    # Select clean boxes which are part of the EV potential boxes as well

    cleanest_boxes_idx = np.intersect1d(cleanest_boxes_idx, valid_auc_boxes_idx)
    cleanest_boxes_idx = np.intersect1d(cleanest_boxes_idx, ev_label_1_idx)

    # If a very low number of clean boxes left, pick boxes on the basis of energy only
    min_clean_boxes = est_config['min_clean_boxes']
    if len(cleanest_boxes_idx) < min_clean_boxes:
        cleanest_boxes_idx = cleanest_boxes_idx_copy

    # Least standard deviation check

    max_boxes = est_config['max_clean_boxes']

    temp_clean_boxes_features = box_features[cleanest_boxes_idx, :]

    n_boxes = np.min([max_boxes, temp_clean_boxes_features.shape[0]]).astype(int)

    low_deviation_idx = np.argsort(temp_clean_boxes_features[:, columns_dict['boxes_energy_std_column']])[:n_boxes]

    # Final cleanest boxes idx

    cleanest_boxes_idx = cleanest_boxes_idx[low_deviation_idx]

    return cleanest_boxes_idx
