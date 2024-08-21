"""
Author - Sahana M
Date - 14-Nov-2023
Module to get the EV L1 User detection
"""

# import python packages
import logging
import numpy as np
from numpy import dot
from copy import deepcopy
from numpy.linalg import norm

# import modules from within the project
from python3.utils.maths_utils.find_seq import find_seq
from python3.disaggregation.aer.ev.functions.deep_learning.init_dl_config import FeatureCol
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import extract_box_features


def low_detection_density(prt_predictions, prt_confidences, dl_debug):
    """
    Function to identify if it is a low density user
    Parameters:
        prt_predictions                 (np.ndarray)            : Partition predictions
        prt_confidences                 (np.ndarray)            : Partition confidences
        dl_debug                        (Dict)                  : Deep learning debug
    Returns:
        detection                       (int)                   : Boolean detection value
        confidence                      (float)                 : Overall EV confidence value
        low_density                     (Boolean)               : Boolean for low density
    """

    # Extract required variables
    a_value = dl_debug.get('config').get('a_value')
    r_value_1 = dl_debug.get('config').get('r_value_1')
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')

    # Create a weighted array for each partition
    length = len(prt_predictions)
    weights = np.asarray([a_value * r_value_1 ** i for i in range(length)]).reshape(-1, 1)[::-1]
    weighted_confidences = prt_confidences*weights

    low_predictions = np.sum(prt_predictions) <= 2
    weighted_confidence = np.mean(prt_confidences)

    # If none of the partitions have detections then the overall prediction is 0

    if np.sum(prt_predictions) == 0:
        confidence = np.mean(prt_confidences)
        detection = 0
        low_density = True
        return detection, confidence, low_density

    # There are very low predictions then get a weighted confidence

    elif low_predictions:
        detected_prt = prt_predictions == 1
        weighted_confidence = weighted_confidences[detected_prt] * prt_confidences[detected_prt]
        weighted_confidence = np.mean(weighted_confidence)

    # Check for recent EV

    prior_ev_confidence = np.mean(prt_confidences[:int(len(prt_confidences)/2)])
    after_ev_confidence = np.mean(prt_confidences[int(len(prt_confidences)/2):])

    # Assign the detection and low density values based on the confidence

    confidence = weighted_confidence
    if weighted_confidence >= confidence_threshold:
        detection = 1
        low_density = False
    else:
        detection = 0
        low_density = True

    # Checks for handling Recent EV Scenarios

    if after_ev_confidence >= confidence_threshold > prior_ev_confidence:
        detection = 1
        low_density = False
        confidence = after_ev_confidence

    return detection, confidence, low_density


def get_season_penalties(season_sequences, detected_prts, final_boxes_detected, dl_debug):
    """
    This funciton is used to identify as seasonal FP user and calculate the penalty score
    Parameters:
        season_sequences                (np.ndarray)            : Season sequences in the data array
        detected_prts                   (np.ndarray)            : Detection partitions array
        final_boxes_detected            (np.ndarray)            : Final EV boxes array
        dl_debug                        (Dict)                  : Debug dictionary
    Returns:
        penalty_1_ratio                 (float)                 : Penalty score 1
        penalty_2_ratio                 (float)                 : Penalty score 2
    """

    # Define the local columns
    s_labels = dl_debug.get('config').get('seasonality_penalty').get('s_labels')
    s_label_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('s_label_row')
    avg_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('avg_cons_row')
    ignore_season_percentage_l1 = dl_debug.get('config').get('seasonality_penalty').get('ignore_season_percentage_l1')
    total_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('total_cons_row')
    repetitions_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('repetitions_row')
    relative_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('relative_cons_row')
    percentage_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('percentage_cons_row')
    season_percentage_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('season_percentage_row')
    season_ev_percentage_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('season_ev_percentage_row')

    # For each unique season identify the number of times that season is repeated in the data

    unique_seasons = np.unique(season_sequences[:, 0], return_counts=True)
    seasons_array = np.asarray(s_labels)
    repetitions_arr = np.zeros(shape=(7, 5))
    for i in range(len(seasons_array)):
        if seasons_array[i] in unique_seasons[0]:
            unique_seasons_index = np.where(seasons_array[i] == unique_seasons[0])[0][0]
            repetitions = unique_seasons[1][unique_seasons_index]
            repetitions_arr[0, i] = repetitions
    seasons_array = np.r_[seasons_array.reshape(1, -1), repetitions_arr]

    # For each season get the average and the total consumption

    for i in range(seasons_array.shape[1]):

        season = seasons_array[s_label_row, i]
        season_partitions = detected_prts[detected_prts[:, FeatureCol.S_LABEL] == season]

        if len(season_partitions):
            season_prt_avg_cons = np.percentile(season_partitions[:, FeatureCol.TOTAL_CONSUMPTION], q=50)
            season_prt_total_cons = np.nansum(season_partitions[:, FeatureCol.TOTAL_CONSUMPTION])
            seasons_array[avg_cons_row, i] = season_prt_avg_cons
            seasons_array[total_cons_row, i] = season_prt_total_cons

    # Get the relative percentage consumption for each season
    seasons_array[percentage_cons_row, :] = seasons_array[total_cons_row, :] / np.sum(seasons_array[total_cons_row, :])

    # Extract the season labels calculated from the weather analytics data

    ev_s_label_data = dl_debug.get('s_label_data')
    final_boxes_detected_idx = final_boxes_detected > 0
    final_boxes_detected_idx = final_boxes_detected_idx.flatten()
    ev_s_label_data = ev_s_label_data.flatten()
    ev_s_label_data = ev_s_label_data[final_boxes_detected_idx]
    all_ev_unique_seasons = np.unique(ev_s_label_data, return_counts=True)

    s_label_data = dl_debug.get('s_label_data')
    s_label_data = s_label_data.flatten()
    all_unique_seasons = np.unique(s_label_data, return_counts=True)

    # Identify the percentage of each season in the data

    for i in range(seasons_array.shape[1]):

        season = seasons_array[s_label_row, i]

        if season in all_ev_unique_seasons[0]:
            idx = np.where(season == all_ev_unique_seasons[0])
            total_season_points = all_ev_unique_seasons[1][idx]
            season_percentage = total_season_points / np.sum(all_ev_unique_seasons[1])
            seasons_array[season_ev_percentage_row, i] = season_percentage

    # Identify the percentage of each season in the data

    for i in range(seasons_array.shape[1]):
        season = seasons_array[s_label_row, i]
        if season in all_unique_seasons[0]:
            idx = np.where(season == all_unique_seasons[0])
            total_season_points = all_unique_seasons[1][idx]
            season_percentage = total_season_points / np.sum(all_unique_seasons[1])
            seasons_array[season_percentage_row, i] = season_percentage

    # Calculate the value in a single proportion

    seasons_array[relative_cons_row, :] = seasons_array[percentage_cons_row, :] / seasons_array[repetitions_row, :]
    current_season_indexes = seasons_array[total_cons_row, :] != 0
    current_season_indexes = np.logical_and(current_season_indexes, seasons_array[percentage_cons_row, :] >
                                            ignore_season_percentage_l1)

    # Calculate the penalty ratios

    final_seasons_array = seasons_array[:, current_season_indexes]
    penalty_1_ratio = final_seasons_array[percentage_cons_row, :] / final_seasons_array[season_ev_percentage_row, :]
    penalty_2_ratio = final_seasons_array[percentage_cons_row, :] / final_seasons_array[season_percentage_row, :]

    penalty_1_ratio[~np.isfinite(penalty_1_ratio)] = 1
    penalty_2_ratio[~np.isfinite(penalty_2_ratio)] = 1

    # Normalise the penalty ratios
    penalty_1_ratio = ((np.max(penalty_1_ratio) - np.min(penalty_1_ratio)) / np.min(penalty_1_ratio)) / 10
    penalty_2_ratio = ((np.max(penalty_2_ratio) - np.min(penalty_2_ratio)) / np.min(penalty_2_ratio)) / 10

    return penalty_1_ratio, penalty_2_ratio


def get_ev_proportion_features(box_features, dl_debug):
    """
    This function is used to get the penalties for EV proportion variation
    Parameters:
        box_features                (np.ndarray)            : Box features array
        dl_debug                    (Dict)                  : Debug dictionary
    Returns:
        ev_prop_frac_var            (np.ndarray)            : EV proportion variation w.r.t seasons
        ev_prop_frac_penalty        (float)                 : EV proportion variation penalty
    """

    s_label_data = dl_debug.get('s_label_data')
    unique_seasons = np.unique(s_label_data)
    ev_proportion = []
    season_proportion = []

    # Calculate the ev_proportion and the season_proportion

    for i in range(len(unique_seasons)):
        curr_s_ev_prop = np.sum(box_features[:, FeatureCol.S_LABEL] == unique_seasons[i]) / len(box_features)
        curr_s_prop = np.sum(s_label_data == unique_seasons[i]) / len(s_label_data.flatten())
        ev_proportion.append(curr_s_ev_prop)
        season_proportion.append(curr_s_prop)

    ev_proportion = np.asarray(ev_proportion)
    season_proportion = np.asarray(season_proportion)

    # Calculate the ev proportion variation from season to season

    ev_prop_frac = ev_proportion/season_proportion
    ev_prop_frac_var = np.var(ev_prop_frac)
    ev_prop_frac_penalty = np.max(ev_prop_frac) - np.min(ev_prop_frac)

    return ev_prop_frac_var, ev_prop_frac_penalty


def seasonality_penalty(box_features, final_boxes_detected, dl_debug):
    """
    Function to enforce seasonal penalty
    Parameters:
        box_features                    (np.ndarray)            : Box features
        final_boxes_detected            (np.ndarray)            : Final boxes detected
        dl_debug                        (Dict)                  : Debug dictionary
    Returns:
        seasonality_penalty_score       (float)                 : Seasonality penalty score
    """

    # Extract required variables
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')
    seasonality_penalty_weight_l1 = dl_debug.get('config').get('seasonality_penalty').get('seasonality_penalty_weight_l1')
    ev_var_thr = dl_debug.get('config').get('seasonality_penalty').get('ev_var_thr')
    ev_prop_penalty = dl_debug.get('config').get('seasonality_penalty').get('ev_prop_penalty')

    # Extract the required variables
    seasonality_penalty_score = 0
    detected_prts = box_features[box_features[:, FeatureCol.PRT_PRED] == 1]
    detected_prts = detected_prts[detected_prts[:, FeatureCol.TOTAL_EV_STRIKES] > 0]

    # Get the season sequences
    season_sequences = find_seq(detected_prts[:, FeatureCol.S_LABEL], min_seq_length=0)

    if len(season_sequences):
        penalty_1_ratio, penalty_2_ratio = \
            get_season_penalties(season_sequences, detected_prts, final_boxes_detected, dl_debug)

        # Calculate the initial seasonal penalty score
        seasonality_penalty_score = confidence_threshold * penalty_1_ratio \
                                    + confidence_threshold * penalty_2_ratio

    # Get EV proportion variations

    ev_prop_frac_var, ev_prop_frac_penalty = get_ev_proportion_features(box_features, dl_debug)

    # Calculate the final seasonality penalty

    seasonality_penalty_score = seasonality_penalty_weight_l1 * seasonality_penalty_score \
                                + ev_var_thr * np.fmin(ev_prop_frac_var, 1) \
                                + ev_prop_penalty * np.fmin(ev_prop_frac_penalty, 1)
    return seasonality_penalty_score


def update_prt_confidences(initial_prt_predictions, updated_prt_predictions, prt_confidences, box_features, dl_debug):
    """
    Function to update the partition confidences
    Parameters:
        initial_prt_predictions             (np.ndarray)            : Initial partition predictions
        updated_prt_predictions             (np.ndarray)            : Updated partition predictions
        prt_confidences                     (np.ndarray)            : Partition confidences
        box_features                        (np.ndarray)            : Box features
        dl_debug                            (Dict)                  : Debug object
    Returns:
        updated_prt_predictions             (np.ndarray)            : Updated partition predictions
    """

    # Extract the required variables
    similarity_default = dl_debug.get('config').get('similarity_default')
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')

    # Initialise the required variables

    ev_prts = np.where(initial_prt_predictions == 1)[0]
    new_prt_confidences = deepcopy(prt_confidences)

    # If a partition has no detection then update the detection based on the similarity with the neighbouring partitions

    for i in range(len(initial_prt_predictions)):
        if np.logical_and(initial_prt_predictions[i] == 0, updated_prt_predictions[i] == 1):
            prt_before = ev_prts < i
            prt_before_idx = None
            if np.sum(prt_before):
                prt_before_idx = np.where(prt_before)[-1]

            prt_after = ev_prts > i
            prt_after_idx = None
            if np.sum(prt_before):
                prt_after_idx = np.where(prt_after)[0]

            # Check the similarity with the previous partition

            similarity_before = similarity_default
            if prt_before_idx is not None:
                prt_before_idx = ev_prts[prt_before_idx][0]
                box_features_before = box_features[prt_before_idx]
                box_features_curr = box_features[i]
                similarity_before = dot(box_features_before, box_features_curr)/(norm(box_features_before)*norm(box_features_curr))

            # Check the similarity with the after partition

            similarity_after = similarity_default
            if prt_after_idx is not None:
                prt_after_idx = ev_prts[prt_after_idx][0]
                box_features_after = box_features[prt_after_idx]
                box_features_curr = box_features[i]
                similarity_after = dot(box_features_after, box_features_curr) / (norm(box_features_after) * norm(box_features_curr))

            # Get the similarity score

            similarity = (similarity_before + similarity_after)/2
            initial_prt_predictions[i] = 1
            ev_prts = np.where(initial_prt_predictions == 1)[0]

            # Update the partition prediction

            new_prt_confidences[i] = new_prt_confidences[i] + confidence_threshold*similarity

    updated_prt_confidences = np.fmin(new_prt_confidences, 1)

    return updated_prt_confidences


def get_detection(box_features, final_boxes_detected, dl_debug):
    """
    Function to get the detection for L1
    Parameters:
        box_features                        (np.ndarray)            : Box features
        final_boxes_detected                (np.ndarray)            : Final boxes detected
        dl_debug                            (Dict)                  : Debug dictionary
    Returns:
        detection                           (int)                   : Detection status
        final_confidence                    (float)                 : Final confidence
        seasonality_penalty_score           (float)                 : Seasonality Penalty score
    """

    # Extract required variables
    a_value = dl_debug.get('config').get('a_value')
    r_value_2 = dl_debug.get('config').get('r_value_2')
    penalty_thr_1 = dl_debug.get('config').get('penalty_thr_1')
    penalty_thr_2 = dl_debug.get('config').get('penalty_thr_2')
    penalty_thr_3 = dl_debug.get('config').get('penalty_thr_3')
    penalty_thr_4 = dl_debug.get('config').get('penalty_thr_4')
    penalty_thr_5 = dl_debug.get('config').get('penalty_thr_5')
    penalty_thr_6 = dl_debug.get('config').get('penalty_thr_6')
    penalty_thr_7 = dl_debug.get('config').get('penalty_thr_7')
    final_weight_1 = dl_debug.get('config').get('final_weight_1')
    final_weight_2 = dl_debug.get('config').get('final_weight_2')
    penalty_weight_l1 = dl_debug.get('config').get('penalty_weight_l1')
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')
    seasonality_penalty_weight_l1 = dl_debug.get('config').get('seasonality_penalty_weight_l1')

    prt_predictions = box_features[:, FeatureCol.PRT_PRED]
    prt_confidences = box_features[:, FeatureCol.PRT_CONFIDENCE]

    # Update the partition confidences based on similarity

    initial_prt_predictions = dl_debug.get('predictions_bool_l1')
    updated_prt_predictions = dl_debug.get('updated_partition_predictions_l1')
    updated_prt_confidences = update_prt_confidences(initial_prt_predictions, updated_prt_predictions, prt_confidences,
                                                     box_features, dl_debug)
    dl_debug['del_l1_new_confidences'] = updated_prt_confidences.reshape(-1, 1)

    detected_prt_bool = prt_predictions == 1

    # Extract the features

    average_confidence = np.mean(updated_prt_confidences[detected_prt_bool])
    median_confidence = np.median(updated_prt_confidences[detected_prt_bool])
    max_confidence = np.max(updated_prt_confidences[detected_prt_bool])
    average_confidence_all = np.mean(updated_prt_confidences)
    median_confidence_all = np.median(updated_prt_confidences)
    max_confidence_all = np.max(updated_prt_confidences)

    # Get a final tentative confidence based on the main features

    final_confidence = (average_confidence + median_confidence + max_confidence)/3
    final_confidence_all = (average_confidence_all + median_confidence_all + max_confidence_all)/3
    final_confidence_combined = (final_weight_1 * final_confidence + final_weight_2 * final_confidence_all)

    # Get the overall changes in the features as a means of percentage

    detected_prts = box_features[box_features[:, FeatureCol.PRT_PRED] == 1]
    detected_prts = detected_prts[detected_prts[:, FeatureCol.TOTAL_EV_STRIKES] > 0]
    percentage_changes = abs(detected_prts[:-1, FeatureCol.MEAN_BOX_ENERGY:] - detected_prts[1:, FeatureCol.MEAN_BOX_ENERGY:]) / detected_prts[:-1, FeatureCol.MEAN_BOX_ENERGY:]
    overall_percentage_changes = np.nanpercentile(percentage_changes, q=50, axis=0)

    # 0-Mean, 1-Min, 2-Max, 3-AUC, 4-Duration, 5-EV_strikes, 6-Charging_freq, 7-Total_consumption

    penalty_score = penalty_thr_1*overall_percentage_changes[0] \
                    + penalty_thr_2*overall_percentage_changes[1] \
                    + penalty_thr_3*overall_percentage_changes[2]\
                    + penalty_thr_4*overall_percentage_changes[3]\
                    + penalty_thr_5*overall_percentage_changes[4]\
                    + penalty_thr_6*overall_percentage_changes[5]\
                    + penalty_thr_7*overall_percentage_changes[6]

    # Add the penalty to get the updated score

    final_confidence = np.fmax(final_confidence_combined - penalty_weight_l1*penalty_score, 0)

    # Update the confidences based on the weights (higher weights for a nearest partition)

    length = len(prt_predictions)
    weights = np.asarray([a_value * r_value_2 ** i for i in range(length)]).reshape(-1, 1)[::-1]
    weighted_confidences = np.mean(updated_prt_confidences.reshape(-1, 1)*weights)

    final_confidence = np.fmax(final_weight_1 * final_confidence + final_weight_2 * weighted_confidences, 0)

    # Perform seasonality checks and get a penalty score

    seasonality_penalty_score = seasonality_penalty(box_features, final_boxes_detected, dl_debug)

    # Perform seasonal penalty

    final_confidence = np.fmax(final_confidence - seasonality_penalty_weight_l1*seasonality_penalty_score, 0)

    if final_confidence >= confidence_threshold:
        detection = 1
    else:
        detection = 0

    return detection, final_confidence, seasonality_penalty_score


def get_final_confidence_score(dl_confidence, low_density, ml_debug, dl_debug):
    """
    Get the final combined confidence score
    Parameters:
        dl_confidence               (float)         : Deep learning model confidence
        low_density                 (Boolean)       : Low density user flag
        ml_debug                    (Dict)          : Machine learning debug object
        dl_debug                    (Dict)          : Deep learning debug object
    Returns:
        combined_hld                (int)           : Combined hld
        combined_conf               (float)         : Combined confidence
        combined_charger_type       (string)        : Combined charger type
    """

    # Get the L1 confidences

    conf_thr_1 = dl_debug.get('config').get('l1_detection_configs').get('conf_thr_1')
    conf_thr_2 = dl_debug.get('config').get('l1_detection_configs').get('conf_thr_2')
    conf_thr_3 = dl_debug.get('config').get('l1_detection_configs').get('conf_thr_3')
    conf_thr_4 = dl_debug.get('config').get('l1_detection_configs').get('conf_thr_4')
    conf_thr_5 = dl_debug.get('config').get('l1_detection_configs').get('conf_thr_5')
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')
    low_density_ml_weight = dl_debug.get('config').get('l1_detection_configs').get('low_density_ml_weight')
    low_density_dl_weight = dl_debug.get('config').get('l1_detection_configs').get('low_density_dl_weight')

    ml_l1_detection_info = ml_debug.get('l1')
    ml_confidence = ml_l1_detection_info.get('ev_probability')

    # Get the final L1 confidence after combining the ML & DL confidences

    if conf_thr_3 <= ml_confidence <= conf_thr_4 and dl_confidence >= conf_thr_3:
        combined_conf = conf_thr_3 * dl_confidence + conf_thr_3 * ml_confidence
    elif conf_thr_4 >= ml_confidence >= conf_thr_3 and dl_confidence < confidence_threshold:
        combined_conf = conf_thr_4 * dl_confidence + conf_thr_2 * ml_confidence
    elif conf_thr_2 <= ml_confidence < conf_thr_3 and dl_confidence >= confidence_threshold:
        combined_conf = conf_thr_5 * dl_confidence + conf_thr_1 * ml_confidence
    else:
        combined_conf = ml_confidence

    # If it's a low density user then update the score accordingly

    if low_density:
        combined_conf = low_density_ml_weight * ml_confidence + low_density_dl_weight * dl_confidence

    # Update the final values

    if combined_conf >= confidence_threshold:
        combined_hld = 1
        combined_charger_type = 'L1'
    else:
        combined_hld = 0
        combined_charger_type = 'None'

    return combined_hld, combined_conf, combined_charger_type


def get_ev_l1_user_detection(final_boxes_detected, dl_debug, ml_debug, logger_base):
    """
    Function to perform EV L1 detection
    Parameters:
        final_boxes_detected                (np.ndarray)        : Final boxes detected
        dl_debug                            (Dict)              : Debug dictionary
        ml_debug                            (Dict)              : Machine learning debugger
        logger_base                          (logger)           : Logger passed
    Returns:
        final_boxes_detected                (np.ndarray)        : Final boxes detected
        debug                               (Dict)              : Debug dictionary
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('get_ev_l1_user_detection')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Initialise the required variables

    prt_predictions = dl_debug.get('final_partition_predictions_l1')
    prt_confidences = dl_debug.get('prediction_confidences_l1')

    # Identifying users with detection in 1 or 2 partitions detection before the recent 6 months

    detection, dl_confidence, low_density = low_detection_density(prt_predictions, prt_confidences, dl_debug)

    logger.info('L1 : Low density detected user | %s ', low_density)

    # ------------------------------------------- USER LEVEL DETECTION ------------------------------------------------

    seasonality_penalty_score = 0

    if not low_density:

        # Extract the box features

        box_features = extract_box_features(final_boxes_detected, dl_debug, 'L1')

        # Get the final detection values

        detection, dl_confidence, seasonality_penalty_score = get_detection(box_features, final_boxes_detected, dl_debug)

    logger.info('DL L1 : Deep learning model confidence | %s ', dl_confidence)
    logger.info('DL L1 : Seasonality penalty score | %s ', seasonality_penalty_score)

    dl_debug['low_density_l1'] = low_density
    dl_debug['user_detection_l1'] = detection
    dl_debug['user_confidence_l1'] = np.round(dl_confidence, 2)

    combined_hld, combined_conf, combined_charger_type = get_final_confidence_score(dl_confidence, low_density, ml_debug,
                                                                                    dl_debug)

    logger.info('DL L1 : Combined HLD | %s ', combined_hld)
    logger.info('DL L1 : Combined confidence | %s ', combined_conf)
    logger.info('DL L1 : Combined charger type | %s ', combined_charger_type)

    dl_debug['combined_hld'] = combined_hld
    dl_debug['combined_charger_type'] = combined_charger_type

    if combined_hld == 0:
        final_boxes_detected[final_boxes_detected > 0] = 0
    else:

        # If there is detection for L1 then update the confidence for L1 confidence else leave it as L2 confidence
        dl_debug['combined_conf'] = combined_conf

    return final_boxes_detected, dl_debug
