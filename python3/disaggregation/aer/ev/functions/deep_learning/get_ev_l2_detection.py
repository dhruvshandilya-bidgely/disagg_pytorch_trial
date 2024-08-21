"""
Author - Sahana M
Date - 15-Nov-2023
Module to get the User detection
"""

# import python packages
import logging
import numpy as np

# import modules from within the project
from python3.utils.maths_utils.find_seq import find_seq
from python3.disaggregation.aer.ev.functions.deep_learning.init_dl_config import FeatureCol
from python3.disaggregation.aer.ev.functions.deep_learning.deeplearning_utils import extract_box_features


def low_detection_density(prt_predictions, prt_confidences, dl_debug):
    """
    Function to identify a low density user
    Parameters:
        prt_predictions                 (np.ndarray)        : Partition predictions array
        prt_confidences                 (np.ndarray)        : Partition confidences
        dl_debug                        (Dict)              : Deep learning debug object
    Returns:
        detection                       (int)               : Detection status
        confidence                      (float)             : Confidence value
        low_density                     (Boolean)           : Low density True/False
    """

    # Extract the required variables
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')
    recent_ev_partitions = dl_debug.get('config').get('recent_ev_partitions')
    recent_ev_partitions_thr = dl_debug.get('config').get('recent_ev_partitions_thr')

    low_predictions = np.sum(prt_predictions) <= 1
    weighted_confidence = np.mean(prt_confidences)

    # Identify definite low density users

    if np.sum(prt_predictions) == 0:
        confidence = np.mean(prt_confidences)
        weighted_confidence = confidence

    elif low_predictions:
        detected_prt = prt_predictions == 1
        weighted_confidence = [detected_prt] * prt_confidences[detected_prt]
        weighted_confidence = np.mean(weighted_confidence)

    confidence = weighted_confidence
    if weighted_confidence >= confidence_threshold:
        detection = 1
        low_density = False
    else:
        detection = 0
        low_density = True

    # Check for recent EV users which can be confused for Low density users

    if np.sum(prt_predictions[-recent_ev_partitions:]) >= recent_ev_partitions_thr and \
            np.sum(prt_predictions[:-recent_ev_partitions]) == 0:
        after_ev_confidence = np.mean(prt_confidences[prt_predictions == 1])
        if after_ev_confidence >= confidence_threshold:
            detection = 1
            low_density = False
            confidence = after_ev_confidence

    # Check for inconsistent detections

    updated_partition_confidences = dl_debug.get('updated_partition_confidences')
    if updated_partition_confidences is not None:
        moving_average_predicted = updated_partition_confidences[prt_predictions == 1]
        moving_average_not_predicted = updated_partition_confidences[prt_predictions == 0]

        detected_partitions = (prt_predictions == 1).sum()/len(prt_predictions)
        undetected_partitions = (prt_predictions == 0).sum()/len(prt_predictions)

        detected_confidences = np.mean(moving_average_predicted)
        undetected_confidences = np.mean(moving_average_not_predicted)

        final_confidence = detected_partitions*detected_confidences + undetected_partitions*undetected_confidences

        if final_confidence >= confidence_threshold:
            detection = 1
            low_density = False
            confidence = final_confidence

    return detection, confidence, low_density


def get_seasonality_penalty_values(final_boxes_detected, detected_prts, dl_debug):
    """
    This function is used to get the penalty values for a possible seasonal FP user
    Parameters:
        final_boxes_detected                (np.ndarray)        : Final boxes detected
        detected_prts                       (np.ndarray)        : Detected partitions
        dl_debug                            (Dict)              : Debug dictionary
    Returns:
        penalty_1_ratio                     (float)             : Penalty 1 value
        penalty_2_ratio                     (float)             : Penalty 2 value
    """

    # Define the local columns
    s_labels = dl_debug.get('config').get('seasonality_penalty').get('s_labels')
    s_label_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('s_label_row')
    avg_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('avg_cons_row')
    ignore_season_percentage = dl_debug.get('config').get('seasonality_penalty').get('ignore_season_percentage')
    total_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('total_cons_row')
    repetitions_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('repetitions_row')
    relative_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('relative_cons_row')
    percentage_cons_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('percentage_cons_row')
    season_percentage_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('season_percentage_row')
    season_ev_percentage_row = dl_debug.get('config').get('seasonality_penalty').get('season_cols').get('season_ev_percentage_row')

    # Get the number of times a season is repeated

    season_sequences = find_seq(detected_prts[:, FeatureCol.S_LABEL], min_seq_length=0)
    unique_seasons = np.unique(season_sequences[:, 0], return_counts=True)

    seasons_array = np.asarray(s_labels)
    repetitions_arr = np.zeros(shape=(7, 5))
    for i in range(len(seasons_array)):
        if seasons_array[i] in unique_seasons[0]:
            unique_seasons_index = np.where(seasons_array[i] == unique_seasons[0])[0][0]
            repetitions = unique_seasons[1][unique_seasons_index]
            repetitions_arr[0, i] = repetitions
    seasons_array = np.r_[seasons_array.reshape(1, -1), repetitions_arr]

    # Get the average and total consumption for each season

    for i in range(seasons_array.shape[1]):
        season = seasons_array[s_label_row, i]
        season_partitions = detected_prts[detected_prts[:, FeatureCol.S_LABEL] == season]
        if len(season_partitions):
            season_prt_avg_cons = np.percentile(season_partitions[:, FeatureCol.TOTAL_CONSUMPTION], q=50)
            season_prt_total_cons = np.nansum(season_partitions[:, FeatureCol.TOTAL_CONSUMPTION])
            seasons_array[avg_cons_row, i] = season_prt_avg_cons
            seasons_array[total_cons_row, i] = season_prt_total_cons

    # get the percentage consumption for each season
    seasons_array[percentage_cons_row, :] = seasons_array[total_cons_row, :] / np.sum(seasons_array[total_cons_row, :])

    # Extract the season labels according to the seasons label data

    ev_s_label_data = dl_debug.get('s_label_data')
    final_boxes_detected_idx = final_boxes_detected > 0
    final_boxes_detected_idx = final_boxes_detected_idx.flatten()
    ev_s_label_data = ev_s_label_data.flatten()
    ev_s_label_data = ev_s_label_data[final_boxes_detected_idx]
    all_ev_unique_seasons = np.unique(ev_s_label_data, return_counts=True)

    s_label_data = dl_debug.get('s_label_data')
    s_label_data = s_label_data.flatten()
    all_unique_seasons = np.unique(s_label_data, return_counts=True)

    # For each season identify the proportion of the season

    for i in range(seasons_array.shape[1]):
        season = seasons_array[s_label_row, i]
        if season in all_ev_unique_seasons[0]:
            idx = np.where(season == all_ev_unique_seasons[0])
            total_season_points = all_ev_unique_seasons[1][idx]
            season_percentage = total_season_points / np.sum(all_ev_unique_seasons[1])
            seasons_array[season_ev_percentage_row, i] = season_percentage

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
                                            ignore_season_percentage)
    final_seasons_array = seasons_array[:, current_season_indexes]

    penalty_1_ratio = penalty_2_ratio = 0

    # Assign the penalty scores

    if final_seasons_array.shape[1]:
        penalty_1_ratio = final_seasons_array[percentage_cons_row, :] / final_seasons_array[season_ev_percentage_row, :]
        penalty_2_ratio = final_seasons_array[percentage_cons_row, :] / final_seasons_array[season_percentage_row, :]
        penalty_1_ratio = ((np.max(penalty_1_ratio) - np.min(penalty_1_ratio)) / np.min(penalty_1_ratio)) / 10
        penalty_2_ratio = ((np.max(penalty_2_ratio) - np.min(penalty_2_ratio)) / np.min(penalty_2_ratio)) / 10

    return penalty_1_ratio, penalty_2_ratio


def seasonality_penalty_for_l2(box_features, final_boxes_detected, dl_debug):
    """
    Function to assign a seasonality penalty for seasonal users
    Parameters:
        box_features                    (np.ndarray)        : Box features
        final_boxes_detected            (np.ndarray)        : Final boxes detected
        dl_debug                        (Dict)              : Debug dictionary
    Returns:
        seasonality_penalty_score       (float)             : Seasonality penalty score
    """

    # Extract required variables
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')

    # Define the local columns

    seasonality_penalty_score_default = dl_debug.get('config').get('seasonality_penalty').get('seasonality_penalty_score_default')

    # Initialise required variables

    seasonality_penalty_score = 0
    detected_prts = box_features[box_features[:, FeatureCol.PRT_PRED] == 1]
    detected_prts = detected_prts[detected_prts[:, FeatureCol.TOTAL_EV_STRIKES] > 0]

    if len(detected_prts):

        penalty_1_ratio, penalty_2_ratio = get_seasonality_penalty_values(final_boxes_detected, detected_prts, dl_debug)

        # Arrive at the final seasonal penalty score
        seasonality_penalty_score = np.fmin(confidence_threshold*penalty_1_ratio
                                            + confidence_threshold*penalty_2_ratio,
                                            seasonality_penalty_score_default)

    return seasonality_penalty_score


def get_detection(box_features, final_boxes_detected, dl_debug):
    """
    Get the Final EV L2 detection status
    Parameters:
        box_features                    (np.ndarray)            : Box features array
        final_boxes_detected            (np.ndarray)            : Final boxes detected
        dl_debug                        (Dict)                  : Deep learning debug dictionary
    Returns:
        detection                       (int)                   : Detection status
        final_confidence                (float)                 : Final confidence
        seasonality_penalty_score       (float)                 : Seasonality penalty
    """

    # Extract the required variables
    penalty_thr_1 = dl_debug.get('config').get('penalty_thr_1')
    penalty_thr_2 = dl_debug.get('config').get('penalty_thr_2')
    penalty_thr_3 = dl_debug.get('config').get('penalty_thr_3')
    penalty_thr_4 = dl_debug.get('config').get('penalty_thr_4')
    penalty_thr_5 = dl_debug.get('config').get('penalty_thr_5')
    penalty_thr_6 = dl_debug.get('config').get('penalty_thr_6')
    penalty_thr_7 = dl_debug.get('config').get('penalty_thr_7')
    penalty_weight = dl_debug.get('config').get('penalty_weight')
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')
    seasonality_penalty_weight = dl_debug.get('config').get('seasonality_penalty_weight')

    # Extract the required features for confidence scoring

    prt_predictions = box_features[:, FeatureCol.PRT_PRED]
    prt_confidences = box_features[:, FeatureCol.PRT_CONFIDENCE]
    detected_prt_bool = prt_predictions == 1
    average_confidence = np.mean(prt_confidences[detected_prt_bool])
    median_confidence = np.median(prt_confidences[detected_prt_bool])
    min_confidence = np.min(prt_confidences[detected_prt_bool])
    max_confidence = np.max(prt_confidences[detected_prt_bool])

    # Get the tentative confidence
    final_confidence = (average_confidence + median_confidence + min_confidence + max_confidence)/4

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

    # Perform penalty
    final_confidence = final_confidence - penalty_weight*penalty_score

    # seasonality checks
    seasonality_penalty_score = seasonality_penalty_for_l2(box_features, final_boxes_detected, dl_debug)

    # perform seasonal penalty
    final_confidence = np.fmax(final_confidence - seasonality_penalty_weight*seasonality_penalty_score, 0)

    detection = 0
    if final_confidence >= confidence_threshold:
        detection = 1

    return detection, final_confidence, seasonality_penalty_score


def get_final_confidence_score(dl_confidence, seasonality_penalty_score, low_density, ml_debug, dl_debug):
    """
    Combine the Machine learning and Deep learning confidences
    Parameters:
        dl_confidence                   (float)         : Deep learning confidence
        seasonality_penalty_score       (float)         : Seasonal penalty score
        low_density                     (Boolean)       : Low density user boolean
        ml_debug                        (Dict)          : ML debug object
        dl_debug                        (Dict)          : DL debug object
    Returns:
        combined_hld                    (int)           : Combined hld
        combined_conf                   (float)         : Combined confidence
        combined_charger_type           (string)        : Combined charger type
    """

    # Extract the required variables
    confidence_threshold = dl_debug.get('config').get('confidence_threshold')
    constant = dl_debug.get('config').get('l2_detection_configs').get('constant')
    ml_confidence_weight = dl_debug.get('config').get('l2_detection_configs').get('ml_confidence_weight')
    dl_confidence_weight = dl_debug.get('config').get('l2_detection_configs').get('dl_confidence_weight')
    low_density_ml_weight = dl_debug.get('config').get('l2_detection_configs').get('low_density_ml_weight')
    seasonal_fp_confidence = dl_debug.get('config').get('l2_detection_configs').get('seasonal_fp_confidence')
    low_density_comb_weight = dl_debug.get('config').get('l2_detection_configs').get('low_density_comb_weight')
    seasonal_penalty_weight = dl_debug.get('config').get('l2_detection_configs').get('seasonal_penalty_weight')
    seasonal_fps_ml_confidences = dl_debug.get('config').get('l2_detection_configs').get('seasonal_fps_ml_confidences')

    # Get the Machine learning values

    if ml_debug.get('disagg_mode') == 'mtd':
        ml_confidence = ml_debug.get('ev_probability')
    else:
        ml_l2_detection_info = ml_debug.get('l2')
        ml_confidence = ml_l2_detection_info.get('ev_probability')

    # Get the combined EV L2 confidences

    combined_conf = (ml_confidence_weight * ml_confidence
                     + dl_confidence_weight * dl_confidence
                     - seasonal_penalty_weight * seasonality_penalty_score) - constant

    combined_conf = np.round(1 / (1 + np.exp(-combined_conf)), 2)

    # Resort to default value if its a seasonal FP user

    if ml_debug.get('seasonality_fp') or ml_confidence in seasonal_fps_ml_confidences:
        combined_conf = seasonal_fp_confidence

    # Low density users handling

    if low_density:
        combined_conf = (ml_confidence + dl_confidence + combined_conf) / 3

    if low_density and dl_debug.get('config').get('region') == 'EU':
        combined_conf = low_density_ml_weight * ml_confidence + low_density_comb_weight * combined_conf

    # Assigning the confidence values

    if combined_conf >= confidence_threshold:
        combined_hld = 1
        combined_charger_type = 'L2'
    else:
        combined_hld = 0
        combined_charger_type = 'None'

    if dl_confidence < confidence_threshold and ml_confidence < confidence_threshold:
        combined_hld = 0
        combined_conf = np.round((ml_confidence + dl_confidence) / 2, 2)
        combined_charger_type = 'None'

    return combined_hld, combined_conf, combined_charger_type


def get_ev_l2_detection(final_boxes_detected, dl_debug, ml_debug, logger_base):
    """
    Function to get the EV L2 detection
    Parameters:
        final_boxes_detected                (np.ndarray)        : Final boxes detected
        dl_debug                            (Dict)              : Deep learning debug object
        ml_debug                            (Dict)              : Machine learning debug object
        logger_base                         (logger)                : Logger passed

    Returns:
        final_boxes_detected                (np.ndarray)        : Final boxes detected
        dl_debug                            (Dict)              : Deep learning debug object
    """

    # Initialise the logger
    logger_local = logger_base.get('logger').getChild('deep_learning_exp')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    prt_predictions = dl_debug.get('final_partition_predictions')
    prt_confidences = dl_debug.get('prediction_confidences')

    # Identifying users with detection in 1 or 2 partitions detection before the recent 6 months

    detection, dl_confidence, low_density = low_detection_density(prt_predictions, prt_confidences, dl_debug)

    logger.info('L2 : Low density detected user | %s ', low_density)

    # ------------------------------------------- USER LEVEL DETECTION ------------------------------------------------

    seasonality_penalty_score = 0

    if not low_density:

        # Extract the box features

        box_features = extract_box_features(final_boxes_detected, dl_debug)

        # Get EV L2 detection

        detection, dl_confidence, seasonality_penalty_score = get_detection(box_features, final_boxes_detected, dl_debug)

    logger.info('DL L2 : Deep learning model confidence | %s ', dl_confidence)
    logger.info('DL L2 : Seasonality penalty score | %s ', seasonality_penalty_score)

    dl_debug['low_density_l2'] = low_density
    dl_debug['user_detection_l2'] = detection
    dl_debug['user_confidence_l2'] = np.round(dl_confidence, 2)

    # Get the combined hld, confidences and charger type

    combined_hld, combined_conf, combined_charger_type = \
        get_final_confidence_score(dl_confidence, seasonality_penalty_score, low_density, ml_debug, dl_debug)

    logger.info('DL L2 : Combined HLD | %s ', combined_hld)
    logger.info('DL L2 : Combined confidence | %s ', combined_conf)
    logger.info('DL L2 : Combined charger type | %s ', combined_charger_type)

    dl_debug['combined_hld'] = combined_hld
    dl_debug['combined_conf'] = combined_conf
    dl_debug['combined_charger_type'] = combined_charger_type
    dl_debug['seasonality_penalty_score_l2'] = seasonality_penalty_score

    if combined_hld == 0:
        final_boxes_detected[final_boxes_detected > 0] = 0

    return final_boxes_detected, dl_debug
