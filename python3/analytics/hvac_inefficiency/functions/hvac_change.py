"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import python packages

import copy
import logging
import numpy as np

# Import functions from within the project
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params
from python3.analytics.hvac_inefficiency.utils.weighted_kl_div import get_divergence_score
from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile


def predict_change_probab(feature_vector, model_dict):

    """
        This function predicts HVAC change along with probability

        Parameters:
            feature_vector          (numpy.ndarray)  numpy array containing input feature
            model_dict              (dict)           dictionary normalisation parameters
        Returns:
            probability             (float)          float containing probability of hvac change
            prediction              (float)          float containing hvac change prediction
            prediction_dictionary   (dict)           Dictionary containing intermediate prediction vals
    """

    static_params = hvac_static_params()

    svm_model = model_dict.get('svm')
    dt_model = model_dict.get('dt')
    log_reg_model = model_dict.get('log_reg')
    normalisation_dict = model_dict.get('normalisation_dict')
    feature_vector_raw = copy.deepcopy(feature_vector)

    change_direction = -1 * np.sign(feature_vector[0])

    probability = static_params.get('ineff').get('init_prob')
    prediction = static_params.get('ineff').get('init_pred')

    if (svm_model is None) | (dt_model is None) | (log_reg_model is None) | (normalisation_dict is None):
        prediction_dictionary = \
            {
                'svm_probab': static_params.get('ineff').get('init_prob'),
                'dt_probab': static_params.get('ineff').get('init_prob'),
                'svm_prediction': static_params.get('ineff').get('init_pred'),
                'dt_prediction': static_params.get('ineff').get('init_pred'),
                'change_direction': 0,
                'raw_feature': feature_vector_raw
            }

        return probability, prediction, prediction_dictionary

    min_val = normalisation_dict.get('previous_year_divergence').get('min')
    max_val = normalisation_dict.get('previous_year_divergence').get('max')
    feature_vector[0] = (feature_vector[0] - min_val) / (max_val - min_val)

    min_val = normalisation_dict.get('current_fcc').get('min')
    max_val = normalisation_dict.get('current_fcc').get('max')
    feature_vector[1] = (feature_vector[1] - min_val) / (max_val - min_val)

    min_val = normalisation_dict.get('current_pre_sat').get('min')
    max_val = normalisation_dict.get('current_pre_sat').get('max')
    feature_vector[2] = (feature_vector[2] - min_val) / (max_val - min_val)

    feature_vector = np.array([feature_vector])

    svm_probab = svm_model.predict_proba(feature_vector)[:, 1]
    dt_probab = dt_model.predict_proba(feature_vector)[:, 1]

    log_reg_feature = np.c_[dt_probab, svm_probab]
    prediction = log_reg_model.predict(log_reg_feature)
    probability = log_reg_model.predict_proba(log_reg_feature)[:, 1]

    if prediction[0] == 0:
        change_direction = 0

    prediction_dictionary =\
        {
            'svm_probab': svm_probab,
            'dt_probab': dt_probab,
            'svm_prediction': svm_model.predict(feature_vector),
            'dt_prediction': dt_model.predict(feature_vector),
            'svm_probability': svm_model.predict_proba(feature_vector),
            'dt_probability': dt_model.predict_proba(feature_vector),
            'feature_vector': feature_vector,
            'raw_feature': feature_vector_raw,
            'change_direction': change_direction
        }

    return probability, prediction, prediction_dictionary


def detect_hvac_change(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device):

    """
        This function estimates hvac change

        Parameters:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
            logger_pass                         (object)        logger object
            device                              (str)           string indicating device, either AC or SH
        Returns:
            input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
            output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    static_params = hvac_static_params()

    logger_local = logger_pass.get("logger").getChild("detect_hvac_change")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Starting App Change | {}'.format(device))

    # Fetch hvac detection

    hsm_dict = input_hvac_inefficiency_object.get('hsm_information')

    col_map = hsm_dict.get('col_map')
    hsm_basic = hsm_dict.get('basic')
    time_col = col_map.get('hsm_time')
    fcc_col = col_map.get('{}_fcc'.format(device))
    pre_sat_col = col_map.get('{}_pre_sat_frac'.format(device))
    duty_cycle_dict = hsm_dict.get('dc_relation')

    valid_hsm = input_hvac_inefficiency_object.get(device, dict({})).get('valid_hsm_time')
    previous_year = valid_hsm.get('previous_year')
    return_dictionary = {'previous_year': previous_year}

    # If enough history not found then exit

    if previous_year is None:
        return_dictionary.update({'reason': 'Last year information are not available'})
        output_hvac_inefficiency_object[device]['app_change'] = return_dictionary
        return input_hvac_inefficiency_object, output_hvac_inefficiency_object

    # Preparing this years data
    cycling_debug_dictionary = output_hvac_inefficiency_object.get(device, {}).get('cycling_debug_dictionary', {})

    current_year_fcc = cycling_debug_dictionary.get('full_cycle_consumption')
    current_year_pre_sat = cycling_debug_dictionary.get('pre_saturation_fraction')
    current_year_dc = cycling_debug_dictionary.get('duty_cycle_relationship')

    # Preparing data for past years

    valid_idx = hsm_basic[:, time_col] == previous_year
    previous_year_fcc = hsm_basic[valid_idx, fcc_col][0]
    previous_year_pre_sat = hsm_basic[valid_idx, pre_sat_col][0]

    # Get Duty Cycle relations

    previous_year = int(previous_year)
    previous_duty_cycle = duty_cycle_dict.get(previous_year).get('{}_duty_cycle'.format(device))
    previous_temp = duty_cycle_dict.get(previous_year).get('{}_temp'.format(device))
    previous_dc_relation = np.c_[previous_temp, previous_duty_cycle]

    # Preparing Feature vector for model

    if device == 'ac':
        weight = static_params.get('ineff').get('hvac_change_wt')
    else:
        weight = -1 * static_params.get('ineff').get('hvac_change_wt')

    # Scaling duty cycle with new FCC

    scaled_previous_dc_relation = copy.deepcopy(previous_dc_relation)

    scaled_previous_dc_relation[:, 1] = \
        (scaled_previous_dc_relation[:, 1] * previous_year_fcc) / current_year_fcc

    current_divergence, length = get_divergence_score(scaled_previous_dc_relation, current_year_dc, weight=weight)

    # Filtering low duty cycle in last year

    if super_percentile(scaled_previous_dc_relation[:, 1], static_params.get('ineff').get('degrade_high_pct') * 100) <= static_params.get('ineff').get('low_duty_cycle_lim'):
        logger.debug('Setting divergence to NaN since low DC in previous year |')
        current_divergence = np.nan

    upper_cap_length = static_params.get('ineff').get('degrade_up_cap_len')
    lower_cap_length = static_params.get('ineff').get('degrade_low_cap_len')
    if length >= upper_cap_length:
        ratio = upper_cap_length / length
        current_divergence *= ratio

    elif length <= lower_cap_length:
        current_divergence = np.nan

    current_fcc = current_year_fcc / previous_year_fcc
    current_pre_sat = current_year_pre_sat - previous_year_pre_sat

    feature_vector = [current_divergence, current_fcc, current_pre_sat]

    model_dict = input_hvac_inefficiency_object.get('models', {}).get('app_change', {})

    if (np.isnan(feature_vector).sum() > 0) | (np.isinf(feature_vector).sum() > 0):
        logger.info('HVAC Change NaN or Inf found in features | ')
        probability = -1
        prediction = -1
        prediction_dictionary = dict({'raw_feature': np.round(feature_vector, 3)})
    else:
        probability, prediction, prediction_dictionary = predict_change_probab(feature_vector, model_dict)
        logger.info('HVAC Change probability={}, final output={}'.format(np.round(probability, 2), prediction))

    # Post processing code

    svm_prediction = prediction_dictionary.get('svm_prediction', 0)
    dt_prediction = prediction_dictionary.get('dt_prediction', 0)

    updated_probability = copy.deepcopy(probability)
    updated_prediction = svm_prediction * dt_prediction

    if (length > lower_cap_length) & (length <= static_params.get('ineff').get('low_len_modify')):
        updated_probability = updated_probability * static_params.get('ineff').get('low_len_modify_factor')
    elif (length > static_params.get('ineff').get('low_len_modify')) & (length <= static_params.get('ineff').get('high_len_modify')):
        updated_probability = updated_probability * static_params.get('ineff').get('high_len_modify_factor')

    if updated_probability < static_params.get('ineff').get('hvac_change_low_prob_limit'):
        updated_prediction = 0

    # End Post processing code

    debug_dictionary = {
        'old_probability': probability,
        'old_prediction': prediction,
        'probability': updated_probability,
        'app_change': updated_prediction,
        'previous_dc_relation': scaled_previous_dc_relation,
        'current_dc_relations': current_year_dc,
        'current_fcc': current_year_fcc,
        'previous_fcc': previous_year_fcc,
    }

    return_dictionary.update(debug_dictionary)
    return_dictionary.update(prediction_dictionary)

    output_hvac_inefficiency_object[device]['app_change'] = return_dictionary

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
