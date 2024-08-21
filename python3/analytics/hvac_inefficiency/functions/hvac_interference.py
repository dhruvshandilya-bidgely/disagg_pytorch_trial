"""
Author - Anand Kumar Singh
Date - 26th June 2021
Call the HVAC inefficiency module and get output
"""

# Import python packages

import logging
import numpy as np

from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def detect_hvac_interference(input_hvac_inefficiency_object, output_hvac_inefficiency_object, logger_pass, device):

    """
        This function detect if hvac is used in office hours

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

    logger_local = logger_pass.get("logger").getChild("detect_hvac_interference")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    logger.info('Starting HVAC interference detection | {}'.format(device))

    if device == 'sh':
        reason = 'not valid for SH'
        return_dictionary = {'reason': reason}
        output_hvac_inefficiency_object[device]['interfering_hvac'] = return_dictionary
        logger.info('{} | {}'.format(reason, device))
        return input_hvac_inefficiency_object, output_hvac_inefficiency_object

    office_goer_winter = output_hvac_inefficiency_object.get('office_goer_probab').get('winter')
    office_goer_summer = output_hvac_inefficiency_object.get('office_goer_probab').get('summer')
    office_goer_transition = output_hvac_inefficiency_object.get('office_goer_probab').get('transition')

    if (office_goer_transition is None) | (office_goer_summer is None) | (office_goer_winter is None):
        reason =\
            'Office goer probability not found,  winter={} | summer={} | transition={}'.format(office_goer_winter,
                                                                                               office_goer_summer,
                                                                                               office_goer_transition)
        return_dictionary = {'reason': reason}
        output_hvac_inefficiency_object[device]['interfering_hvac'] = return_dictionary
        logger.info('{} | {}'.format(reason, device))
        return input_hvac_inefficiency_object, output_hvac_inefficiency_object

    start_office_hour = static_params.get('ineff').get('start_office_hour')
    end_office_hour = static_params.get('ineff').get('end_office_hour')

    hourly_hvac = input_hvac_inefficiency_object.get(device).get('demand_hvac_pivot').get('values')
    days_with_hvac = np.sum((np.nansum(hourly_hvac, axis=1) > 0))
    num_days_hvac_hour_vector = np.count_nonzero(hourly_hvac, axis=0) / days_with_hvac
    overlap_score = np.nanmean(num_days_hvac_hour_vector[start_office_hour:end_office_hour])
    overlap_score = np.round(overlap_score, 2)

    feature_vector = [[office_goer_transition, office_goer_winter, office_goer_summer, overlap_score]]
    feature_vector = np.array(feature_vector)
    model_dict = input_hvac_inefficiency_object.get('models', {}).get('creeping_hvac', {})
    svm_model = model_dict.get('svm')
    dt_model = model_dict.get('dt')

    if (np.isnan(feature_vector).sum() > 0) | (np.isinf(feature_vector).sum() > 0):
        logger.info('HVAC interference NaN or Inf found in features | ')
        svm_prediction = static_params.get('ineff').get('init_pred')
        svm_probability = static_params.get('ineff').get('init_pred')
        dt_prediction = static_params.get('ineff').get('init_pred')
        dt_probability = static_params.get('ineff').get('init_pred')
    else:
        svm_prediction = svm_model.predict(feature_vector)[0]
        svm_probability = svm_model.predict_proba(feature_vector)[0]

        dt_prediction = dt_model.predict(feature_vector)[0]
        dt_probability = dt_model.predict_proba(feature_vector)[0]

        logger.info('HVAC Interference pred={}, probab={} | DT model |'.format(dt_prediction, dt_probability))
        logger.info('HVAC Interference pred={}, probab={} | SVM model |'.format(svm_prediction, svm_probability))

    # Correcting HVAC probability

    if svm_prediction == 1.:
        svm_probability[1] = svm_probability[1] * overlap_score * (1 - (office_goer_summer * office_goer_summer))
    else:
        svm_probability = np.nan

    if dt_prediction == 1.:
        dt_probability[1] = dt_probability[1] * overlap_score * (1 - (office_goer_summer * office_goer_summer))
    else:
        dt_probability = np.nan

    logger.info('Updated Interference pred={}, probab={} - SVM model |'.format(svm_prediction, svm_probability))
    logger.info('Updated Interference pred={}, probab={} - SVM model |'.format(dt_prediction, dt_probability))

    return_dictionary = {
        'transition': office_goer_transition,
        'winter': office_goer_winter,
        'summer': office_goer_summer,
        'overlap_score': overlap_score,
        'svm_prediction': svm_prediction,
        'svm_probability': svm_probability,
        'dt_prediction': dt_prediction,
        'dt_probability': dt_probability
    }

    output_hvac_inefficiency_object[device]['interfering_hvac'] = return_dictionary

    return input_hvac_inefficiency_object, output_hvac_inefficiency_object
