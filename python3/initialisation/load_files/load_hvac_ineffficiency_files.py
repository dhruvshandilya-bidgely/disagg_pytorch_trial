"""
Author - Anand Kumar Singh
Date - 15th June 2021
This module loads hvac inefficiency files
"""

import os
import pickle
import joblib
import logging
from datetime import datetime

# Import functions from within the project

from python3.config.path_constants import PathConstants
from python3.utils.time.get_time_diff import get_time_diff


def load_model(model_path, logger):

    """
    Function to load solar estimation model
        Parameters:
            model_path          (string)                : String containing the path of the estimation model
            logger              (logging.LoggerAdapter) : logger

        Returns:
            model               (pickle)                : pickle file containing solar estimation model
    """

    t_model_load_start = datetime.now()
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.debug('Model loaded | %s', model_path)

    else:
        logger.info('Model file not found | ')
        return None

    t_model_load_end = datetime.now()

    logger.debug('Model load took | %.3f s', get_time_diff(t_model_load_start, t_model_load_end))

    return model


def load_hvac_inefficiency_files(disagg_version, job_tag, logger_base):

    """
    Function to load files for solar disagg modules

    Parameters:
        disagg_version          (string)            : String containing the version information of the build
        job_tag                 (string)            : String containing information regarding the build process used
        logger_base             (dict)              : Contains information needed for logging

    Returns:
        hvac_inefficiency_model (dict)              : Contains all loaded files
    """

    # Initiate logger for the water heater module

    logger_local = logger_base.get("logger").getChild("load_hvac_inefficiency_files")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Dictionary containing all water heater models

    hvac_inefficiency_model = {
        'app_change': {
            'dt':   None,
            'dt_path': 'dt_classifier_app_change.pkl',
            'svm':  None,
            'svm_path': 'svm_classifier_app_change.pkl',
            'log_reg': None,
            'log_reg_path': 'log_reg_classifier_app_change.pkl',
            'normalisation_dict': None,
            'normalisation_dict_path': 'normalisation_dict_app_change.pkl'
        },
        'app_degradation': {
            'dt':   None,
            'dt_path': 'dt_classifier_app_degradation.pkl',
            'svm':  None,
            'svm_path': 'svm_classifier_app_degradation.pkl',
            'log_reg': None,
            'log_reg_path': 'log_reg_classifier_app_degradation.pkl',
            'normalisation_dict': None,
            'normalisation_dict_path': 'normalisation_dict_app_degradation.pkl'
        },
        'creeping_hvac': {
            'dt':   None,
            'dt_path': 'dt_classifier_creeping_hvac.pkl',
            'svm':  None,
            'svm_path': 'svm_classifier_creeping_hvac.pkl',
            'normalisation_dict': None,
            'normalisation_dict_path': 'normalisation_dict_creeping_hvac.pkl'
        }
    }

    local_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/'

    model_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['hvac_inefficiency']

    for key in hvac_inefficiency_model:
        for individual_model_path in hvac_inefficiency_model[key]:
            if '_path' in individual_model_path:
                full_model_path = model_path + hvac_inefficiency_model[key][individual_model_path]
                model = individual_model_path.split('_path')[0]
                hvac_inefficiency_model[key][model] = load_model(full_model_path, logger)
    logger.debug('Finished loading HVAC Inefficiency models ')

    return hvac_inefficiency_model
