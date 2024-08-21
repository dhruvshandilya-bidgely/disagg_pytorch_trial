"""
Author - Paras Tehria
Date - 12th Nov 2019
This module loads solar models
"""

import os
import torch
import pickle
import joblib
import logging
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.config.path_constants import PathConstants
from python3.disaggregation.aer.solar.functions.cnn_model import Net


def load_detection_model(model_path, logger, model_config):

    """
    Load solar detection models

    Parameters:
        model_path          (string)                : String containing the path of the model
        logger              (logging.LoggerAdapter) : logger
        model_config        (dict)                  : config file for cnn model

    Returns:
        model               (Net)               : loaded CNN model in evaluation mode
    """

    t_model_load_start = datetime.now()

    n_channel = model_config.get('n_channel')
    width = model_config.get('width')
    height = model_config.get('height')
    output_dim = model_config.get('output_dim')
    batch_size = model_config.get('batch_size')

    if os.path.exists(model_path):
        model_file = torch.load(model_path)
        model = Net(n_channel, width, height, batch_size, output_dim, model_config)
        model.load_state_dict(model_file['model'])
        model.eval()
        logger.info('Solar model loaded | ')

    else:
        logger.info('Solar model file not found | ')
        return None

    t_model_load_end = datetime.now()

    logger.info('Solar model load took | %.3f s', get_time_diff(t_model_load_start, t_model_load_end))

    return model


def load_estimation_model(model_path, logger):

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
        logger.info('Solar estimation model loaded | ')

    else:
        logger.info('Solar estimation model file not found | ')
        return None

    t_model_load_end = datetime.now()

    logger.info('Solar estimation model load took | %.3f s', get_time_diff(t_model_load_start, t_model_load_end))

    return model


def load_lgb_model(model_path, logger):

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
        logger.info('Solar detection lgb model loaded | ')

    else:
        logger.info('Solar detection lgb model file not found | ')
        return None

    t_model_load_end = datetime.now()

    logger.info('Solar estimation model load took | %.3f s', get_time_diff(t_model_load_start, t_model_load_end))

    return model


def load_propensity_model(propensity_model_path, logger):

    """
    Function to load solar propensity models
        Parameters:
            propensity_model_path          (string)                : String containing the path of the estimation model
            logger                         (logging.LoggerAdapter) : logger

        Returns:
            model                          (pickle)                : pickle file containing solar estimation model
    """
    t_model_load_start = datetime.now()

    propensity_model = {}

    # Loading propensity model 1 and 2. Model 2 showed better result in dev-qa, but intergrating both for now
    model_1_path = propensity_model_path + 'propensity_model_1.sav'
    if os.path.exists(model_1_path):
        model_1 = joblib.load(model_1_path)
        propensity_model['model_1'] = model_1
        logger.info('Solar propensity model 1 loaded | ')

    else:
        logger.warning('Solar propensity model file not found | ')

    model_2_path = propensity_model_path + 'propensity_model_2.sav'
    if os.path.exists(model_2_path):
        model_2 = joblib.load(model_2_path)
        propensity_model['model_2'] = model_2
        logger.info('Solar propensity model 2 loaded | ')

    else:
        logger.warning('Solar propensity model file not found | ')

    t_model_load_end = datetime.now()

    logger.info('Solar model load took | %.3f s', get_time_diff(t_model_load_start, t_model_load_end))

    return propensity_model


def load_solar_files(disagg_version, job_tag, logger_base):

    """
    Function to load files for solar disagg modules

    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_base         (dict)              : Contains information needed for logging

    Returns:
        solar_models        (dict)              : Contains all loaded files
    """

    # Initiate logger for the water heater module

    logger_local = logger_base.get("logger").getChild("load_so_files")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Dictionary containing all water heater models

    solar_models = {}

    local_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/'

    # Load metadata file

    metadata_dict = {}

    metadata_file = local_path + PathConstants.MODULE_FILES_ROOT_DIR['solar'] + 'model_info.txt'

    if os.path.exists(metadata_file):
        f = open(metadata_file, 'r')

        model_start_bool = True
        model_name = ''
        line = f.readline()

        while line:
            if model_start_bool:
                metadata_dict[line] = {}
                model_name = line
                model_start_bool = False
            elif line == '\n':
                model_start_bool = True
            else:
                line_split = line.split(':')

                field_title = line_split[0].strip()
                field_value = line_split[1].strip()

                metadata_dict[model_name][field_title] = field_value

    # The home level detection model filename

    filename_model = 'model.pth'
    filename_config = 'config.pb'
    detection_model_lgb = 'lgbm_solar_detection.pkl'
    estimation_model = 'xgb_curve_estimation.pkl'

    model_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['solar'] + filename_model
    config_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['solar'] + filename_config

    detection_model_lgb_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['solar'] + detection_model_lgb
    estimation_model_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['solar'] + estimation_model

    if os.path.exists(config_path):
        with open(config_path, 'rb') as handle:
            model_config = pickle.load(handle)
        solar_models['detection_model'] = load_detection_model(model_path, logger, model_config)

    else:
        logger.info('Solar detection model files not found | ')
        solar_models['detection_model'] = None

    propensity_model_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['solar']

    solar_models['detection_lgb_model'] = load_lgb_model(detection_model_lgb_path, logger)
    solar_models['estimation_model'] = load_estimation_model(estimation_model_path, logger)
    solar_models['propensity_model'] = load_propensity_model(propensity_model_path, logger)
    logger.debug('Finished loading solar detection and estimation model ')

    return solar_models
