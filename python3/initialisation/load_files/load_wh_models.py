"""
Author - Nikhil Singh Chauhan
Date - 12/08/19
This file has functions to load machine learning models for water heater
"""

# Import python packages

import os
import pickle
import logging
from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff
from python3.config.path_constants import PathConstants


def load_hld_model(hld_model_path, meta_data_dict, logger):
    """
    Parameters:
        hld_model_path              (str)       : Path to water heater detection model
        meta_data_dict              (dict)      : Meta data information
        logger                      (logger)    : Logger object

    Returns:
        hld_model                   (dict)      : Water heater detection model
    """

    t_model_load_start = datetime.now()

    # Initialize the detection model dictionary

    hld_model = {}

    # Check if model file exists in the given path

    if os.path.exists(hld_model_path):
        # If model file found in the path, load the model

        fp = open(hld_model_path, 'rb')

        hld_model = pickle.load(fp)

        fp.close()

        logger.info('HLD model loaded | ')

        # Load model description from meta data

        model_metadata = meta_data_dict.get('wh_hld_model')

        # Log the info for the model if present

        if model_metadata:
            # Iterate over each model attributes and log the information

            for key_name in model_metadata.keys():
                logger.info('HLD model %s | %s', key_name, model_metadata.get(key_name))
    else:
        # If model file not found

        logger.info('HLD model file not found | ')

    t_model_load_end = datetime.now()

    logger.info('HLD model load took | %.3f s', get_time_diff(t_model_load_start, t_model_load_end))

    return hld_model


def load_thin_model(thin_model_path, meta_data_dict, logger):
    """
    Parameters:
        thin_model_path             (str)       : Path to thin pulse duration model
        meta_data_dict              (dict)      : Meta data information
        logger                      (logger)    : Logger object

    Returns:
        thin_model                  (dict)      : Thin pulse duration model
    """

    t_model_load_start = datetime.now()

    # Initialize the detection model dictionary

    thin_model = {}

    # Check if model file exists in the given path

    if os.path.exists(thin_model_path):
        # If model file found in the path, load the model

        fp = open(thin_model_path, 'rb')

        thin_model = pickle.load(fp)

        fp.close()

        logger.info('Thin duration model loaded | ')

        # Load model description from meta data

        model_metadata = meta_data_dict.get('thin_pulse_duration_model')

        # Log the info for the model if present

        if model_metadata:
            # Iterate over each model attributes and log the information

            for key_name in model_metadata.keys():
                logger.info('Thin duration model %s | %s', key_name, model_metadata.get(key_name))
    else:
        # If model file not found

        logger.info('Thin duration model file not found | ')

    t_model_load_end = datetime.now()

    logger.info('Thin duration model load took | %.3f s', get_time_diff(t_model_load_start, t_model_load_end))

    return thin_model


def load_wh_files(disagg_version, job_tag, logger_base):
    """
    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_base         (dict)              : Contains information needed for logging

    Returns:
        loaded_files        (dict)              : Contains all loaded files
    """

    # Initiate logger for the water heater module

    logger_local = logger_base.get('logger').getChild('load_wh_files')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Dictionary containing all water heater models

    wh_models = {}

    local_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/'

    # Initialize the meta data dict

    metadata_dict = {}

    # Get models meta data information file

    metadata_file = local_path + PathConstants.MODULE_FILES_ROOT_DIR['wh'] + 'model_info.txt'

    # If metadata file present

    if os.path.exists(metadata_file):
        # Load the meta data file

        f = open(metadata_file, 'r')

        model_start_bool = True
        model_name = ''
        line = f.readline()

        # Read the meta data line by line and store to meta data dict

        while line:
            if model_start_bool:
                model_name = line.strip()
                metadata_dict[model_name] = {}
                model_start_bool = False

            elif line == '\n':
                model_start_bool = True
            else:
                line_split = line.split(':')

                field_title = line_split[0].strip()
                field_value = line_split[1].strip()

                metadata_dict[model_name][field_title] = field_value

            line = f.readline()

    # The home level detection model filename

    filename_hld = 'wh_hld_model.pkl'

    # Load the water heater detection model

    hld_model_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['wh'] + filename_hld
    wh_models['hld_model'] = load_hld_model(hld_model_path, metadata_dict, logger)

    # The thin pulse duration model filename

    filename_thin = 'thin_pulse_duration_model.pkl'

    # Load the thin pulse duration model

    thin_model_path = local_path + PathConstants.MODULE_FILES_ROOT_DIR['wh'] + filename_thin
    wh_models['thin_model'] = load_thin_model(thin_model_path, metadata_dict, logger)

    return wh_models


def get_waterheater_models(disagg_input_object, logger_base):
    """
    Parameters:
        disagg_input_object             (dict)          : Dictionary containing all inputs
        logger_base                     (dict)          : Logger object

    Returns:
        models                          (dict)          : Water heater models
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_waterheater_models')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the water heater models from disagg input object

    all_models = disagg_input_object.get('loaded_files')
    models = all_models.get('wh_files', {})

    # Check if the models exist in the models data

    hld_model = models.get('hld_model')

    if hld_model in [None, {}]:
        logger.warning('The water heater models are not present | ')
    else:
        logger.info('The water heater models loaded successfully | ')

    return models
