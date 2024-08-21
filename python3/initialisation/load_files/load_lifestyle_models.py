"""
Author: Prasoon Patidar
Date - 02th June 2020
This file has functions to load machine learning models for water heater

"""

import logging
import pickle
import os
import json
from datetime import datetime

# import function from inside the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants
from python3.utils.logs_utils import log_prefix
from python3.utils.time.get_time_diff import get_time_diff
from python3.config.path_constants import PathConstants


def load_daily_kmeans_models(kmeans_model_filepath, logger):

    """
    Parameters:
        kmeans_model_filepath      (string)     : Filepath for pickle file storing all daily loadtype models
        logger                     (dict)       : Logger Object

    Returns:
        daily_kmeans_models        (dict)       : Contains all kmeans model for daily load profile

    """
    t_model_load_start = datetime.now()

    # Load Daily Consumption Models
    daily_kmeans_models = dict()

    if os.path.isfile(kmeans_model_filepath):
        # If model file found, load the model

        file_pointer = open(kmeans_model_filepath, 'rb')

        daily_models = pickle.load(file_pointer)
        daily_kmeans_models['universal'] = daily_models

        file_pointer.close()

        # Log the info for the model if present

        logger.info("%s Universal Daily kmeans models loaded | ", log_prefix('DailyLoadType'))

        for key in daily_models.keys():
            # iterate over each consumption/variation based model and log the information

            logger.info("%s Available Universal Daily Kmeans Model | %s", log_prefix('DailyLoadType'), key)

    else:
        # if model filename not found

        logger.error("%s Universal Daily kmeans models not found |", log_prefix('DailyLoadType'))

    lifestyle_pilots = PilotConstants.LIFESTYLE_ENABLED_PILOTS

    for pilot in lifestyle_pilots:

        kmeans_model_pilot_filepath = kmeans_model_filepath[:-4] + "_" + str(pilot) + ".pkl"

        if os.path.isfile(kmeans_model_pilot_filepath):
            # If model file found, load the model

            file_pointer = open(kmeans_model_pilot_filepath, 'rb')

            daily_kmeans_models[str(pilot)] = pickle.load(file_pointer)

            file_pointer.close()

            # Log the info for the model if present

            logger.info("Daily kmeans models loaded for pilot | %s", pilot)

        else:
            # if model filename not found

            logger.warning("Daily kmeans models not found for pilot | %s", pilot)

    t_model_load_end = datetime.now()

    logger.info("%s Lifestyle Module: Daily kmeans model loading took | %.3f s", log_prefix('DailyLoadType'),
                get_time_diff(t_model_load_start, t_model_load_end))

    return daily_kmeans_models


def load_yearly_kmeans_models(kmeans_model_filepath, logger):

    """
    Parameters:
        kmeans_model_filepath      (string)     : Filepath for pickle file storing all yearly loadtype model
        logger                     (dict)       : Logger Object

    Returns:
        yearly_kmeans_model         (dict)       : kmeans model for yearly load profile

    """
    t_model_load_start = datetime.now()

    # Load Daily Consumption Models
    yearly_kmeans_models = dict()

    if os.path.exists(kmeans_model_filepath):
        # If model file found, load the model

        file_pointer = open(kmeans_model_filepath, 'rb')

        yearly_models = pickle.load(file_pointer)
        yearly_kmeans_models['universal'] = yearly_models

        file_pointer.close()

        # Log the info for the model if present

        logger.info("%s Universal Yearly kmeans models loaded | ", log_prefix('SeasonalLoadType'))

    else:
        # if model filename not found

        logger.error("%s Universal Yearly kmeans models not found |", log_prefix('SeasonalLoadType'))

    lifestyle_pilots = PilotConstants.LIFESTYLE_ENABLED_PILOTS

    for pilot in lifestyle_pilots:

        kmeans_model_pilot_filepath = kmeans_model_filepath[:-4] + "_" + str(pilot) + ".pkl"

        if os.path.isfile(kmeans_model_pilot_filepath):
            # If model file found, load the model

            file_pointer = open(kmeans_model_pilot_filepath, 'rb')

            yearly_kmeans_models[str(pilot)] = pickle.load(file_pointer)

            file_pointer.close()

            # Log the info for the model if present

            logger.info("Yearly kmeans models loaded for pilot | %s", pilot)

        else:
            # if model filename not found

            logger.warning("Yearly kmeans models not found for pilot | %s", pilot)

    t_model_load_end = datetime.now()

    logger.info("%s Lifestyle Module: Daily kmeans model loading took | %.3f s", log_prefix('SeasonalLoadType'),
                get_time_diff(t_model_load_start, t_model_load_end))

    return yearly_kmeans_models


def load_pilot_based_info(pilot_info_filepath, logger):

    """
    Parameters:
        pilot_info_filepath      (string)       : Json filepath containing all pilot level config
        logger                     (dict)       : Logger Object

    Returns:
        pilot_based_info        (dict)          : Contains all pilot_level_configs

    """

    t_pilot_based_info_start = datetime.now()

    # Load Daily Consumption Models
    pilot_based_info = None

    if os.path.exists(pilot_info_filepath):
        # If model file found, load the model

        file_pointer = open(pilot_info_filepath, 'r')

        pilot_based_info = json.load(file_pointer)

        file_pointer.close()

        # Log the info for the model if present

        logger.info("%s pilot based information loaded | ", log_prefix('DailyLoadType'))

        for key in pilot_based_info.keys():
            # iterate over each consumption/variation based model and log the information

            logger.debug("%s Available pilot info for | %s", log_prefix('DailyLoadType'), key)

    else:
        # if model filename not found

        logger.error("%s pilot based information not found |", log_prefix('DailyLoadType'))

    t_pilot_based_info_end = datetime.now()

    logger.info("%s Lifestyle Module: pilot based info loading took | %.3f s", log_prefix('DailyLoadType'),
                get_time_diff(t_pilot_based_info_start, t_pilot_based_info_end))

    return pilot_based_info


def load_lf_files(disagg_version, job_tag, logger_base):

    """
    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_base         (dict)              : Contains information needed for logging

    Returns:
        loaded_files        (dict)              : Contains all loaded files
    """

    # Initiate logger for the lifestyle model fetch module

    logger_local = logger_base.get('logger').getChild('load_lf_files')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Dictionary containing all lifestyle models

    lf_models = {}

    local_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/'

    # Filename and path for daily load models

    filename_daily_kmeans_models = 'kmeans_daily_load_models.pkl'
    filepath_daily_kmeans_models = \
        local_path + PathConstants.MODULE_FILES_ROOT_DIR['lf'] + filename_daily_kmeans_models

    # Load daily kmeans models

    lf_models['daily_kmeans_models'] = load_daily_kmeans_models(filepath_daily_kmeans_models, logger)

    # Filename and path for yearly load models

    filename_yearly_kmeans_models = 'kmeans_yearly_load_models.pkl'
    filepath_yearly_kmeans_models = \
        local_path + PathConstants.MODULE_FILES_ROOT_DIR['lf'] + filename_yearly_kmeans_models

    # Load yearly kmeans models

    lf_models['yearly_kmeans_model'] = load_yearly_kmeans_models(filepath_yearly_kmeans_models, logger)

    # load pilot based config info

    filename_pilot_based_info = 'pilot_level_config.json'
    filepath_pilot_based_info = \
        local_path + PathConstants.MODULE_FILES_ROOT_DIR['lf'] + filename_pilot_based_info

    lf_models['pilot_based_info'] = load_pilot_based_info(filepath_pilot_based_info, logger)

    # check if models are not load, and return lifestyle run module status

    lf_models['run_lifestyle_module_status'] = True

    if (lf_models['pilot_based_info'] is None) | (lf_models['daily_kmeans_models'] is None) | (
            lf_models['yearly_kmeans_model'] is None):

        lf_models['run_lifestyle_module_status'] = False

    return lf_models


def get_daily_kmeans_lifestyle_models(disagg_input_object, model_id, pilot_id_str, logger_base):

    """
    Parameters:
        disagg_input_object             (dict)          : Dictionary containing all inputs
        model_id                        (dict)          : Model id for daily kmeans load file
        logger_base                     (dict)          : Logger object

    Returns:
        daily_load_kmeans_model         (dict)          : lifestyle daily load profile model
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger_base').getChild('get_lifestyle_models')

    # Specific logger for this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the water heater models from disagg input object

    all_models = disagg_input_object.get('loaded_files')
    all_lifestyle_models = all_models.get('lf_files')
    all_daily_kmeans_models = all_lifestyle_models.get('daily_kmeans_models')

    if pilot_id_str not in all_daily_kmeans_models.keys():
        # Use global comparision keys to process, but throw warning

        pilot_based_info_key = 'universal'

        logger.warning("%s pilot id is not present in daily kmeans model, using universal config for this user",
                       pilot_id_str)

    else:

        pilot_based_info_key = pilot_id_str

        logger.info("%s pilot id is present in daily kmeans model, using pilot config for this user",
                    pilot_id_str)

    all_daily_kmeans_models = all_daily_kmeans_models.get(pilot_based_info_key)

    # Select one out of all models based on mapping model_id to model_key

    daily_kmeans_model_id_key_mapping = {
        'c': 'consumption',
        'v': 'variation',
        'l': 'low',
        'a': 'avg',
        'h': 'high'
    }

    model_key = '_'.join([daily_kmeans_model_id_key_mapping.get(k, '') for k in model_id])

    model = all_daily_kmeans_models.get(model_key)

    # Check if the models exist in the models data

    if model is None:
        logger.error("%s Daily load kmeans model is not present | ", log_prefix('DailyLoadType'))
    else:
        logger.info("%s Daily load kmeans model loaded successfully | ", log_prefix('DailyLoadType'))

    return model


def get_pilot_based_info(disagg_input_object, pilot_id_str, logger_base):

    """
    Parameters:
        disagg_input_object             (dict)          : Dictionary containing all inputs
        pilot_id_str                    (str)           : pilot id for this user(str format)
        logger_base                     (dict)          : Logger object

    Returns:
        pilot based info                (dict)          : pilot based info for lifestyle modules
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger_base').getChild('get_pilot_based_info')

    # Specific logger for this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the lifestyle models from disagg input object

    all_models = disagg_input_object.get('loaded_files')
    all_lifestyle_models = all_models.get('lf_files')
    all_pilot_based_info = all_lifestyle_models.get('pilot_based_info')

    # Check if pilot Id is present in the pilot configs

    if pilot_id_str not in all_pilot_based_info.keys():
        # Use global comparision keys to process, but throw warning

        pilot_based_info_key = 'universal'

        logger.warning("%s pilot id %s is not present in pilot config, using universal config for this user",
                       log_prefix('DailyLoadType'), pilot_id_str)

    else:

        pilot_based_info_key = pilot_id_str

        logger.info("%s pilot id %s is present in pilot config, using pilot config for this user",
                    log_prefix('DailyLoadType'), pilot_id_str)

    # get config for given pilot_based info key

    pilot_based_info = all_pilot_based_info.get(pilot_based_info_key)

    return pilot_based_info
