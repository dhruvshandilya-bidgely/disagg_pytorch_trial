"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to load EV models
"""

# Import python packages

import os
import time
import pickle
import logging
import traceback
import uszipcode
import pandas as pd

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants


def load_ev_propensity_files(disagg_version, job_tag, logger_base):
    """
    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_base         (dict)              : Contains information needed for logging


    Returns:
        loaded_files        (dict)              : Contains all loaded files
    """

    # Initiate logger for the water heater module

    logger_local = logger_base.get("logger").getChild("load_ev_propensity_files")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Dictionary containing all water heater models

    ev_propensity_files = dict()

    # Loading ev propensity model

    propensity_model_filename = 'ev_propensity_model.pb'

    propensity_model_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/' + \
                    PathConstants.MODULE_FILES_ROOT_DIR['ev_propensity'] + propensity_model_filename

    if os.path.exists(propensity_model_path):
        loaded_model = pickle.load(open(propensity_model_path, 'rb'))
        ev_propensity_files['model'] = loaded_model

        logger.info('EV Propensity model loaded successfully | ')
    else:
        logger.info('EV Propensity model file does not exist | ')

    # Loading zipcode ev stations files

    charging_station_filename = 'na_charging_station_data.csv'

    charging_station_data_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/' + \
                                 PathConstants.MODULE_FILES_ROOT_DIR['ev_propensity'] + charging_station_filename

    if os.path.exists(charging_station_data_path):
        loaded_file = pd.read_csv(charging_station_data_path)
        ev_propensity_files['charging_station_data'] = loaded_file

        logger.info('EV Charging Station data loaded successfully | ')
    else:
        logger.info('EV Charging Station data file does not exist | ')

    # Download us zip-code database

    zipcode_db_file_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/' + \
                           PathConstants.MODULE_FILES_ROOT_DIR['ev_propensity'] + '/db.sqlite'

    t_start = time.time()

    # noinspection PyBroadException
    try:
        uszipcode.SearchEngine(simple_or_comprehensive=uszipcode.SearchEngine.SimpleOrComprehensiveArgEnum.comprehensive,
                               db_file_path=zipcode_db_file_path)

        ev_propensity_files['zipcode_db_file_path'] = zipcode_db_file_path

        t_end = time.time()

        logger.info('US zipcode database downloaded, time taken: {} | '.format(t_end - t_start))

    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger.info('US zipcode database could not be downloaded | {}'.format(error_str))

    return ev_propensity_files


def get_ev_propensity_models(disagg_input_object, logger_base):
    """
    Parameters:
        disagg_input_object             (dict)          : Dictionary containing all inputs
        logger_base                     (dict)          : Logger object
    Returns:
        models                          (dict)          : Water heater models
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_ev_propensity_models')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve the electric vehicle models from disagg input object

    all_models = disagg_input_object.get('loaded_files')
    models = all_models.get('ev_files', {})

    # Check if the models exist in the models data

    hld_model = models.get('ev_hld')

    detection_model_present = True
    if hld_model in [None, {}]:
        logger.warning('The electric vehicle models are not present | ')
        detection_model_present = False

    else:
        logger.info('The electric vehicle models loaded successfully | ')

    return models, detection_model_present
