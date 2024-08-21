"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to load EV models
"""

# Import python packages
import datetime
import os
import pickle
import logging

import torch
from python3.utils.time.get_time_diff import get_time_diff
# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants


def load_ev_files(disagg_version, job_tag, logger_base):
    """
    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_base         (dict)              : Contains information needed for logging
    Returns:
        loaded_files        (dict)              : Contains all loaded files
    """

    # Initiate logger for the water heater module

    logger_local = logger_base.get("logger").getChild("load_ev_files")
    logger = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # Dictionary containing all water heater models

    ev_models = {'ev_hld': {}}

    # Loading home level detection gradient boosting (High recall) model

    xgb_filename = 'ev_detection_xgb.pkl'

    xgb_model_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/' + \
                     PathConstants.MODULE_FILES_ROOT_DIR['ev'] + xgb_filename

    if os.path.exists(xgb_model_path):
        loaded_model = pickle.load(open(xgb_model_path, 'rb'))
        ev_models['ev_hld']['xgb'] = loaded_model
        logger.info('EV XGB detection model loaded successfully | ')
    else:
        logger.info('EV XGB detection model file does not exist | ')

    # Loading home level detection Random forest (High precision) model
    rf_filename = 'ev_detection_rf.pkl'

    rf_model_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/' + \
                    PathConstants.MODULE_FILES_ROOT_DIR['ev'] + rf_filename

    if os.path.exists(rf_model_path):
        loaded_model = pickle.load(open(rf_model_path, 'rb'))
        ev_models['ev_hld']['rf'] = loaded_model

        logger.info('EV RF detection model loaded successfully | ')
    else:
        logger.info('EV RF detection model file does not exist | ')

    # Loading home level detection XGB (high Recall model) for EV L1

    xgb_l1_filename = 'ev_l1_detection_xgb.pkl'

    xgb_l1_model_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/' + \
                        PathConstants.MODULE_FILES_ROOT_DIR['ev'] + xgb_l1_filename

    if os.path.exists(xgb_l1_model_path):
        loaded_model = pickle.load(open(xgb_l1_model_path, 'rb'))
        ev_models['ev_hld']['l1'] = loaded_model

        logger.info('EV L1 detection model loaded successfully | ')
    else:
        logger.info('EV L1 detection model file does not exist | ')

    return ev_models


def get_ev_models(disagg_input_object, logger_base):
    """
    Parameters:
        disagg_input_object             (dict)          : Dictionary containing all inputs
        logger_base                     (dict)          : Logger object

    Returns:
        models                          (dict)          : Water heater models
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_ev_models')

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
        logger.info('The ML electric vehicle models loaded successfully | ')

    # Loading tensorflow models
    t_start = datetime.datetime.now()
    hld_model = load_ev_tensorflow_models(hld_model, logger, disagg_input_object)
    t_end = datetime.datetime.now()
    logger.info("Debugging : Loading DL models took | %s", get_time_diff(t_start, t_end))

    if hld_model.get('l1_cnn') in [None, {}] or hld_model.get('l2_cnn') in [None, {}]:
        logger.warning('The Tensorflow electric vehicle models are not present | ')
        detection_model_present = False
    else:
        logger.info('The Tensorflow electric vehicle models loaded successfully | ')

    return models, detection_model_present


def load_ev_tensorflow_models(hld_model, logger, disagg_input_object):
    """
    Loading tensorflow EV models
    Parameters:
        hld_model                   (Dict)          : Loading HLD model
        logger                      (Logger)        : Logger
        disagg_input_object         (Dict)          : Disagg input object
    Returns:
        hld_model                   (Dict)          : Loading HLD model
    """

    build_info = disagg_input_object.get('config').get('build_info').split('.')[-1]

    # Loading L1 CNN model files

    ev_l1_cnn_filename = 'ev_l1_cnn_pytorch.pt'

    ev_l1_cnn_model_path = PathConstants.FILES_LOCAL_ROOT_DIR + build_info + '/' + \
                           PathConstants.MODULE_FILES_ROOT_DIR['ev'] + ev_l1_cnn_filename

    logger.info("EV L1 path | %s", ev_l1_cnn_model_path)

    if os.path.exists(ev_l1_cnn_model_path):
        model = torch.load(ev_l1_cnn_model_path)
        hld_model['l1_cnn'] = model

        logger.info('EV L1 CNN model path loaded successfully | ')
    else:
        logger.info('EV L1 CNN model file does not exist | ')

    # Loading L2 CNN model files

    ev_l2_cnn_filename = 'ev_l2_cnn_pytorch.pt'

    ev_l2_cnn_model_path = PathConstants.FILES_LOCAL_ROOT_DIR + build_info + '/' + \
                           PathConstants.MODULE_FILES_ROOT_DIR['ev'] + ev_l2_cnn_filename
    logger.info("EV L1 path | %s", ev_l2_cnn_model_path)
    if os.path.exists(ev_l2_cnn_model_path):
        model = torch.load(ev_l2_cnn_model_path)
        hld_model['l2_cnn'] = model

        logger.info('EV L2 CNN model path loaded successfully | ')
    else:
        logger.info('EV L2 CNN model file does not exist | ')

    return hld_model
