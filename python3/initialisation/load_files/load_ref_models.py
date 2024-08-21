"""
Author - Nisha Agarwal
Date - 12th Aug 2020
This module loads Ref models
"""
# Import python packages

import pickle

# import function from inside the project

from python3.config.path_constants import PathConstants


def load_ref_files(disagg_version, job_tag, logger_base):

    """
    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_base         (dict)              : Contains information needed for logging

    Returns:
        loaded_files        (dict)              : Contains all loaded files
    """

    # Dictionary containing all lifestyle models

    ref_models = {}

    local_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/'

    filepath = local_path + PathConstants.MODULE_FILES_ROOT_DIR['ref']

    ref_models['raw'] = pickle.load(open(filepath + "raw.sav", "rb"))

    ref_models['raw_meta'] = pickle.load(open(filepath + "raw_meta.sav", "rb"))

    ref_models['raw_hvac'] = pickle.load(open(filepath + "raw_hvac.sav", "rb"))

    ref_models['raw_hvac_meta'] = pickle.load(open(filepath + "raw_hvac_meta.sav", "rb"))

    return ref_models
