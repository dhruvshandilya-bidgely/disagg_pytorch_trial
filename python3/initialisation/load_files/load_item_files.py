"""
Author - Nisha Agarwal
Date - 12th Aug 2020
This module loads hybrid v2 models
"""
# Import python packages

import os
import pandas as pd

# import function from inside the project

from python3.config.path_constants import PathConstants


def load_item_files(disagg_version, job_tag, logger_base):

    """
    Parameters:
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger_base         (dict)              : Contains information needed for logging

    Returns:
        loaded_files        (dict)              : Contains all loaded files
    """

    # Dictionary containing all lifestyle models

    item_models = {}

    pilot_config = dict()

    local_path = PathConstants.FILES_LOCAL_ROOT_DIR + disagg_version + '_' + job_tag + '/'

    filepath = local_path + PathConstants.MODULE_FILES_ROOT_DIR['item']

    for i, file in enumerate(os.listdir(filepath)):

        if '.csv' not in file:
            continue

        config = pd.read_csv(filepath + file, index_col=None, header=None)

        pilot_config[file] = config

    # Load daily kmeans models

    item_models['pilot_config'] = pilot_config

    return item_models
