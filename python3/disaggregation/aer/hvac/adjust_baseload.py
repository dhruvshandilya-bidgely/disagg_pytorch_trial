"""
Author - Abhinav
Date - 10/10/2018
Calculate adjusted ao
"""

# Import python packages
import copy
import logging
import numpy as np
from scipy.stats.mstats import mquantiles

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg


def adjust_baseload(hvac_input_data, baseload, hvac_params, logger_base,
                    past_baseload_from_hsm=None, min_baseload_from_hsm=None):

    """
    Function adjusts the algo detected ao, to eliminate possibility of hvac creeping in always on (ao)
    Parameters:
        hvac_input_data         (np.ndarray)   : 2D Array of epoch level input data frame flowing into hvac module
        baseload                (np.ndarray)   : Array of epoch level ao estimated in AO module
        hvac_params             (dict)         : Dictionary containing hvac algo related initialized parameters
        logger_base             (logger)       : Writes logs during code flow
        past_baseload_from_hsm  (int)          : Read from HSM - last ao value seen while stabilizing
        min_baseload_from_hsm   (int)          : Read from HSM - minimum epoch level ao seen while stabilizing
    Returns:
        ao                      (np.ndarray)   : Array of epoch level stabilized ao
        last_baseload           (int)          : Last ao value seen while stabilizing
        min_baseload            (int)          : Minimum epoch level ao seen while stabilizing
    """

    logger_local = logger_base.get("logger").getChild("adjust_baseload")
    logger_hvac = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    if len(baseload) == 0:
        logger_hvac.info('returning without adjustment. ao absent from ao algo |')
        return baseload, None, None

    # keeping a copy of ao from ao module
    baseload = copy.deepcopy(baseload)

    # seeiing when ao values change at epoch level
    baseload_diff = (np.diff(baseload.T) != 0).astype(int)
    rez_data = np.nonzero(baseload_diff)[0]

    # keeping a note of epoch indexes where ao changes
    distinct_baseload_idx = rez_data

    # getting minimum ao at epoch level
    min_baseload = mquantiles((baseload[baseload > 0]), 0.1, alphap=0.5, betap=0.5)
    logger_hvac.debug('minimum ao at epoch level is : {} |'.format(min_baseload))

    # setting last ao variable to minimum ao
    last_baseload = min_baseload

    mode_mtd = False

    if min_baseload_from_hsm is not None:
        logger_hvac.info('adjusting ao in mtd mode |')

        # adjusting ao in mtd mode after reading from hsm attributes
        min_baseload = min_baseload_from_hsm
        last_baseload = past_baseload_from_hsm
        rez_data = np.array(np.nonzero((np.diff(baseload) != 0).astype(int)))

        # getting indexes wher ao changes
        distinct_baseload_idx = np.array(rez_data[0])
        logger_hvac.info('minimum ao at epoch level is : {} |'.format(min_baseload))
        mode_mtd = True

    # reading temperature from hvac input data
    temperature = hvac_input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX]

    # if ao doesn't change at all at epoch level, no need of adjustment
    if np.all(distinct_baseload_idx == 0):
        logger_hvac.debug('last ao at epoch level is : {} |'.format(last_baseload))
        logger_hvac.info('no need of ao adjustment. constant ao |')
        return baseload, last_baseload, min_baseload

    for i in range(distinct_baseload_idx.shape[0] + 1):

        if i == len(distinct_baseload_idx):
            start_idx = distinct_baseload_idx[i - 1] + 1
            end_idx = len(baseload)
        elif i == 0:

            # initializing start index
            start_idx = 0

            # initializing end index
            end_idx = distinct_baseload_idx[i] + 1
        else:
            start_idx = distinct_baseload_idx[i - 1] + 1
            end_idx = distinct_baseload_idx[i] + 1

        # measuring how much ao is different from last ao seen
        diff_baseload = baseload[start_idx] - last_baseload

        temp1 = temperature[start_idx:end_idx]

        # getting cdd and hdd based on base temperature of 65F
        cdd = np.sum(np.fmax(temp1 - 65., 0))
        hdd = np.sum(np.fmax(60 - temp1, 0))

        threshold = np.inf

        # hdd and cdd decides the threshold for change in ao from last seen ao
        if hdd > cdd:
            threshold = hvac_params['adjustment']['SH']['MIN_AMPLITUDE']
        elif cdd > hdd:
            threshold = hvac_params['adjustment']['AC']['MIN_AMPLITUDE']

        # if difference in ao jump is above threshold, stabilizing is required
        if diff_baseload > threshold:
            baseload[start_idx:end_idx] = last_baseload
        elif ~mode_mtd and baseload[start_idx] > min_baseload:
            last_baseload = baseload[start_idx]

    logger_hvac.info('ao adjustment successful |')

    return baseload, last_baseload, min_baseload
