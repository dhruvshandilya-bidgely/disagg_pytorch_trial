"""
Author: Abhinav Srivastava
Date:   08-Feb-2020
Postprocessing AO HVAC for SMB [Function separated and moved to this new file by Neelabh on 14 June 2023]
"""

# Import python packages

import numpy as np
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def postprocess_ao_seasonality(disagg_input_object, disagg_output_object, month_epoch, month_idx):
    """
    Function to Take off False positive of AO HVAC

    Parameters:

        disagg_input_object (dict)          : Dictionary containing all the inputs
        disagg_output_object (dict)         : Dictionary containing all the outputs
        month_epoch (np.ndarray)            : Array containing month epochs
        month_idx (np.ndarray)              : Array containing Month indexes

    Returns:
        None
    """

    static_params = hvac_static_params()

    month_identifier = disagg_input_object['input_data'][:, Cgbdisagg.INPUT_BILL_CYCLE_IDX]

    ao_out = disagg_output_object['ao_seasonality']

    epoch_ac = ao_out['epoch_cooling']
    epoch_sh = ao_out['epoch_heating']
    epoch_grey = ao_out['epoch_grey']

    ac_to_suppress = (ao_out['cooling'] < static_params['ao']['suppress_fp_hvac']) & (ao_out['cooling'] > 0)
    sh_to_suppress = (ao_out['heating'] < static_params['ao']['suppress_fp_hvac']) & (ao_out['heating'] > 0)

    if any(ac_to_suppress):

        month_to_suppress = month_epoch[ac_to_suppress]

        for month in month_to_suppress:
            suppress_month_epochs = (month_identifier == month)
            epoch_grey[suppress_month_epochs] = epoch_grey[suppress_month_epochs] + epoch_ac[suppress_month_epochs]
            epoch_ac[suppress_month_epochs] = 0

        disagg_output_object['ao_seasonality']['epoch_cooling'] = epoch_ac

    if any(sh_to_suppress):

        month_to_suppress = month_epoch[sh_to_suppress]

        for month in month_to_suppress:
            suppress_month_epochs = (month_identifier == month)
            epoch_grey[suppress_month_epochs] = epoch_grey[suppress_month_epochs] + epoch_sh[suppress_month_epochs]
            epoch_sh[suppress_month_epochs] = 0

        disagg_output_object['ao_seasonality']['epoch_heating'] = epoch_sh

    disagg_output_object['ao_seasonality']['epoch_grey'] = epoch_grey

    month_ao_cool = np.bincount(month_idx, epoch_ac)
    month_ao_heat = np.bincount(month_idx, epoch_sh)
    month_grey = np.bincount(month_idx, epoch_grey)

    disagg_output_object['ao_seasonality']['cooling'] = month_ao_cool
    disagg_output_object['ao_seasonality']['heating'] = month_ao_heat
    disagg_output_object['ao_seasonality']['grey'] = month_grey
