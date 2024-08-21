"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to regulate HVAC under-estimations
"""

# Import python packages
import logging
import numpy as np

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.get_cut_level import binary_search_for_cut_level


def get_hvac_from_residue(underestimated_months, epoch_hvac_contenders, residual, appliance_df_deepcopy, hvac_months,
                          info_carrier, logger_base):

    """
    Function to extract hvac component out of residue, in case of underestimation

    Parameters:

        underestimated_months (np.array)        : Array containing boolean of underestimated months
        epoch_hvac_contenders (np.array)        : Array containing booleans of qualified entries w.r.t ac/sh
        residual              (np.array)        : Array containing epoch level residue information
        appliance_df_deepcopy (pd.Dataframe)    : Dataframe carrying HVAC consumption info
        hvac_months           (np.array)        : Month boolean for valid AC/SH
        info_carrier          (dict)            : Dictionary containing general data required for this function
        logger_base           (logging object)  : Writes logs during code flow

    Returns:
        hvac_from_residue     (np.array)        : Array containing hvac at epoch level, extracted out of residue
    """

    # initializing logger
    logger_local = logger_base.get("logger").getChild("get_hvac_from_residue")
    logger_pass = {"logger": logger_local, "logging_dict": logger_base.get("logging_dict")}
    logger_hvac_from_res = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # reading key attributes to control excess residue
    appliance_df = info_carrier.get('appliance_df')
    epoch_input_data = info_carrier.get('epoch_input_data')
    residual_to_meet = info_carrier.get('residual_to_meet')
    month_epoch = info_carrier.get('month_epoch')

    logger_hvac_from_res.debug(" initializing hvac array underestimated due to high residue |")

    # initializing hvac array underestimated due to high residue
    hvac_from_residue = np.zeros((len(appliance_df['residue']), 1))

    # checking underestimation
    hvac_underestimated_months_bool = underestimated_months & hvac_months
    hvac_underestimated_months = month_epoch[hvac_underestimated_months_bool]

    logger_hvac_from_res.debug(" checking underestimation |")

    # controlling severe hvac underestimation
    for month in hvac_underestimated_months:

        month_rows = (epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX] == month).astype(bool)
        hvac_contenders_in_month = month_rows & epoch_hvac_contenders

        # getting initial month residue
        month_base_residue = residual[month_epoch == month]
        required_reduction = month_base_residue - residual_to_meet

        # getting maximum possible reduction in residue for hvac retrieval
        max_possible_reduction_in_month = np.sum(appliance_df['residue'][hvac_contenders_in_month]) / 1000

        # getting fail-safe cut levels
        if required_reduction <= 0:

            guess = np.max(appliance_df['residue'][hvac_contenders_in_month]) + 1
            consumption_more_than_guess = appliance_df['residue'] > guess

        elif max_possible_reduction_in_month < required_reduction:

            guess = 0
            consumption_more_than_guess = appliance_df['residue'] > guess

        else:

            # getting valid cut levels on consumption towers
            guess, consumption_more_than_guess = binary_search_for_cut_level(appliance_df, hvac_contenders_in_month,
                                                                             month_base_residue, residual_to_meet, logger_pass)

        # updating residues based on cut levels found
        appliance_df['residue'][hvac_contenders_in_month & consumption_more_than_guess] = guess

        # updating successful hvac retrieval from residue
        hvac_from_residue[hvac_contenders_in_month & consumption_more_than_guess] = np.fmax(np.array(
            appliance_df_deepcopy['residue'][hvac_contenders_in_month & consumption_more_than_guess]) - guess, 0).\
            reshape(hvac_from_residue[hvac_contenders_in_month & consumption_more_than_guess].shape)

    logger_hvac_from_res.debug(" hvac extracted from residue |")

    return hvac_from_residue
