"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to regulate HVAC estimations
"""

# Import python packages
import logging
import numpy as np

# Import functions from within the project
from python3.disaggregation.aes.hvac.get_smb_params import get_smb_params


def binary_search_for_cut_level(appliance_df, hvac_contenders_in_month, month_base_residue, residual_to_meet, logger_base):

    """
    Function to search for the consumption level cut. Consumption over the cut will go into hvac.

    Parameters:

        appliance_df                (pd.Dataftame)              : Dataframe containing hvac consumption info
        hvac_contenders_in_month    (np.ndarray)                : Array containing valid ac/sh points
        month_base_residue          (float)                     : The original residue for month
        residual_to_meet            (float)                     : The required residue level for the month

    Returns:
        guess                       (float)                     : guess level of cut
        consumption_more_than_guess (np.ndarray)                : Contains boolean of consumption more than guess
    """

    # initializing logger
    logger_local = logger_base.get("logger").getChild("binary_search")
    logger_binary_search = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    smb_params = get_smb_params()

    # making initial guess for cut level
    residue_low = 0
    residue_high = np.max(appliance_df['residue'][hvac_contenders_in_month])
    guess = (residue_low + residue_high) / 2
    consumption_more_than_guess = appliance_df['residue'] > guess

    logger_binary_search.info('Making initial guess |')

    # checking if any hvac contender exists in residue
    hvac_contender_residue = appliance_df['residue'][
        hvac_contenders_in_month & consumption_more_than_guess]

    # checking the extent of gap from residue to meet
    gap_between_residue = (month_base_residue - np.sum(
        np.fmax(hvac_contender_residue - guess, 0)) / 1000) - residual_to_meet

    logger_binary_search.info('gap between residue is {} |'.format(gap_between_residue))

    # Binary search for where to make the cut in consumption towers
    while abs(gap_between_residue) > smb_params.get('utility').get('binary_search_cut_gap'):

        if gap_between_residue > 0:
            residue_high = guess
        else:
            residue_low = guess

        # updating guess level
        guess = (residue_low + residue_high) / 2

        logger_binary_search.debug('guess level is {} |'.format(guess))

        consumption_more_than_guess = appliance_df['residue'] > guess
        hvac_contender_residue = appliance_df['residue'][hvac_contenders_in_month & consumption_more_than_guess]

        # updating gap between residue based on current guess
        gap_between_residue = (month_base_residue - np.sum(
            np.fmax(hvac_contender_residue - guess, 0)) / 1000) - residual_to_meet

    logger_binary_search.info(' final guess level is {} |'.format(guess))

    return guess, consumption_more_than_guess
