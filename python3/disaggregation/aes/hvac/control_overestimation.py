"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to regulate HVAC over-estimations
"""

# Import python packages
import logging
import numpy as np


def get_guess_in_overestimation(hvac_contenders_in_month, appliance_df, hvac, required_reduction, residual_to_meet,
                                month_base_residue, logger_base):

    """
    Function to get the cut point, below which hvac estimates are moved to residue from hvac

    Parameters:

        hvac_contenders_in_month(np.ndarray)   : Array containing booleans of qualified entries w.r.t ac/sh
        appliance_df            (Pd.Dataframe) : Dataframe containing hvac consumption
        hvac                    (str)          : String identifier of whether hvac is ac or sh
        required_reduction      (float)        : The required amount of reduction to handle overestimation
        residual_to_meet        (float)        : The desired residual in month
        month_base_residue      (float)        : The original residue from month
        logger_base             (logging object): Keeps log of code flow

    Returns:

        guess             (float)             : The cut point, below which hvac estimates are moved to residue from hvac
        valid_consumption (np.ndarray)        : Array containing booleans where consumption is greater than guess
    """

    # initializing logger
    logger_local = logger_base.get("logger").getChild("get_guess")
    logger_get_guess = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # initializing hvac array
    concerned_hvac_array = np.zeros((len(hvac_contenders_in_month)))
    concerned_hvac_array[hvac_contenders_in_month] = appliance_df[hvac][hvac_contenders_in_month]

    # sorting hvac array
    concerned_hvac_array_sorted = np.sort(concerned_hvac_array)

    # sorting valid hvac array
    non_zero_concerned_hvac_array_sorted = concerned_hvac_array_sorted[concerned_hvac_array_sorted > 0]
    non_zero_concerned_hvac_array_sorted_diff = (np.diff(non_zero_concerned_hvac_array_sorted.T) != 0).astype(int)
    non_zero_concerned_hvac_array_sorted_diff = np.r_[0, non_zero_concerned_hvac_array_sorted_diff]
    non_zero_concerned_hvac_array_sorted_diff_idx = np.argwhere(non_zero_concerned_hvac_array_sorted_diff == 1)
    non_zero_concerned_hvac_array_sorted_diff_idx = np.r_[0, non_zero_concerned_hvac_array_sorted_diff_idx[:, 0]]

    # initializing cut level
    guesses = []
    guesses_idx = []
    residues_from_hvac = []
    non_zero_array_length = len(non_zero_concerned_hvac_array_sorted)

    # Finding cut level on valid hvac array
    for idx in non_zero_concerned_hvac_array_sorted_diff_idx:
        guess = non_zero_concerned_hvac_array_sorted[idx]
        guesses.append(guess)
        guesses_idx.append(idx)

        residue_from_hvac = guess * (non_zero_array_length - idx) / 1000
        residues_from_hvac.append(residue_from_hvac)

    # Checking current distance from required residue
    distance_from_required_residue = np.array(residues_from_hvac) - required_reduction

    # Making best guess on cut level based on current distance from required residue
    try:

        best_guess_bet = np.min(distance_from_required_residue[distance_from_required_residue > 0])
        best_guess_bet_location = np.argwhere(distance_from_required_residue == best_guess_bet)[0][0]

        best_guess_idx = guesses_idx[best_guess_bet_location]
        best_guess = non_zero_concerned_hvac_array_sorted[best_guess_idx]

        valid_consumption = appliance_df[hvac] >= best_guess
        residue_contender_in_hvac = appliance_df[hvac][hvac_contenders_in_month & valid_consumption]

        # Assigning guess level found
        guess = (residual_to_meet - month_base_residue) * 1000 / residue_contender_in_hvac.shape[0]

        logger_get_guess.debug("Guess level found |")

    except (ValueError, IndexError, KeyError):

        # Checking failsafe for cut level
        best_guess_bet = np.max(distance_from_required_residue[distance_from_required_residue < 0])
        best_guess_bet_location = np.argwhere(distance_from_required_residue == best_guess_bet)[0][0]

        best_guess_idx = guesses_idx[best_guess_bet_location]
        best_guess = non_zero_concerned_hvac_array_sorted[best_guess_idx]

        valid_consumption = appliance_df[hvac] >= best_guess

        # Assigning best guess level found
        guess = best_guess

        logger_get_guess.debug("Failsafe guess found |")

    logger_get_guess.debug("guess level found is : {} |".format(guess))

    return valid_consumption, guess
