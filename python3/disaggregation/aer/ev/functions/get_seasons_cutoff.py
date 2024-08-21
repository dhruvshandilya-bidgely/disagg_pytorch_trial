"""
Author - Nikhil Singh Chauhan
Date - 15-May-2020
Module to find seasonal cutoffs for the user
"""

# Import python packages

import numpy as np


def get_seasons_cutoff(seasons_info, season_columns, transition_idx, unique_months, one_season, setpoint=65, width=9):
    """
    Used in the mtd mode to give one season for the whole data

    Parameters:
        seasons_info        (np.ndarray)    : Seasons information
        season_columns      (dict)          : The columns of season data and the corresponding numbers
        transition_idx      (np.ndarray)    : Transition index
        unique_months       (np.ndarray)    : Bill cycle timestamps
        one_season          (bool)          : Flag for one season
        setpoint            (int)           : Default setpoint
        width               (int)           : Allowed temperature variation

    Returns:
        wtr_cutoff          (float)         : Winter temperature upper limit
        smr_cutoff          (float)         : Summer temperature lower limit
    """

    # Calculate the temperature variation allowed with respect to setpoint

    wtr_bound = setpoint - width
    smr_bound = setpoint + width

    if one_season:
        # If only one season to be returned, get season of last bill cycle

        last_bill_cycle_ts = unique_months[-1]
        last_bill_cycle_temp = seasons_info[seasons_info[:, season_columns['bill_cycle_ts']] ==
                                            last_bill_cycle_ts].squeeze()[1]

        if last_bill_cycle_temp < wtr_bound:
            # If last bill cycle season is winter

            wtr_cutoff = np.max(seasons_info[:, season_columns['average_temp']]) + 1
            smr_cutoff = wtr_cutoff + 1

        elif last_bill_cycle_temp > smr_bound:
            # If last bill cycle season is summer

            smr_cutoff = np.min(seasons_info[:, season_columns['average_temp']]) - 1
            wtr_cutoff = smr_cutoff - 1
        else:
            # If last bill cycle season is transition

            wtr_cutoff = np.min(seasons_info[:, season_columns['average_temp']]) - 1
            smr_cutoff = np.max(seasons_info[:, season_columns['average_temp']]) + 1
    else:
        # Return season cutoff on winter and summer bounds

        wtr_cutoff = np.fmin(seasons_info[transition_idx[0], 1], wtr_bound)
        smr_cutoff = np.fmax(seasons_info[transition_idx[-1], 1], smr_bound)

    return wtr_cutoff, smr_cutoff
