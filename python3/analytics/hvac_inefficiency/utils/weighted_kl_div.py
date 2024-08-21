"""
Author - Anand Kumar Singh
Date - 26th June 2021
HVAC inefficiency HSM related operations
"""

# Import python packages

import numpy as np


def get_kl_divergence(first_distribution, second_distribution, weight=0.5):

    """
    Function measures the difference between Two normal distributions

    Parameters:
        first_distribution      (numpy.ndarray)         First Normal distribution
        second_distribution     (numpy.ndarray)         Second Normal Distribution
        weight                  (float)                 Weights for divergence
    Return:
        kl_divergence           (np.float64)            Measure of difference between between two distributions

    """

    temp = first_distribution * np.log(first_distribution.astype(float) / second_distribution.astype(float))
    temp = temp.astype(float)

    start = 1 - weight
    end = 1 + weight

    weights = np.linspace(start, end, num=len(first_distribution))

    temp[np.logical_or(np.isnan(temp), np.isinf(temp))] = 0
    kl_div = np.sum(temp * weights)

    return kl_div


def get_divergence_score(end_year_relation, start_year_relation, weight=0.3):

    """
    Function to get divergence score

    Parameters:
        end_year_relation       (np.ndarray)    : End year relation
        start_year_relation     (np.ndarray)    : Start year relation
        weight                  (float)         : weightage

    Returns:
        divergence_score        (float)         : divergence score
        length_dc_relationship (np.ndarray)     : relationship
    """

    temp_col = 0
    duty_cycle_col = 1
    end_dc_col = 1
    start_dc_col = 2

    unique_temp = np.r_[end_year_relation[:, temp_col], start_year_relation[:, temp_col]]
    unique_temp = np.unique(unique_temp)

    if len(unique_temp) == 0:
        divergence_score = np.nan
        length_dc_relationship = 0
        return divergence_score, length_dc_relationship

    # Initialise

    concat_dc_relationship = []

    for temperature in unique_temp:
        val_idx = start_year_relation[:, temp_col] == temperature
        if val_idx.sum() == 0:
            start_dc_val = np.nan
        else:
            start_dc_val = start_year_relation[val_idx, duty_cycle_col]

        val_idx = end_year_relation[:, temp_col] == temperature
        if val_idx.sum() == 0:
            end_dc_val = np.nan
        else:
            end_dc_val = end_year_relation[val_idx, duty_cycle_col]

        concat_dc_relationship.append([temperature, end_dc_val, start_dc_val])

    concat_dc_relationship = np.array(concat_dc_relationship)

    zero_duty_cycle_idx = (concat_dc_relationship[:, end_dc_col] != 0) & (concat_dc_relationship[:, start_dc_col] != 0)
    concat_dc_relationship = concat_dc_relationship[zero_duty_cycle_idx]

    if concat_dc_relationship.shape[0] == 0:
        divergence_score = np.nan
    else:
        concat_dc_relationship = concat_dc_relationship[concat_dc_relationship[:, temp_col].argsort()]
        divergence_score = get_kl_divergence(concat_dc_relationship[:, end_dc_col],
                                             concat_dc_relationship[:, start_dc_col], weight=weight)

    nan_idx = np.isnan(concat_dc_relationship[:, start_dc_col].astype(float))
    length_dc_relationship = concat_dc_relationship[~nan_idx].shape[0]

    return divergence_score, length_dc_relationship
