
"""
Author - Nisha Agarwal
Date - 3rd Nov 20
Calculate cleanliness score for each day
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def get_temperature_score(temp, config):

    """
    Calculate temperature dependent score

    Parameters:
        temp             (int)              : Temperature of the day
        config           (dict)             : config for clean day score calculation

    Returns:
        score            (float)            : temperature score
    """

    deviation = config.get('clean_day_score_config').get('temp_dev')

    score = np.exp(-(np.power(temp - config.get('clean_day_score_config').get('temp_setpoint'), 2) / (2 * deviation * deviation)))

    return score


def get_day_level_energy_score(consumption, perc_35, perc_25):

    """
    Calculate score based on consumption of the day

    Parameters:
        consumption     (float)             : day level consumption
        perc_25         (float)             : 25th percentile of the input data
        perc_35         (float)             : 35th percentile of the input data

    Returns:
        score           (float)             : day energy score
    """

    score = np.exp(-(np.power(consumption - perc_35, 2) / (2 * perc_25 * perc_25)))

    return score


def get_continuous_high_consumption_score(continuous_high_consumption_count, samples_per_hours, config):

    """
    Calculate max high consumption points score

    Parameters:
        continuous_high_consumption_count   (int)   : length of continuous high consumption points
        samples_per_hours                   (int)   : samples in an hour
        config           (dict)             : config for clean day score calculation

    Returns:
        score                               (float)  : continuous high consumption points score
    """

    continuous_high_consumption_count = continuous_high_consumption_count / samples_per_hours

    score = np.exp(-continuous_high_consumption_count / config.get('clean_day_score_config').get('cont_high_cons_count_factor'))

    return score


def get_high_consumption_score(high_consumption_count, samples_per_hours, config):

    """
    Calculate no of high cons points score

    Parameters:
        high_consumption_count        (np.ndarray)     : number of high consumption points
        samples_per_hours             (int)            : samples in an hour
        config                        (dict)           : config for clean day score calculation

    Returns:
        score                         (float)   : high consumption score
    """

    high_consumption_count = high_consumption_count / samples_per_hours

    score = np.exp(-high_consumption_count / config.get('clean_day_score_config').get('high_cons_count_factor'))

    return score


def get_continuous_segments_score(high_consumption_seq, samples_per_hours):

    """
    Calculate continuous high consumption segments score

    Parameters:
        high_consumption_seq        (list)          : length of all continuous high consumption points
        samples_per_hours           (int)           : samples in an hour

    Returns:
        score                       (float)         : continuous high consumption segments score
    """

    score = 1

    if not len(high_consumption_seq):
        return score

    high_consumption_seq = np.array(high_consumption_seq)
    high_consumption_seq = high_consumption_seq / samples_per_hours

    score = np.sum(np.power((Cgbdisagg.HRS_IN_DAY - high_consumption_seq) / Cgbdisagg.HRS_IN_DAY, 2))

    score = score / len(high_consumption_seq)

    return score


def get_clean_days_score(continuous_high_consumption_count, samples_per_hours, day_consumption, day_temperature,
                         perc_25, perc_35, high_consumption_seq, high_consumption_count, config):

    """
    Calculate cleanliness score for a given day

    Parameters:
        continuous_high_consumption_count   (int)        : length of longest continuous high consumption points
        samples_per_hours                   (int)        : samples in an hour
        day_consumption                     (int)        : Consumption of the day
        day_temperature                     (int)        : Temperature of the day
        perc_25                             (float)      : 25th percentile of the input data
        perc_35                             (float)      : 35th percentile of the input data
        high_consumption_seq                (list)       : length of all continuous high consumption points
        high_consumption_count              (np.ndarray) : number of high consumption points
        config                              (dict)       : dict containing all scoring config values

    Returns:
        final_score                 (float)      : final cleanliness score of the day
    """

    # temperature score

    temperature_score = get_temperature_score(day_temperature, config)

    # day level energy score

    day_level_energy_score = get_day_level_energy_score(day_consumption, perc_35, perc_25)

    # continuous high consumption point

    continuous_high_consumption_score = get_continuous_high_consumption_score(continuous_high_consumption_count, samples_per_hours, config)

    # score of continuity

    continuous_segments_score = get_continuous_segments_score(high_consumption_seq, samples_per_hours)

    # high consumption points

    high_consumption_score = get_high_consumption_score(high_consumption_count, samples_per_hours, config)

    final_score =  config.get('clean_day_score_weightage_config').get('temperature_score')*temperature_score + \
                   config.get('clean_day_score_weightage_config').get('day_level_energy_score')*day_level_energy_score + \
                   config.get('clean_day_score_weightage_config').get('continuous_high_consumption_score')*continuous_high_consumption_score + \
                   config.get('clean_day_score_weightage_config').get('continuous_segments_score')*continuous_segments_score + \
                   config.get('clean_day_score_weightage_config').get('high_consumption_score')*high_consumption_score

    return final_score
