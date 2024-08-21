"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module contains the operations related to hsm
"""

# Import python packages

import logging

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants


def get_hsm(disagg_input_object, global_config, logger_base):
    """
    Parameters:
        disagg_input_object     (dict)          : Dictionary containing all inputs
        global_config           (dict)          : Dictionary containing all input configuration
        logger_base             (dict)          : Logger object

    Returns:
        hsm_in                  (dict)          : Water heater HSM (Home Specific Model)
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_hsm')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Try extracting the hsm from input object

    try:
        hsm_dict = disagg_input_object.get('appliances_hsm')
        hsm_in = hsm_dict.get('wh')
        logger.info('HSM retrieved from the disagg input object | ')
    except KeyError:
        # If hsm not found

        hsm_in = None
        logger.info('HSM not found in the disagg input object | ')

    # Get the disagg mode from the global config

    disagg_mode = global_config.get('disagg_mode')

    # Check if HSM attributes present for MTD mode

    hsm_fail = ((hsm_in is None) or (len(hsm_in) == 0) or (len(hsm_in.get('attributes')) == 0)) and \
               (disagg_mode == 'mtd')

    return hsm_in, hsm_fail


def make_hsm_attributes(debug, wh_config, logger):
    """
    Parameters:
        debug           (dict)                  : Contains data saved at each step of algorithm
        wh_config       (dict)                  : The config params for the algorithm
        logger          (logger)                : The logger object to write logs

    Returns:
        attributes      (dict)                  : Dictionary with hsm parameters
    """

    # Initialize the hsm attributes with hld values
    # timed_hld             : Timed water heater hld
    # timed_wh_amplitude    : Timed water heater amplitude
    # thermostat_hld        : Thermostat water heater hld

    attributes = {
        'timed_hld': debug.get('timed_hld'),
        'timed_confidence_score': debug.get('timed_confidence_score'),
        'timed_wh_amplitude': debug.get('timed_wh_amplitude'),
        'thermostat_hld': debug.get('thermostat_hld')
    }

    # Add values to hsm based on detection of water heater

    if debug['thermostat_hld'] == 1:
        # If thermostat water heater detected, add relevant values

        attributes['thin_fat_ratio'] = debug['thin_fat_ratio']

        attributes['thin_scale_factor'] = debug['thin_scale_factor']
        attributes['fat_scale_factor'] = debug['fat_scale_factor']

        # Populate the attributes for each season

        for season in wh_config['seasons']:
            # Retrieve features dict for current season

            season_features = debug['season_features'][season]

            # Thin pulse attributes for each season

            attributes[season + '_' + 'duration'] = season_features['duration']
            attributes[season + '_' + 'thin_peak_energy'] = season_features['thin_peak_energy']
            attributes[season + '_' + 'inter_pulse_gap'] = season_features['inter_pulse_gap']
            attributes[season + '_' + 'thin_peak_energy_range'] = season_features['thin_peak_energy_range']

            # Fat pulse attributes for each season

            attributes[season + '_' + 'fat_hours'] = season_features['fat_hours']
            attributes[season + '_' + 'best_fat_amp'] = season_features['best_fat_amp']
            attributes[season + '_' + 'lower_fat_amp'] = season_features['lower_fat_amp']
            attributes[season + '_' + 'upper_fat_amp'] = season_features['upper_fat_amp']

    elif debug['timed_hld'] == 1 and debug['pilot_id'] in PilotConstants.TIMED_WH_JAPAN_PILOTS:
        attributes['twh_time_bands'] = debug.get('timed_debug').get('twh_time_bands')
        attributes['twh_time_band_scores'] = debug.get('timed_debug').get('twh_time_band_scores')

    else:
        # Thermostat water heater not present

        logger.info('No HSM for thermostat water heater | ')

    return attributes


def check_hsm_validity(debug, logger_base):
    """
    Check if a valid hsm present for MTD mode

    Parameters:
        debug           (dict)              : Algorithm intermediate steps output
        logger_base     (dict)              : Logger object

    Returns:
        valid_hsm       (bool)              : The boolean to mark validity of the HSM
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('check_hsm_validity')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Retrieve disagg_mode and hsm from debug object

    hsm_in = debug.get('hsm_in')
    disagg_mode = debug.get('disagg_mode')

    # Initialize default hsm validity to True

    valid_hsm = True

    # If the disagg mode is MTD

    if disagg_mode == 'mtd':
        # If the HSM is not None, check for thin_peak_energy values

        if hsm_in is not None:
            # If valid hsm present, check for thin_peak_energy values

            wtr_thin_peak_energy = hsm_in.get('wtr_thin_peak_energy')
            itr_thin_peak_energy = hsm_in.get('itr_thin_peak_energy')
            smr_thin_peak_energy = hsm_in.get('smr_thin_peak_energy')
            twh_time_bands = hsm_in.get('twh_time_bands')

            # HSM valid if any valid seasonal 'thin_peak_energy' value found
            if debug['pilot_id'] in PilotConstants.TIMED_WH_JAPAN_PILOTS:

                if twh_time_bands is None:
                    valid_hsm = False

            else:
                if (wtr_thin_peak_energy is None) and (itr_thin_peak_energy is None) and (smr_thin_peak_energy is None):

                    # All invalid 'thin_peak_energy' values

                    valid_hsm = False

                logger.info('HSM invalid because no valid thin_peak_energy values found | ')
        else:
            # If HSM None in MTD mode

            valid_hsm = False

            logger.info('HSM invalid because HSM is None for mode {} | '.format(disagg_mode))

    return valid_hsm


def make_hsm_from_debug(input_data, debug, wh_config, error_list, logger):
    """
    Parameters:
        input_data      (np.ndarray)            : Raw data input for the user
        debug           (dict)                  : The dictionary containing all module level output
        wh_config       (dict)                  : Water heater parameters
        error_list      (list)                  : List of errors encountered in the code run
        logger          (logger)                : The logger object to write logs

    Returns:
        wh_hsm          (dict)                  : The new hsm dictionary
        debug           (dict)                  : Updated debug object
        error_list      (list)                  : The list of errors
    """

    wh_hsm = dict({})

    # If the mode is historical/incremental, make hsm

    if debug['make_hsm']:
        # Extract the relevant values from debug dict to create hsm

        wh_hsm = {
            'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': make_hsm_attributes(debug, wh_config, logger)
        }

        # Saving new HSM to the debug object

        debug['hsm'] = wh_hsm

        logger.info('Writing new HSM for the run | ')
    else:
        logger.info('Not writing HSM for this mode | ')

    return wh_hsm, debug, error_list


def get_hsm_values(hsm, season, logger_base):
    """
    Parameters:
        hsm                     (dict)          : Input HSM
        season                  (str)           : Season of the data
        logger_base             (logger)        : Logger object

    Returns:
        thin_peak_energy        (float)         : Thin pulse amplitude
        inter_pulse_gap         (float)         : Gap between thin pulses (in hours)
        duration                (float)         : Duration of thin pulse
        lower_fat_amp           (float)         : Lower fat energy bound
        best_fat_amp            (float)         : Best fat energy bound
        upper_fat_amp           (float)         : Upper fat energy bound
        thin_peak_energy_range  (np.ndarray)    : Thin pulse energy range
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_hsm_values')

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Try to extract the values from the hsm

    try:
        # If values taken up from old hsm (MATLAB pipeline)

        duration = hsm.get(season + '_' + 'duration')[0]
        thin_peak_energy = hsm.get(season + '_' + 'thin_peak_energy')[0]
        thin_peak_energy_range = hsm.get(season + '_' + 'thin_peak_energy_range')

        best_fat_amp = hsm.get(season + '_' + 'best_fat_amp')[0]
        lower_fat_amp = hsm.get(season + '_' + 'lower_fat_amp')[0]
        upper_fat_amp = hsm.get(season + '_' + 'upper_fat_amp')[0]

        inter_pulse_gap = hsm.get(season + '_' + 'inter_pulse_gap')[0]

        logger.info('HSM values present as array | ')

    except IndexError:
        # If values taken up from new hsm (Python pipeline)

        duration = hsm.get(season + '_' + 'duration')
        thin_peak_energy = hsm.get(season + '_' + 'thin_peak_energy')
        thin_peak_energy_range = hsm.get(season + '_' + 'thin_peak_energy_range')

        best_fat_amp = hsm.get(season + '_' + 'best_fat_amp')
        lower_fat_amp = hsm.get(season + '_' + 'lower_fat_amp')
        upper_fat_amp = hsm.get(season + '_' + 'upper_fat_amp')

        inter_pulse_gap = hsm.get(season + '_' + 'inter_pulse_gap')

        logger.info('HSM values not present as array | ')

    # Log all the extracted params

    logger.info('The params loaded from hsm are as follows | ')

    logger.info('thin_peak_energy | {}'.format(thin_peak_energy))
    logger.info('duration | {}'.format(duration))
    logger.info('thin_peak_energy_range | {}'.format(thin_peak_energy_range))

    logger.info('best_fat_amp | {}'.format(best_fat_amp))
    logger.info('lower_fat_amp | {}'.format(lower_fat_amp))
    logger.info('upper_fat_amp | {}'.format(upper_fat_amp))

    logger.info('inter_pulse_gap | {}'.format(inter_pulse_gap))

    # Stack all the hsm extracted parameters to a list

    final_hsm_attributes = [thin_peak_energy, inter_pulse_gap, duration, lower_fat_amp, best_fat_amp, upper_fat_amp,
                            thin_peak_energy_range]

    return final_hsm_attributes
