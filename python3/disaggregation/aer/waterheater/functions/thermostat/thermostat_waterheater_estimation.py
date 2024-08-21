"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
This module detects the active usage of Water Heater (referred as Fat pulse) and estimate the consumption
"""

# Import python packages

import logging
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.hsm_utils import get_hsm_values
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_fat_boxes import get_fat_boxes
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.get_fat_pulse_range import get_fat_energy_range
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.thin_pulse_estimation import get_inter_pulse_gap
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.fat_pulse_estimation import fat_pulse_consumption
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.thin_pulse_estimation import find_thin_peak_energy
from python3.disaggregation.aer.waterheater.functions.thermostat.functions.thin_pulse_estimation import thin_pulse_estimation

from python3.disaggregation.aer.waterheater.functions.thermostat.functions.combine_seasons_output import combining_seasons_output


def get_seasonal_estimation(debug, wh_config, logger_base):
    """
    Parameters:
        debug               (dict)      : Algorithm intermediate steps output
        wh_config           (dict)      : Config params
        logger_base         (dict)      : Logger object

    Returns:
        debug               (dict)      : Algorithm intermediate steps output
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('get_seasonal_estimation')
    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Taking a deepcopy of input data to keep local instances

    all_features = deepcopy(debug['season_features'])

    # Check if hsm to be used

    if debug['use_hsm']:
        # If MTD run mode, use hsm

        logger.info('Using hsm for estimation | ')

        # Extract hsm from debug object

        hsm_in = debug['hsm_in']

        # Extract season info and features for the given data

        season = debug['season']
        season_features = debug['season_features'][season]

        # Extract the potential fat hours info from hsm for this season

        debug['possible_fat_hours'] = hsm_in.get(season + '_' + 'fat_hours')

        # Retrieve all the values from hsm

        attributes_from_hsm = get_hsm_values(hsm_in, season, logger_pass)

        # Extract the variables from the hsm output list

        thin_peak_energy, inter_pulse_gap, dur, lower_fat_amp, best_fat_amp, upper_fat_amp, \
        thin_peak_energy_range = attributes_from_hsm

        # Thin pulse estimation

        thin_pulse_output, max_thin_output = thin_pulse_estimation(season_features, thin_peak_energy,
                                                                   thin_peak_energy_range, inter_pulse_gap, wh_config,
                                                                   season, logger_pass)

        # Subtract the thin pulse consumption from raw consumption

        season_features['data'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= \
            thin_pulse_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # Find potential area of active usage of water heater (fat pulses)

        box_data = get_fat_boxes(season_features['data'], wh_config, lower_fat_amp, logger_pass)

        # Fat pulse estimation

        fat_hours, fat_pulse_output, final_fat_boxes, debug = fat_pulse_consumption(season_features['data'], box_data,
                                                                                    wh_config, best_fat_amp,
                                                                                    upper_fat_amp, debug, logger_pass)

        # Saving seasonal output to debug object

        debug['final_box_output'] = box_data[:, :Cgbdisagg.INPUT_DIMENSION]
        debug['final_thin_output'] = thin_pulse_output[:, :Cgbdisagg.INPUT_DIMENSION]
        debug['final_fat_output'] = fat_pulse_output[:, :Cgbdisagg.INPUT_DIMENSION]

        # Save final output by adding fat pulse output to thin pulse output

        final_output = deepcopy(debug['final_thin_output'])
        final_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] += fat_pulse_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # Calculate the residual consumption after removing water heater

        residual = deepcopy(debug['input_data'])
        residual[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= final_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

        # Save final water heater consumption and residual to debug object

        debug['final_wh_signal'] = final_output
        debug['residual'] = residual
        debug['new_fat_amp'] = best_fat_amp
        debug['thin_peak_energy'] = thin_peak_energy

    else:
        # If historical / incremental run mode

        # Iterate over each season

        for season in ['wtr', 'itr', 'smr']:
            logger.info('Processing for season: | {}'.format(season))

            # Extract features of current seasons

            season_features = all_features[season]

            # Calculate thin_peak_energy for the current season

            thin_peak_energy = find_thin_peak_energy(season_features['features'], wh_config, logger)

            # Calculate inter pulse gap for current season

            inter_pulse_gap = get_inter_pulse_gap(season_features, wh_config, logger)

            # Extract the thin pulse duration model

            duration_model = debug['models']['thin_model']

            # Get the list energy range for thin pulse and fat pulse

            energy_ranges = get_fat_energy_range(thin_peak_energy, wh_config, duration_model, logger)

            # Retrieve the values from energy ranges

            dur, lower_fat_amp, best_fat_amp, upper_fat_amp, thin_peak_energy_range = energy_ranges

            # Thin pulse consumption

            thin_pulse_output, max_thin_output = thin_pulse_estimation(season_features, thin_peak_energy,
                                                                       thin_peak_energy_range, inter_pulse_gap,
                                                                       wh_config, season, logger_pass)

            # Subtract the thin pulse consumption from raw consumption

            season_features['data'][:, Cgbdisagg.INPUT_CONSUMPTION_IDX] -= \
                thin_pulse_output[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]

            # Find potential area of active usage of water heater (fat pulses)

            box_data = get_fat_boxes(season_features['data'], wh_config, lower_fat_amp, logger_pass)

            # Fat pulse estimation

            fat_hours, fat_pulse_output, final_fat_boxes, debug = fat_pulse_consumption(season_features['data'],
                                                                                        box_data, wh_config,
                                                                                        best_fat_amp, upper_fat_amp,
                                                                                        debug, logger_pass)
            # Saving thin pulse params for this season to debug object

            season_features['thin_peak_energy'] = thin_peak_energy
            season_features['duration'] = dur
            season_features['thin_peak_energy_range'] = thin_peak_energy_range
            season_features['inter_pulse_gap'] = inter_pulse_gap

            # Saving fat pulse params for this season to debug object

            season_features['fat_hours'] = fat_hours
            season_features['best_fat_amp'] = best_fat_amp
            season_features['lower_fat_amp'] = lower_fat_amp
            season_features['upper_fat_amp'] = upper_fat_amp

            # Saving output consumption for this season to debug object

            season_features['box_output'] = box_data
            season_features['thin_output'] = thin_pulse_output
            season_features['fat_output'] = fat_pulse_output
            season_features['max_thin_output'] = max_thin_output
            season_features['final_fat_pulses'] = final_fat_boxes

            # Update the old season features with new for current season

            all_features[season] = season_features

        # Save the updated season features to debug object

        debug['season_features'] = all_features

        # Calculate the overall thin_peak_energy using all season features

        debug['thin_peak_energy'] = find_thin_peak_energy(debug["all_features"], wh_config, logger, False)

        # Combine all the seasonal consumption data frames to one

        debug = combining_seasons_output(debug, logger_pass)

    return debug
