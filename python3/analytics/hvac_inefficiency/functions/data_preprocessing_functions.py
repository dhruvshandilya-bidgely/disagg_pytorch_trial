"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file contains code for pre processing data and data preparation
"""

# Import python packages

import copy
import logging
import datetime
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.time.get_time_diff import get_time_diff
from python3.analytics.hvac_inefficiency.utils.correct_ao_hvac import correct_ao_hvac
from python3.analytics.hvac_inefficiency.functions.process_hsm import use_previous_hsm
from python3.analytics.hvac_inefficiency.functions.season_marking import mark_and_analyse_season
from python3.analytics.hvac_inefficiency.functions.temporal_change_prep import prep_hvac_potential
from python3.analytics.hvac_inefficiency.functions.temporal_change_prep import prep_hvac_for_temporal_change
from python3.analytics.hvac_inefficiency.functions.temporal_change_prep import prep_weather_for_temporal_change


def preprocess_data(disagg_input_object, disagg_output_object, logger_pass):
    """
    Parameters:
        disagg_input_object                 (dict)          Dictionary containing all inputs
        disagg_output_object                (dict)          Dictionary containing all outputs
        logger_pass                         (object)        logger object

    Returns:
        input_hvac_inefficiency_object      (dict)          dictionary containing all input the information
        output_hvac_inefficiency_object     (dict)          dictionary containing all output information
    """

    # Initializing logger function

    logger_local = logger_pass.get("logger").getChild("preprocess_data")
    logger_pass = {"logger": logger_local, "logging_dict": logger_pass.get("logging_dict")}
    logger_inefficiency = logging.LoggerAdapter(logger_local, logger_pass.get("logging_dict"))

    global_config = disagg_input_object.get('config')
    column_idx_dict = disagg_output_object.get('hvac_debug').get('write').get('epoch_idx_dentify')

    uuid = global_config.get('uuid')
    sampling_rate = global_config.get('sampling_rate')

    input_inefficiency_object = dict({})
    output_inefficiency_object = dict({})

    input_inefficiency_object['uuid'] = uuid

    # Ensuring reproducibility by using common random state

    input_inefficiency_object['RANDOM_STATE'] = 43
    input_inefficiency_object['sampling_rate'] = sampling_rate
    input_inefficiency_object['pilot_id'] = global_config.get('pilot_id')
    input_inefficiency_object['raw_input_data'] = copy.deepcopy(disagg_input_object.get('input_data'))
    input_inefficiency_object['meta_data'] = copy.deepcopy(disagg_input_object.get('home_meta_data'))

    hvac_epoch_estimate = disagg_output_object.get('hvac_debug', {}).get('write', {}).get('epoch_ao_hvac_true', {})

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))

    # Store HVAC consumption data at TOU level

    ac_ao_column = column_idx_dict.get('ac_ao')
    sh_ao_column = column_idx_dict.get('sh_ao')
    ac_demand_column = column_idx_dict.get('ac_od')
    sh_demand_column = column_idx_dict.get('sh_od')

    ac_ao = copy.deepcopy(hvac_epoch_estimate[:, ac_ao_column])
    sh_ao = copy.deepcopy(hvac_epoch_estimate[:, sh_ao_column])

    sh_demand = copy.deepcopy(hvac_epoch_estimate[:, sh_demand_column])
    ac_demand = copy.deepcopy(hvac_epoch_estimate[:, ac_demand_column])

    day_idx = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_DAY_IDX])
    hour_of_day = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_HOD_IDX])

    epoch_data = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX])
    temperature = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_TEMPERATURE_IDX])
    net_consumption = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    cooling_potential = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_COOLING_POTENTIAL_IDX])
    heating_potential = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_HEATING_POTENTIAL_IDX])
    season_label = copy.deepcopy(input_data[:, Cgbdisagg.INPUT_S_LABEL_IDX])

    # Correct AO hvac consumption if there is demand HVAC

    if np.nansum(ac_demand) > 0:
        ac_demand, ac_ao = correct_ao_hvac(ac_demand, ac_ao)

    if np.nansum(sh_demand) > 0:
        sh_demand, sh_ao = correct_ao_hvac(sh_demand, sh_ao)

    raw_input_values = np.c_[epoch_data, day_idx, net_consumption, ac_ao, ac_demand, sh_ao, sh_demand, temperature,
                             hour_of_day, cooling_potential, heating_potential, season_label]

    # Preparing dictionary for column index and column value map

    raw_input_idx = {'epoch_data': 0,
                     'day_idx': 1,
                     'consumption': 2,
                     'ac_ao': 3,
                     'ac_demand': 4,
                     'sh_ao': 5,
                     'sh_demand': 6,
                     'temperature': 7,
                     'hour_of_day': 8,
                     'cooling_potential': 9,
                     'heating_potential': 10,
                     'season_label': 11}

    # Prepare HVAC potential data
    hvac_potential_input_data = dict({})
    feels_like_temperature = input_data[:, Cgbdisagg.INPUT_FEELS_LIKE_IDX]
    prec_data = input_data[:, Cgbdisagg.INPUT_PREC_IDX]
    snow_data = input_data[:, Cgbdisagg.INPUT_SNOW_IDX]

    hvac_potential_input_data['weather'] = dict({})
    hvac_potential_input_data['weather']['raw_weather'] = np.c_[epoch_data, day_idx, temperature,
                                                                feels_like_temperature, prec_data, snow_data,
                                                                cooling_potential, heating_potential, season_label]

    hvac_potential_input_data['meta_data'] = copy.deepcopy(disagg_input_object.get('home_meta_data'))

    input_inefficiency_object['raw_input_idx'] = raw_input_idx
    input_inefficiency_object['raw_input_values'] = raw_input_values
    input_inefficiency_object['days_of_data'] = len(np.unique(day_idx))
    input_inefficiency_object['temperature'] = copy.deepcopy(temperature.reshape(-1, 1))
    input_inefficiency_object['total_consumption'] = copy.deepcopy(net_consumption)

    cool_pot_pivot, heat_pot_pivot, s_label_pivot = prep_hvac_potential(input_inefficiency_object, logger_pass)

    # Find and attaching season summary
    season_marking = mark_and_analyse_season(s_label_pivot)
    input_inefficiency_object['seasons'] = season_marking

    input_inefficiency_object['uuid'] = global_config.get('uuid')

    # Preparing device based HVAC

    input_inefficiency_object['ac'] = dict({})
    input_inefficiency_object['sh'] = dict({})
    output_inefficiency_object['ac'] = dict({})
    output_inefficiency_object['sh'] = dict({})

    input_inefficiency_object['ac']['ao_column'] = raw_input_idx.get('ac_ao')
    input_inefficiency_object['ac']['demand_column'] = raw_input_idx.get('ac_demand')

    input_inefficiency_object['sh']['ao_column'] = raw_input_idx.get('sh_ao')
    input_inefficiency_object['sh']['demand_column'] = raw_input_idx.get('sh_demand')

    # Storing setpoint information to inefficiency dictionary

    input_inefficiency_object['ac']['setpoint'] =\
        disagg_output_object.get('hvac_debug').get('estimation').get('cdd').get('setpoint')

    input_inefficiency_object['sh']['setpoint'] =\
        disagg_output_object.get('hvac_debug').get('estimation').get('hdd').get('setpoint')

    cool_pot_pivot, heat_pot_pivot, s_label_pivot = prep_hvac_potential(input_inefficiency_object, logger_pass)

    # Prepare temperature information
    input_inefficiency_object, output_inefficiency_object =\
        prep_weather_for_temporal_change(input_inefficiency_object, output_inefficiency_object, logger_pass)

    sh_potential = heat_pot_pivot
    ac_potential = cool_pot_pivot

    row = input_inefficiency_object.get('sh').get('temperature_pivot').get('row')

    # Remove NaNs in HVAC potential
    if sh_potential is None:

        sh_potential =\
            np.zeros_like(input_inefficiency_object.get('sh').get('temperature_pivot').get('values'), dtype=float)

        row = input_inefficiency_object.get('sh').get('temperature_pivot').get('row')

    if ac_potential is None:

        ac_potential =\
            np.zeros_like(input_inefficiency_object.get('sh').get('temperature_pivot').get('values'), dtype=float)

        row = input_inefficiency_object.get('sh').get('temperature_pivot').get('row')

    input_inefficiency_object['sh']['sh_potential_pivot'] = {
        'values': sh_potential,
        'row': row,
    }

    input_inefficiency_object['ac']['ac_potential_pivot'] = {
        'values': ac_potential,
        'row': row,
    }

    # Prepare HVAC information for dictionary
    temp_time = datetime.datetime.now()

    input_inefficiency_object, output_inefficiency_object =\
        prep_hvac_for_temporal_change(input_inefficiency_object, output_inefficiency_object, logger_pass, 'ac')

    time_taken = get_time_diff(temp_time, datetime.datetime.now())
    logger_inefficiency.info('Time taken for preparing_data | ac | {}'.format(time_taken))

    temp_time = datetime.datetime.now()
    input_inefficiency_object, output_inefficiency_object =\
        prep_hvac_for_temporal_change(input_inefficiency_object, output_inefficiency_object, logger_pass, 'sh')

    time_taken = get_time_diff(temp_time, datetime.datetime.now())
    logger_inefficiency.info('Time taken for preparing_data | sh | {}'.format(time_taken))

    # Parse and use HSMs
    hsm_basic, col_map, hsm_dict = use_previous_hsm(disagg_input_object)
    input_inefficiency_object['hsm_information'] = {'basic': hsm_basic, 'dc_relation': hsm_dict, 'col_map': col_map}

    input_inefficiency_object['models'] = disagg_input_object.get('loaded_files', {}).get('hvac_inefficiency_files', {})

    # Prepare office goer probabilty

    office_goer_input =\
        {
            'winter': disagg_output_object.get('lifestyle_season', {}).get('winter', {}).get('office_goer_prob'),
            'summer': disagg_output_object.get('lifestyle_season', {}).get('summer', {}).get('office_goer_prob'),
            'transition': disagg_output_object.get('lifestyle_season', {}).get('transition', {}).get('office_goer_prob')
        }

    output_inefficiency_object['office_goer_probab'] = office_goer_input

    return input_inefficiency_object, output_inefficiency_object
