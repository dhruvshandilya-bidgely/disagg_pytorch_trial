
"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Prepare data required in itemization pipeline
"""

# Import python packages

import copy
import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.get_app_ids import app_id_count

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.functions.get_day_data import get_hybrid_day_data

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.initialisations.fetch_hybrid_v2_pilot_config_info import convert_conf_from_csv_to_json
from python3.itemization.initialisations.fetch_hybrid_v2_pilot_config_info import update_bounds

from python3.itemization.initialisations.utils_for_preparing_hybrid_input_data import get_app_list
from python3.itemization.initialisations.utils_for_preparing_hybrid_input_data import fetch_wh_app_prof_info
from python3.itemization.initialisations.utils_for_preparing_hybrid_input_data import prepare_hsm_data
from python3.itemization.initialisations.utils_for_preparing_hybrid_input_data import update_object_with_disagg_debug_params


def prepare_input_day_data(item_input_object, input_data, output_data, appliance_list, vacation, logger):

    """
    Preprocessing steps for day input data

    Parameters:
        item_input_object           (dict)         : Dict containing all inputs
        input_data                  (np.ndarray)   : Day level raw consumption values
        output_data                 (np.ndarray)   : Day level disagg output data
        appliance_list              (list)         : list of appliances
        vacation                    (np.ndarray)   : vacation output
        logger                      (np.ndarray)   : logger object

    Returns:
        input_data                  (np.ndarray)   : Day level raw consumption values (timed and high consumption removed)
        input_data_without_wh       (np.ndarray)   : Day level raw consumption values without wh output
        pp_user                     (bool)         : True if the user is PP user
        ev_user                     (bool)         : True if the user is EV user
        ao_cons                     (float)        : AO consumption of the user
        timed_wh_user               (bool)         : True if the user is TWH user
        removed_cons                (np.ndarray)   : Total TS level consumption removed (PP + EV + TWH)
    """

    input_data = np.fmax(0, input_data)

    samples_per_hour = int(len(input_data[0]) / Cgbdisagg.HRS_IN_DAY)

    # Remove high percentile values

    input_data[input_data > np.percentile(input_data, 95)] = np.percentile(input_data, 95)

    ao_index = np.where(np.array(appliance_list) == 'ao')[0]

    logger.debug("Removed outlier consumption points")

    pp_user = 0
    ev_user = 0
    ao_cons = 0

    if not np.all(np.logical_or(output_data[ao_index[0], :, :] == 0, np.isnan(output_data[ao_index[0], :, :]))):
        ao_cons = output_data[ao_index[0] + 1, :, :]
        ao_cons = ao_cons[np.logical_not(vacation[:, 0])]
        ao_cons = np.median(np.sum(ao_cons, axis=1))

    wh_index = np.where(np.array(appliance_list) == 'wh')[0]

    timed_wh_user = 0

    # Remove Timed water heater consumption

    input_data_without_wh = copy.deepcopy(input_data)

    if len(wh_index) and \
            not np.all(np.logical_or(output_data[wh_index[0] + 1, :, :] == 0, np.isnan(output_data[wh_index[0] + 1, :, :]))):

        wh_consumption = output_data[wh_index[0] + 1, :, :]
        wh_consumption = np.nan_to_num(wh_consumption)

        input_data_without_wh = input_data_without_wh - wh_consumption

    input_data = np.nan_to_num(input_data)
    input_data = np.fmax(input_data, 0)

    input_data = np.fmax(0, input_data)

    config = init_itemization_params().get('data_preparation')

    pp_index = np.where(np.array(appliance_list) == 'pp')[0]

    logger.debug("Removed outlier consumption points")

    timed_consumption = np.zeros(input_data.shape)
    removed_cons = np.zeros(input_data.shape)

    # remove Pool pump

    if len(pp_index) and not np.all(np.logical_or(output_data[pp_index[0] + 1, :, :] == 0,
                                                  np.isnan(output_data[pp_index[0] + 1, :, :]))):

        pp_consumption = output_data[pp_index[0] + 1, :, :]
        pp_consumption = np.nan_to_num(pp_consumption)

        median = np.median(pp_consumption[np.nonzero(pp_consumption)])
        pp_consumption[pp_consumption > config.get("pp_amp_limit")/samples_per_hour] = median

        input_data = input_data - pp_consumption

        pp_user = 1

        timed_consumption = timed_consumption + pp_consumption

        logger.debug("Subtracted PP from input data")

    # Remove EV consumption

    ev_index = np.where(np.array(appliance_list) == 'ev')[0]

    if len(ev_index) and \
            not np.all(np.logical_or(output_data[ev_index[0] + 1, :, :] == 0,
                                     np.isnan(output_data[ev_index[0] + 1, :, :]))):

        ev_consumption = output_data[ev_index[0] + 1, :, :]
        ev_consumption = np.nan_to_num(ev_consumption)

        input_data = input_data - ev_consumption

        timed_consumption = timed_consumption + ev_consumption

        ev_user = 1

        removed_cons = removed_cons + ev_consumption

        logger.debug("Subtracted EV from input data")


    wh_index = np.where(np.array(appliance_list) == 'wh')[0]

    # Remove Timed water heater consumption

    if item_input_object.get("disagg_special_outputs") is None:
        twh_consumption = None
    else:
        twh_consumption = item_input_object.get("disagg_special_outputs").get("timed_water_heater")

    if twh_consumption is None:
        twh_consumption = 0

    elif np.shape(twh_consumption)[1] == 2:
        twh_consumption = np.nan_to_num(twh_consumption[:, 1])

    elif np.shape(twh_consumption)[1] == Cgbdisagg.INPUT_DIMENSION:
        twh_consumption = np.nan_to_num(twh_consumption[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    if len(wh_index) and np.sum(twh_consumption):

        timed_wh_user = 1

        wh_consumption = output_data[wh_index[0] + 1, :, :]
        wh_consumption = np.nan_to_num(wh_consumption)

        input_data = input_data - wh_consumption

        timed_consumption = timed_consumption + wh_consumption

        logger.debug("Subtracted WH from input data")

    input_data = np.nan_to_num(input_data)
    input_data = np.fmax(input_data, 0)

    input_data = fill_input_data_after_app_removal(input_data, timed_consumption, samples_per_hour)

    return input_data, input_data_without_wh, pp_user, ev_user, ao_cons, timed_wh_user, removed_cons


def fill_input_data_after_app_removal(input_data, timed_consumption, samples_per_hour):

    """
    Fill input data after timed appliance removal

    Parameters:
        input_data              (np.ndarray)        : Day input data
        timed_consumption       (np.ndarray)        : timed app consumption
        samples_per_hour        (int)               : samples in an hour

    Returns:
        input_data              (np.ndarray)        : Day input data
    """

    seq_config = init_itemization_params().get("seq_config")

    length = len(timed_consumption[0])

    if np.any(timed_consumption > 0):

        timed_tou = np.sum(timed_consumption > 0, axis=0) / len(timed_consumption)
        timed_day = np.sum(timed_consumption > 0, axis=1) > 0

        threshold = 0.05

        if np.any(timed_tou > threshold):
            seq = find_seq(timed_tou > threshold, np.zeros(length), np.zeros(length))

            seq = seq[seq[:, seq_config.get("label")] == 1]

            for index in range(len(seq)):
                start = seq[index, seq_config.get("start")]
                end = seq[index, seq_config.get("end")]

                consumption = np.sum(input_data[:, get_index_array(start - samples_per_hour, start - 1, length)], axis=1) + \
                              np.sum(input_data[:, get_index_array(end + 1, end + samples_per_hour, length)], axis=1)

                consumption = consumption / (2*samples_per_hour)

                temp_cons = np.zeros(input_data.shape)
                temp_cons[:, get_index_array(start, end, length)] = consumption[:, None]

                temp_cons[timed_consumption == 0] = 0

                input_data[timed_day] = np.maximum(input_data[timed_day], temp_cons[timed_day])

    return input_data


def prepare_data(item_input_object, item_output_object, logger_pass):

    """
    Prepare hybrid input object

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger_pass               (dict)      : Contains base logger and logging dictionary

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
    """

    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild('prepare_data')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_prepare_data_start = datetime.now()

    # Fetch necessary information from disagg pipeline output

    input_data = copy.deepcopy(item_input_object.get("input_data"))
    output_data = copy.deepcopy(item_input_object.get("disagg_epoch_estimate"))

    sampling_rate = item_input_object.get("config").get("sampling_rate")

    item_input_object['config']['sampling_rate'] = sampling_rate

    samples_per_hour = int(Cgbdisagg.SEC_IN_HOUR / sampling_rate)

    # Prepare output appliance list

    item_input_object.get("disagg_output_write_idx_map")['ld'] = 12
    item_input_object.get("disagg_output_write_idx_map")['ent'] = 13
    item_input_object.get("disagg_output_write_idx_map")['cook'] = 14

    # fetching list of target app

    appliance_list, appliance_output_index, output_data, solar_gen = get_app_list(item_input_object, output_data)

    t3 = datetime.now()

    if item_input_object.get("disagg_special_outputs") is None:
        twh_consumption = None
    else:
        twh_consumption = item_input_object.get("disagg_special_outputs").get("timed_water_heater")

    if twh_consumption is None:
        twh_consumption = 0

    elif np.shape(twh_consumption)[1] == 2:
        twh_consumption = np.nan_to_num(twh_consumption[:, 1])

    elif np.shape(twh_consumption)[1] == Cgbdisagg.INPUT_DIMENSION:
        twh_consumption = np.nan_to_num(twh_consumption[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    if "wh" in appliance_list and np.sum(twh_consumption):
        index = appliance_list.index("wh") + 1
        output_data[:, index] = twh_consumption
        logger.info('Timed Water heater output is non-zero')

    input_data = np.nan_to_num(input_data)
    output_data = np.nan_to_num(output_data)

    # prepare sunrise data

    tod = (input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX] - input_data[:, Cgbdisagg.INPUT_DAY_IDX]) / Cgbdisagg.SEC_IN_HOUR
    tod[tod < 0] = 0
    minutes = tod - tod.astype(int)
    minutes = np.digitize(minutes, [0, 0.25, 0.5, 0.75, 1]) / 4
    tod = (tod.astype(int) + minutes)
    input_data[:, Cgbdisagg.INPUT_SUNRISE_IDX] = tod

    t4 = datetime.now()

    logger.info("Preparation of sunrise data took  | %.3f s", get_time_diff(t3, t4))

    # prepare sunset data

    tod = (input_data[:, Cgbdisagg.INPUT_SUNSET_IDX] - input_data[:, Cgbdisagg.INPUT_DAY_IDX]) / Cgbdisagg.SEC_IN_HOUR
    tod[tod < 0] = 0
    minutes = tod - tod.astype(int)
    minutes = np.digitize(minutes, [0, 0.25, 0.5, 0.75, 1]) / 4
    tod = (tod.astype(int) + minutes)
    input_data[:, Cgbdisagg.INPUT_SUNSET_IDX] = tod

    t5 = datetime.now()

    logger.info("Preparation of sunset data took | %.3f s", get_time_diff(t4, t5))

    common_ts, input_ts, output_ts = np.intersect1d(input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], output_data[:, 0], return_indices=True)

    output_data = output_data[output_ts, :]
    input_data = input_data[input_ts, :]

    # convert timestamp level to day level data (2D -> 3D format)
    # axis - (appliances, dates, hours)

    input_data, output_data, month_ts = get_hybrid_day_data(input_data, output_data, sampling_rate)

    item_input_object["pilot_level_config_present"] = 1

    # fetching hybrid v2 model file info

    item_input_object, item_output_object, error_code, pilot_level_config, app_killer, run_hybrid_v2_flag, monthly_app_killer = \
        fetch_hybrid_v2_model_file(item_input_object, item_output_object, output_data, appliance_list, logger)

    item_input_object["pilot_level_config"] = pilot_level_config

    t6 = datetime.now()

    logger.info("Conversion of tou to day level data took | %.3f s", get_time_diff(t5, t6))

    t7 = datetime.now()

    logger.info("Preparation of date list took | %.3f s", get_time_diff(t6, t7))

    temperature = copy.deepcopy(input_data[Cgbdisagg.INPUT_TEMPERATURE_IDX, :, :])

    v1_output = np.zeros(len(temperature))
    v2_output = np.zeros(len(temperature))

    # prepare vacation data

    if 'va' in appliance_output_index:
        v1_index = appliance_list.index('va1') + 1
        v2_index = appliance_list.index('va2') + 1

        vacation = np.logical_or(output_data[v1_index, :, :], output_data[v2_index, :, :])

        v1_output = output_data[v1_index, :, 0]
        v2_output = output_data[v2_index, :, 0]

    else:
        vacation = np.zeros(input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :].shape)

    day_input_data = copy.deepcopy(input_data[Cgbdisagg.INPUT_CONSUMPTION_IDX, :, :])

    vacation[np.sum(day_input_data, axis=1) == 0] = 1

    # Prepare day input data
    # Remove high percentile and timed appliances

    dow = input_data[Cgbdisagg.INPUT_DOW_IDX, :, 0]
    weekend_days = np.logical_or(dow == 1, dow == 7)

    original_input_data = copy.deepcopy(day_input_data)

    day_input_data, input_data_without_wh, pp_user, ev_user, ao_cons, timed_wh_user, timed_consumption = \
        prepare_input_day_data(item_input_object, day_input_data, output_data, appliance_list, vacation, logger)

    date_ts_list = input_data[Cgbdisagg.INPUT_EPOCH_IDX, :, 0]

    t8 = datetime.now()

    logger.info("Preparation of input day data took | %.3f s", get_time_diff(t7, t8))

    hsm_params = prepare_hsm_data(item_input_object, day_input_data, samples_per_hour)
    valid_pp_hsm = hsm_params[0]
    pp_hsm = hsm_params[1]
    valid_ev_hsm = hsm_params[2]
    ev_hsm = hsm_params[3]
    valid_wh_hsm = hsm_params[4]
    wh_hsm = hsm_params[5]
    valid_ref_hsm = hsm_params[6]
    ref_hsm = hsm_params[7]
    valid_life_hsm = hsm_params[8]
    created_life_hsm = hsm_params[9]
    life_hsm = hsm_params[10]

    app_profile_pp = item_input_object.get("app_profile").get('pp')

    if app_profile_pp is not None:
        app_profile_pp = app_profile_pp.get("number", 0)
    else:
        app_profile_pp = 0

    if run_hybrid_v2_flag:
        item_input_object = app_id_count(item_input_object, item_input_object.get("app_profile"), logger)

    tankless_wh = fetch_wh_app_prof_info(item_input_object, logger)

    post_hsm_flag = \
        ((item_input_object.get('config').get('disagg_mode') == 'historical' and len(day_input_data) >= 30) or
         (item_input_object.get('config').get('disagg_mode') == 'incremental' and len(day_input_data) >= 70))

    itemization_input_object = {
        "post_hsm_flag": post_hsm_flag,
        "run_hybrid_v2_flag": run_hybrid_v2_flag,
        "vacation_data": vacation.astype(int),
        "temperature_data": temperature,
        "samples_per_hour": samples_per_hour,
        "ts_list": common_ts,
        "date_ts_list": date_ts_list,
        "input_data": input_data,
        "output_data": output_data,
        "day_input_data": day_input_data,
        "month_ts": month_ts,
        "ao_cons": ao_cons,
        "pp_user": pp_user,
        "ev_user": ev_user,
        "timed_wh_user": timed_wh_user,
        "input_data_without_wh": input_data_without_wh,
        "dow": dow,
        "weekend_days": weekend_days,
        "app_list": appliance_list,
        "timed_cons": timed_consumption,
        "original_input_data": original_input_data,
        "change_cool_hld": 0,
        "change_heat_hld": 0,
        "cooling_present": 0,
        "cooling_absent": 0,
        "heating_present": 0,
        "heating_absent": 0,
        'vac_v1': v1_output,
        'vac_v2': v2_output,
        'valid_ev_hsm': valid_ev_hsm,
        'valid_pp_hsm': valid_pp_hsm,
        'valid_wh_hsm': valid_wh_hsm,
        'ev_hsm': ev_hsm,
        'valid_life_hsm': valid_life_hsm,
        'created_life_hsm': created_life_hsm,
        'life_hsm': life_hsm,
        'wh_hsm': wh_hsm,
        'pp_hsm': pp_hsm,
        'backup_ev': 0,
        'backup_pp': 0,
        'valid_ref_hsm': valid_ref_hsm,
        'ref_hsm': ref_hsm,
        'pp_prof_present': app_profile_pp,
        "tankless_wh": tankless_wh,
        "hybrid_thin_pulse": np.zeros_like(day_input_data),
        "wh_added_type": None,
        'backup_app':[],
        'app_killer': app_killer,
        'monthly_app_killer': monthly_app_killer,
        'low_inertia_cool_pilots': [],
        'low_inertia_heat_pilots': [],
    }

    item_input_object.update({
        "item_input_params": itemization_input_object
    })

    item_input_object['item_input_params']['swh_hld'] = 0

    if (item_output_object.get("created_hsm") is not None and
            item_output_object.get("created_hsm").get('wh') is not None and
            item_output_object.get("created_hsm").get('wh').get("attributes") is not None and
            item_output_object.get("created_hsm").get('wh').get("attributes").get('swh_hld') is not None):

        item_input_object['item_input_params']['swh_hld'] = item_output_object.get("created_hsm").get('wh').get("attributes").get('swh_hld')

    logger.info("Samples per hour | %s", samples_per_hour)

    t_prepare_data_end = datetime.now()

    logger.info("Preparation of data took | %.3f s",
                get_time_diff(t_prepare_data_start, t_prepare_data_end))

    item_input_object["item_input_params"]["pp_removed"] = 0

    item_input_object = update_object_with_disagg_debug_params(item_input_object, sampling_rate)

    input_data_dict = {
        "vacation_data": vacation.astype(int),
        "temperature_data": temperature,
        "samples_per_hour": samples_per_hour,
        "ts_list": common_ts,
        "date_ts_list": date_ts_list,
        "input_data": input_data,
        "output_data": output_data,
        "day_input_data": day_input_data,
        "month_ts": month_ts,
        "timed_cons": timed_consumption
    }

    item_output_object["debug"] = dict()

    item_output_object.get('debug').update({
        "input_data_dict": input_data_dict
    })

    item_output_object["output_write_idx_map"] = item_input_object.get("disagg_output_write_idx_map")

    return item_input_object, item_output_object, error_code


def fetch_hybrid_v2_model_file(item_input_object, item_output_object, output_data,  appliance_list, logger):

    """
    Prepare hybrid v2 pilot config inputs

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        output_data               (np.ndarray): disagg output
        appliance_list            (np.ndarray): list of appliances
        logger                    (logger)    : logger object

    Returns:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        error_code                (np.ndarray): exit code of pipeline
        pilot_level_config        (np.ndarray): hybrid v2 config
        app_killer                (np.ndarray): app killer config
        run_hybrid_v2_flag        (np.ndarray): flag that represent whether to run hybrid v2
    """

    enable_flag = copy.deepcopy(item_input_object.get('global_config').get('enable_hybrid_v2'))
    model_file_name = item_input_object.get('global_config').get('hybrid_model_files')
    pilot = item_input_object.get('config').get('pilot_id')
    app_killer = np.zeros_like(appliance_list)
    monthly_app_killer = np.zeros((len(appliance_list), 12))

    error_code = 1

    run_hybrid_v2_flag = enable_flag

    logger.info('Hybrid v2 enable flag value | %s', item_input_object.get('global_config').get('enable_hybrid_v2'))

    pilot_level_config = dict()

    keys_not_missing = (run_hybrid_v2_flag is not None) and \
                       (model_file_name is not None) and \
                       (model_file_name.get('estimationFileS3Path') is not None) and \
                       (model_file_name.get('appKillersFileS3Path') is not None)

    if keys_not_missing and run_hybrid_v2_flag:

        # checking availability of model file

        file1 = model_file_name.get('estimationFileS3Path').split('/')[-1]
        file2 = model_file_name.get('appKillersFileS3Path').split('/')[-1]

        if item_input_object.get("loaded_files").get("item_files") is not None and \
                item_input_object.get("loaded_files").get("item_files").get('pilot_config').get(file1) is not None and \
                (item_input_object.get("loaded_files").get("item_files").get('pilot_config').get(file2) is not None):

            logger.info("Using pilot config for | %s", pilot)

            pilot_level_config = item_input_object.get("loaded_files").get("item_files").get('pilot_config').get(file1).values
            app_prof_config = item_input_object.get("loaded_files").get("item_files").get('pilot_config').get(file2).values

            pilot_level_config, model_file_fetch_is_successful = convert_conf_from_csv_to_json(item_input_object, pilot_level_config, logger)

            if (not model_file_fetch_is_successful):

                # either of the column is missing in model file

                item_input_object["pilot_level_config_present"] = -1

                logger.info('format of hybrid v2 model is not correct | ')

                error_code = -5

                run_hybrid_v2_flag = 0

            else:
                pilot_level_config, app_killer, model_file_fetch_is_successful, monthly_app_killer = \
                    update_bounds(item_input_object, pilot_level_config, app_prof_config, logger)

                if (not model_file_fetch_is_successful):
                    # either of the column is missing in model file

                    item_input_object["pilot_level_config_present"] = -1

                    logger.info('format of hybrid v2 model is not correct | ')

                    error_code = -5

                    run_hybrid_v2_flag = 0

                logger.info('App killer flag array | %s', app_killer)

            return item_input_object, item_output_object, error_code, pilot_level_config, app_killer, run_hybrid_v2_flag, monthly_app_killer

        else:

            run_hybrid_v2_flag = 0

            # complete model file is missing in the location

            item_input_object["pilot_level_config_present"] = 0
            error_code = -4

            logger.info('hybrid v2 model file is missing | ')

            pilot_level_config = dict()

            return item_input_object, item_output_object, error_code, pilot_level_config, app_killer, run_hybrid_v2_flag, monthly_app_killer

    return item_input_object, item_output_object, error_code, pilot_level_config, app_killer, run_hybrid_v2_flag, monthly_app_killer
