
"""
Author - Nisha Agarwal
Date - 7th Sep 2023
process hybrid v2 pilot config file
"""

# Import python packages

import numpy as np
import pandas as pd

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.config.mappings.get_app_id import get_app_name


def update_bounds(item_input_object, pilot_config, app_prof_conf, logger):

    """
    process hybrid v2 pilot config file

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        pilot_config              (dict)      : hybrid v2 pilot config
        app_prof_conf             (dict)      : hybrid v2 app killer config
        logger                    (logger)    : logger object

    Returns:
        pilot_config              (dict)      : hybrid v2 pilot config
        app_killer                (list)      : array containing information of appliance to be killed in v2 module
    """

    appliance_list = np.array(["ao", "ev", "cooling", "heating", 'li', "pp", "ref", "va1", "va2", "wh", "ld", "ent", "cook"])

    app_prof_list = list(item_input_object.get('app_profile').keys())
    count_arr = np.ones(len(app_prof_list))
    app_id_arr = np.zeros(len(app_prof_list))
    type_arr = [['']] * len(app_prof_list)
    size_arr = [['']] * len(app_prof_list)
    attr_arr = [['']] * len(app_prof_list)

    app_killer = np.zeros(len(appliance_list))

    monthly_app_killer = np.zeros((len(appliance_list), 12))

    column_list = app_prof_conf[0]
    app_prof_conf = app_prof_conf[1:]

    model_file_fetch_successful, reason = check_validity_for_app_killer_model_file(column_list, logger)

    if not model_file_fetch_successful:
        return pilot_config, app_killer, model_file_fetch_successful, monthly_app_killer

    # prepare survey app profile info for all applianes

    for count, profile in enumerate(app_prof_list):

        app_profile = item_input_object.get("app_profile").get(profile)

        if (app_profile is not None):
            type_arr[count] = app_profile.get("type", "")
            count_arr[count] = app_profile.get("number", 1)
            size_arr[count] = app_profile.get("size", "")
            attr_arr[count] = app_profile.get("attributes", "")
            app_id_arr[count] = app_profile.get("appID", 0)

    if not (item_input_object.get('app_profile') == {}):
        app_id_arr = app_id_arr.astype(int)

        type_arr = np.array(type_arr)
        attr_arr = np.array(attr_arr)
        size_arr = np.array(size_arr)

        type_arr[type_arr == None] = ""
        attr_arr[attr_arr == None] = ""
        size_arr[size_arr == None] = ""

    mapping = dict()
    mapping['9'] = 6
    mapping['18'] = 1
    mapping['4'] = 2
    mapping['3'] = 3
    mapping['71'] = 4
    mapping['2'] = 5
    mapping['7'] = 9
    mapping['59'] = 10
    mapping['66'] = 11
    mapping['5'] = 12

    user_app_prof_params = [app_id_arr, type_arr, count_arr, size_arr, attr_arr]

    # initialializing app killer array

    meta_data = item_input_object.get('home_meta_data')

    for i in range(len(app_prof_conf)):
        pilot_config, app_killer = \
            app_killer_conf_for_each_appliance(pilot_config, app_prof_conf, app_killer, i, mapping, user_app_prof_params)

        values = app_prof_conf[i]

        # checking whether app profile tag is available

        if (values[1] == "META_DATA") and (not pd.isna(values[8])) and (not pd.isna(values[8])):

            meta_data_attr = str(values[7])
            meta_data_val = str(values[8]).replace('.0', '')

            if (meta_data.get(meta_data_attr) is not None) and str(meta_data.get(meta_data_attr)) == meta_data_val:
                app_row_idx = mapping[str(int(values[0]))]
                app_killer[app_row_idx] = 1

        if (values[1] == "MONTH"):
            app_row_idx = mapping[str(int(values[0]))]
            monthly_app_killer[app_row_idx, int(float(str(values[9])))] = 1

    return pilot_config, app_killer, True, monthly_app_killer


def app_killer_conf_for_each_appliance(pilot_config, app_prof_conf, app_killer, idx, mapping, user_app_prof_params):

    """
    process hybrid v2 pilot config file

    Parameters:
        pilot_config              (dict)      : hybrid v2 pilot config
        app_prof_conf             (dict)      : hybrid v2 app killer config
        app_killer                (list)      : array containing app killer info
        idx                       (int)       : target app index
        mapping                   (dict)      : app mapping
        user_app_prof_params      (list)      : app prof input of the user

    Returns:
        pilot_config              (dict)      : hybrid v2 pilot config
        app_killer                (list)      : array containing information of appliance to be killed in v2 module
    """

    app_id_arr = user_app_prof_params[0]
    type_arr = user_app_prof_params[1]
    count_arr = user_app_prof_params[2]
    size_arr = user_app_prof_params[3]
    attr_arr = user_app_prof_params[4]

    values = app_prof_conf[idx]

    # checking whether app profile tag is available

    if (values[1] != "APP_PROF") or pd.isna(values[2]):
        return pilot_config, app_killer

    app_id_val = np.array(values[2].split('|')).astype(int)

    if len(np.intersect1d(app_id_val, app_id_arr)) != len(app_id_val):
        return pilot_config, app_killer

    valid_idx = np.isin(app_id_arr, app_id_val)

    type_arr_val = type_arr[valid_idx]
    count_arr_val = count_arr[valid_idx]
    size_arr_val = size_arr[valid_idx]
    attr_arr_val = attr_arr[valid_idx]

    # fetching attribute/count/type/size for each row in the input config

    attr = str(values[4]).split('|')
    count = str(values[3]).split('|')
    type = str(values[5]).split('|')
    size = str(values[6]).split('|')

    # filling the value with empty string in case of any missing values

    if pd.isna(values[4]):
        attr = [''] * len(app_id_val)
        attr = np.array(attr, dtype=object)

    if pd.isna(values[3]):
        count = [''] * len(app_id_val)
        count = np.array(count, dtype=object)
    else:
        count = np.array(count).astype(float).astype(int).astype(str)

    if pd.isna(values[5]):
        type = [''] * len(app_id_val)
        type = np.array(type, dtype=object)

    if pd.isna(values[6]):
        size = [''] * len(app_id_val)
        size = np.array(size, dtype=object)
    else:
        size = np.array(size).astype(float).astype(int).astype(str)

    faulty_app_killer_conf = (len(app_id_val) != len(count)) or (len(app_id_val) != len(type)) or \
                             (len(app_id_val) != len(attr)) or (len(app_id_val) != len(size))

    count = np.array(count, dtype=object)
    type = np.array(type, dtype=object)
    attr = np.array(attr, dtype=object)
    size = np.array(size, dtype=object)

    # handling cases where these attributes wont be used(value given as -1)

    attr[attr == '-1'] = attr_arr_val[attr == '-1']
    count[count == '-1'] = count_arr_val[count == '-1']
    type[type == '-1'] = type_arr_val[type == '-1']
    size[size == '-1'] = size_arr_val[size == '-1']

    attr[attr == ''] = attr_arr_val[attr == '']
    count[count == ''] = count_arr_val[count == '']
    type[type == ''] = type_arr_val[type == '']
    size[size == ''] = size_arr_val[size == '']

    attr[attr == 'nan'] = attr_arr_val[attr == 'nan']
    count[count == 'nan'] = count_arr_val[count == 'nan']
    type[type == 'nan'] = type_arr_val[type == 'nan']
    size[size == 'nan'] = size_arr_val[size == 'nan']

    count = count.astype(int)
    count_arr_val = count_arr_val.astype(int)

    size_arr_val = size_arr_val.astype(str)
    size = size.astype(str)

    size = np.char.lower(size)
    size_arr_val = np.char.lower(size_arr_val)

    attr = np.char.lower(attr.astype(str))
    attr_arr_val = np.char.lower(attr_arr_val.astype(str))

    type = np.char.lower(type.astype(str))
    type_arr_val = np.char.lower(type_arr_val.astype(str))

    # if all the conditions of app killer config matches with survey input, the consumption of given appliance is killed

    if (np.all(attr == attr_arr_val) and np.all(count == count_arr_val) and np.all(type == type_arr_val) and np.all(size == size_arr_val)) and \
            (not faulty_app_killer_conf):
        target_idx = mapping[str(int(values[0]))]
        app_killer[target_idx] = 1

    return pilot_config, app_killer


def convert_conf_from_csv_to_json(item_input_object, csv_config, logger):

    """
    This file converts the original estimation config into json format, to be used further in the hybrid v2 pipeline

    Parameters:
        item_input_object             (dict)      : Dict containing all inputs
        csv_config                    (np.ndarray): original hybrid v2 pilot config

    Returns:
        json_config                   (dict)      : processed hybrid v2 pilot config
        model_file_fetch_successful   (bool)      : flag denoting whether model file format is correct
    """

    json_config = fetch_default_json_config(item_input_object)

    column_list = csv_config[0]
    csv_config = csv_config[1:]

    # checking whether all required columns are present in the csv

    model_file_fetch_successful, reason = check_validity_for_model_file(column_list, csv_config[:, 0], logger)

    if not model_file_fetch_successful:
        return json_config, model_file_fetch_successful

    app_list = csv_config[:, np.where(column_list == 'APP_ID')[0][0]]

    app_name_col = csv_config[:, np.where(column_list == 'APP_NAME')[0][0]]

    avg_val_col = csv_config[:, np.where(column_list == 'AVG_KWH_PM')[0][0]]

    min_val_col = csv_config[:, np.where(column_list == 'MIN_KWH_PM')[0][0]]

    max_val_col = csv_config[:, np.where(column_list == 'MAX_KWH_PM')[0][0]]

    take_this_from_disagg_col = csv_config[:, np.where(column_list == 'TAKE_THIS_FROM_DISAGG')[0][0]]

    block_if_less_than_col = csv_config[:, np.where(column_list == 'BLOCK_IF_LESS_THAN')[0][0]]

    coverage_col = csv_config[:, np.where(column_list == 'COVERAGE')[0][0]]

    if 'DEFAULT_FUEL_TYPE ' in column_list:
        fuel_type_col = csv_config[:, np.where(column_list == 'DEFAULT_FUEL_TYPE ')[0][0]]
    else:
        fuel_type_col = csv_config[:, np.where(column_list == 'DEFAULT_FUEL_TYPE')[0][0]]

    # updating config values for each appliance

    for i, target_app in enumerate(app_list):

        app_code = get_app_name(target_app)

        if app_code in ['ao', 'others']:
            continue

        if app_code == 'ac':
            app_code = 'cooling'
        if app_code == 'sh':
            app_code = 'heating'

        if (app_code == 'cook') and ('DW' in app_name_col[i]):
            json_config['dish_washer_cat'] = 'cook'

        json_config[app_code + "_config"]['bounds']['min_cons'] = float(min_val_col[i])
        json_config[app_code + "_config"]['bounds']['mid_cons'] = float(avg_val_col[i])
        json_config[app_code + "_config"]['bounds']['max_cons'] = float(max_val_col[i])
        json_config[app_code + "_config"]['bounds']['take_from_disagg'] = float(take_this_from_disagg_col[i])
        json_config[app_code + "_config"]['bounds']['block_if_less_than'] = float(block_if_less_than_col[i])
        json_config[app_code + "_config"]['coverage'] = float(coverage_col[i])

        if np.isnan(float(coverage_col[i])):
            json_config[app_code + "_config"]['coverage'] = -1

        json_config[app_code + "_config"]['type'] = str(fuel_type_col[i])

    return json_config, model_file_fetch_successful


def check_validity_for_model_file(column_list, app_list, logger):

    """
    This file converts the original estimation config into json format, to be used further in the hybrid v2 pipeline

    Parameters:
        item_input_object             (dict)      : Dict containing all inputs

    Returns:
        json_config                   (dict)      : processed hybrid v2 pilot config
        model_file_fetch_successful   (bool)      : flag denoting whether model file format is correct
    """

    model_file_fetch_successful = 1

    reason = ""

    if '9' not in app_list:
        model_file_fetch_successful = 0
        reason = "REF row is missing"

    if '5' not in app_list:
        model_file_fetch_successful = 0
        reason = "cooking row is missing"

    if '59' not in app_list:
        model_file_fetch_successful = 0
        reason = "Laundry row is missing"

    if '66' not in app_list:
        model_file_fetch_successful = 0
        reason = "entertainment row is missing"

    if '2' not in app_list:
        model_file_fetch_successful = 0
        reason = "PP row is missing"

    if '7' not in app_list:
        model_file_fetch_successful = 0
        reason = "WH row is missing"

    if '18' not in app_list:
        model_file_fetch_successful = 0
        reason = "EV row is missing"

    if '3' not in app_list:
        model_file_fetch_successful = 0
        reason = "heating row is missing"

    if '4' not in app_list:
        model_file_fetch_successful = 0
        reason = "cooling row is missing"

    if 'APP_ID' not in column_list:
        model_file_fetch_successful = 0
        reason = "APP_ID column is missing"

    if 'AVG_KWH_PM' not in column_list:
        model_file_fetch_successful = 0
        reason = "AVG_KWH_PM column is missing"

    if 'MIN_KWH_PM' not in column_list:
        model_file_fetch_successful = 0
        reason = "MIN_KWH_PM column is missing"

    if 'MAX_KWH_PM' not in column_list:
        model_file_fetch_successful = 0
        reason = "MAX_KWH_PM column is missing"

    if 'TAKE_THIS_FROM_DISAGG' not in column_list:
        model_file_fetch_successful = 0
        reason = "TAKE_THIS_FROM_DISAGG column is missing"

    if 'BLOCK_IF_LESS_THAN' not in column_list:
        model_file_fetch_successful = 0
        reason = "BLOCK_IF_LESS_THAN column is missing"

    if 'COVERAGE' not in column_list:
        model_file_fetch_successful = 0
        reason = "COVERAGE column is missing"

    if 'APP_NAME' not in column_list:
        model_file_fetch_successful = 0
        reason = "APP_NAME column is missing"

    logger.info('Not processing hybrid v2 model file due to | %s', reason)

    return model_file_fetch_successful, reason


def check_validity_for_app_killer_model_file(column_list, logger):

    """
    This file converts the original estimation config into json format, to be used further in the hybrid v2 pipeline

    Parameters:
        item_input_object             (dict)      : Dict containing all inputs

    Returns:
        json_config                   (dict)      : processed hybrid v2 pilot config
        model_file_fetch_successful   (bool)      : flag denoting whether model file format is correct
    """

    model_file_fetch_successful = 1

    reason = ""

    if 'APPLIANCE_CATEGORY' not in column_list:
        model_file_fetch_successful = 0
        reason = "APPLIANCE_CATEGORY column is missing"

    if 'INPUT_PARAM' not in column_list:
        model_file_fetch_successful = 0
        reason = "INPUT_PARAM column is missing"

    if 'APP_ID' not in column_list and 'APP_ID ' not in column_list:
        model_file_fetch_successful = 0
        reason = "APP_ID column is missing"

    if 'APP_COUNT' not in column_list:
        model_file_fetch_successful = 0
        reason = "APP_COUNT column is missing"

    if 'APP_ATTRIBUTE' not in column_list:
        model_file_fetch_successful = 0
        reason = "APP_ATTRIBUTE column is missing"

    if 'APP_FUEL_TYPE' not in column_list:
        model_file_fetch_successful = 0
        reason = "APP_FUEL_TYPE column is missing"

    if 'APP_SIZE' not in column_list:
        model_file_fetch_successful = 0
        reason = "APP_SIZE column is missing"

    if 'META_DATA_ATTRIBUTE' not in column_list:
        model_file_fetch_successful = 0
        reason = "META_DATA_ATTRIBUTE column is missing"

    if 'META_DATA_VAL' not in column_list:
        model_file_fetch_successful = 0
        reason = "META_DATA_VAL column is missing"

    if 'MONTH_VAL' not in column_list:
        model_file_fetch_successful = 0
        reason = "MONTH_VAL column is missing"

    if not model_file_fetch_successful:
        logger.info('Not processing hybrid v2 model file due to | %s', reason)

    return model_file_fetch_successful, reason


def fetch_default_json_config(item_input_object):

    """
    This file converts the original estimation config into json format, to be used further in the hybrid v2 pipeline

    Parameters:
        item_input_object             (dict)      : Dict containing all inputs

    Returns:
        json_config                   (dict)      : processed hybrid v2 pilot config
        model_file_fetch_successful   (bool)      : flag denoting whether model file format is correct
    """

    # intializing config

    json_config = \
        {

            "dish_washer_cat": 'ld',
            "geography": 'na',

            "ref_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "li_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "cooling_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "heating_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "wh_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "pp_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "ev_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "cook_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "ent_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": -1,
                "drop_app": []
            },
            "ld_config": {
                "bounds": {
                    "min_cons": 0.0,
                    "mid_cons": 0.0,
                    "max_cons": 0.0,
                    "take_from_disagg": 2.0,
                    "block_if_less_than": -1.0
                },
                "coverage": 90,
                "drop_app": []
            }
        }

    pilot = item_input_object.get("config").get("pilot_id")

    if pilot in PilotConstants.EU_PILOTS:
        json_config['geography'] = 'eu'

    return json_config
