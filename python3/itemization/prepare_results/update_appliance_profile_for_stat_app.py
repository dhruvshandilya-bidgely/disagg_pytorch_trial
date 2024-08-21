
"""
Author - Nisha Agarwal
Date - 7th Sep 2022
Master file for updating appliance profile
"""

# Import python packages

import copy
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def update_cook_appliance_profile(item_input_object, item_output_object, logger):
    """
    update appliance profile for cases where we add cook appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    cook_idx = np.where(app_list == 'cook')[0][0] + 1

    out_bill_cycles = item_input_object.get('out_bill_cycles')

    bc_start_col = 0
    bc_end_col = 1

    for bill_cycle_idx in range(out_bill_cycles.shape[0]):

        # Extract the bill cycle to populate the profile for

        bill_cycle_start = out_bill_cycles[bill_cycle_idx, bc_start_col]
        bill_cycle_end = out_bill_cycles[bill_cycle_idx, bc_end_col]

        logger.debug('Cooking appliance profile population started for | %d', bill_cycle_start)

        # Initialize the dictionary containing the user profile attributes

        user_profile_object = dict(
            {
                "validity": None,
                "isPresent": None,
                "detectionConfidence": None,
                "count": None,
                "attributes": {
                    "cookingConsumption": None,
                    "timeOfUsage": None,
                },
                "debugAttributes": {}
            }
        )

        user_profile_object['isPresent'] = bool(np.sum(final_output[cook_idx]) > 0)
        user_profile_object['count'] = int(np.sum(final_output[cook_idx]) > 0)
        user_profile_object['detectionConfidence'] = float(np.sum(final_output[cook_idx]) > 0)

        user_profile_object['validity'] = dict()
        user_profile_object['validity']['start'] = int(bill_cycle_start)
        user_profile_object['validity']['end'] = int(bill_cycle_end)

        # Populate consumption for the bill cycle

        app_output = final_output[cook_idx]
        app_cons = app_output[bill_cycle_data == bill_cycle_start].sum()

        app_tou = prepare_bc_tou_for_profile(app_output[bill_cycle_data == bill_cycle_start].sum(axis=0))

        user_profile_object['attributes']['cookingConsumption'] = float(np.round(app_cons, 2))

        user_profile_object['attributes']["timeOfUsage"] = list(app_tou)

        user_profile_list = [copy.deepcopy(user_profile_object)]

        # Populate appliance profile for the given bill cycle

        item_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str('5')] = user_profile_list

    return item_output_object


def update_ent_appliance_profile(item_input_object, item_output_object, logger):
    """
    update appliance profile for cases where we add ent appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    ent_idx = np.where(app_list == 'ent')[0][0] + 1

    out_bill_cycles = item_input_object.get('out_bill_cycles')

    bc_start_col = 0
    bc_end_col = 1

    for bill_cycle_idx in range(out_bill_cycles.shape[0]):
        # Extract the bill cycle to populate the profile for

        bill_cycle_start = out_bill_cycles[bill_cycle_idx, bc_start_col]
        bill_cycle_end = out_bill_cycles[bill_cycle_idx, bc_end_col]

        logger.debug('Entertainment appliance profile population started for | %d', bill_cycle_start)

        # Initialize the dictionary containing the user profile attributes

        user_profile_object = dict(
            {
                "validity": None,
                "isPresent": None,
                "detectionConfidence": None,
                "count": None,
                "attributes": {
                    "entertainmentConsumption": None,
                    "timeOfUsage": None,
                },
                "debugAttributes": {}
            }
        )

        user_profile_object['isPresent'] = bool(np.sum(final_output[ent_idx]) > 0)
        user_profile_object['count'] = int(np.sum(final_output[ent_idx]) > 0)
        user_profile_object['detectionConfidence'] = float(np.sum(final_output[ent_idx]) > 0)

        user_profile_object['validity'] = dict()
        user_profile_object['validity']['start'] = int(bill_cycle_start)
        user_profile_object['validity']['end'] = int(bill_cycle_end)

        # Populate consumption for the bill cycle

        app_output = final_output[ent_idx]
        app_cons = app_output[bill_cycle_data == bill_cycle_start].sum()

        app_tou = prepare_bc_tou_for_profile(app_output[bill_cycle_data == bill_cycle_start].sum(axis=0))

        user_profile_object['attributes']['entertainmentConsumption'] = float(np.round(app_cons, 2))

        user_profile_object['attributes']["timeOfUsage"] = list(app_tou)

        user_profile_list = [copy.deepcopy(user_profile_object)]

        # Populate appliance profile for the given bill cycle

        item_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str('66')] = user_profile_list

    return item_output_object


def update_ld_appliance_profile(item_input_object, item_output_object, logger):
    """
    update appliance profile for cases where we add ld appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    ld_idx = np.where(app_list == 'ld')[0][0] + 1

    out_bill_cycles = item_input_object.get('out_bill_cycles')

    bc_start_col = 0
    bc_end_col = 1

    for bill_cycle_idx in range(out_bill_cycles.shape[0]):
        # Extract the bill cycle to populate the profile for

        bill_cycle_start = out_bill_cycles[bill_cycle_idx, bc_start_col]
        bill_cycle_end = out_bill_cycles[bill_cycle_idx, bc_end_col]

        logger.debug('Laundry appliance profile population started for | %d', bill_cycle_start)

        # Initialize the dictionary containing the user profile attributes

        user_profile_object = dict(
            {
                "validity": None,
                "isPresent": None,
                "detectionConfidence": None,
                "count": None,
                "attributes": {
                    "laundryConsumption": None,
                    "timeOfUsage": None,
                },
                "debugAttributes": {}
            }
        )

        user_profile_object['isPresent'] = bool(np.sum(final_output[ld_idx]) > 0)
        user_profile_object['count'] = int(np.sum(final_output[ld_idx]) > 0)
        user_profile_object['detectionConfidence'] = float(np.sum(final_output[ld_idx]) > 0)

        user_profile_object['validity'] = dict()
        user_profile_object['validity']['start'] = int(bill_cycle_start)
        user_profile_object['validity']['end'] = int(bill_cycle_end)

        # Populate consumption for the bill cycle

        app_output = final_output[ld_idx]
        app_cons = app_output[bill_cycle_data == bill_cycle_start].sum()

        app_tou = prepare_bc_tou_for_profile(app_output[bill_cycle_data == bill_cycle_start].sum(axis=0))

        user_profile_object['attributes']['laundryConsumption'] = float(np.round(app_cons, 2))

        user_profile_object['attributes']["timeOfUsage"] = list(app_tou)

        user_profile_list = [copy.deepcopy(user_profile_object)]

        # Populate appliance profile for the given bill cycle

        item_output_object['appliance_profile'][bill_cycle_start]['profileList'][0][str('59')] = user_profile_list

    return item_output_object


def prepare_bc_tou_for_profile(disagg_output):

    """
    For each bill cycle in out_bill_cycles create an hourly array to be filled in TOU for appliance profile

    Parameters:
        disagg_output         (np.ndarray)      : disagg output

    Returns:
        app_tou               (list)            : disagg load curve

    """

    disagg_output = resample_day_data(disagg_output, 24)

    if np.sum(disagg_output) == 0:
        app_tou = disagg_output.astype(float)
        app_tou = list(app_tou)
        return app_tou

    app_tou = disagg_output / np.sum(disagg_output)

    app_tou = app_tou.astype(float)
    app_tou = np.round(app_tou, 3)
    app_tou = list(app_tou)

    return app_tou


def resample_day_data(data, total_samples):

    """
    This function resamples data, to the number of samples required,, eg 15min to 30 min user data conversion

    Parameters:
        data                       (np.ndarray)        : target array
        total_samples              (int)               : number of target samples in a day

    Returns:
        resampled_data             (np.ndarray)        : resampled array
    """

    total_samples = int(total_samples)

    samples_in_an_hour = len(data) / (total_samples)

    # no sampling required

    if samples_in_an_hour == 1:
        return data

    # Downsample data

    if samples_in_an_hour > 1:

        samples_in_an_hour = int(samples_in_an_hour)

        aggregated_data = np.zeros(data.shape)

        for sample in range(samples_in_an_hour):

            aggregated_data = aggregated_data + np.roll(data, sample)

        resampled_data = aggregated_data[np.arange(samples_in_an_hour-1, len(data), samples_in_an_hour)]

    else:
        resampled_data = data

    return resampled_data


def update_li_appliance_profile(item_input_object, item_output_object, logger):
    """
    update appliance profile for cases where we add wh appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    appliance_profile = item_output_object.get('appliance_profile')
    bc_list = list(appliance_profile.keys())

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    li_idx = np.where(app_list == 'li')[0][0] + 1

    if np.sum(final_output[li_idx]) == 0:
        item_output_object['appliance_profile'] = appliance_profile
        logger.info("Not updating Li appliance profile in disagg postprocessing | ")
        return item_output_object

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['71'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['71']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['71'][0]

            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle)):
                li_output = final_output[li_idx]
                li_cons = li_output[bill_cycle_data == bill_cycle].sum()
                bill_cycle_prof['attributes']['lightingConsumption'] = float(li_cons)

                li_tou = prepare_bc_tou_for_profile(li_output[bill_cycle_data == bill_cycle].sum(axis=0))
                bill_cycle_prof['attributes']['timeOfUsage'] = li_tou

            appliance_profile[bill_cycle].get('profileList')[0]['71'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

    return item_output_object
