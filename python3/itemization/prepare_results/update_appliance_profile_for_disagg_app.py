
"""
Author - Nisha Agarwal
Date - 7th Sep 2022
Master file for updating appliance profile
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.itemization_utils import find_seq


def update_removed_pp_appliance_profile(item_input_object, item_output_object, logger):

    """
    update appliance profile for cases where we remove appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    if (item_input_object.get("item_input_params") is not None) and \
            (item_input_object.get("item_input_params").get("pp_removed") is not None) and \
            item_input_object.get("item_input_params").get("pp_removed") == 1:

        logger.info('updating pp appliance profile since pp is removed in disagg postprocessing | ')

        appliance_profile = item_output_object.get('appliance_profile')
        bc_list = list(appliance_profile.keys())

        for bill_cycle in bc_list:

            if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['2'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['2']) > 0):

                bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['2'][0]

                bill_cycle_prof['isPresent'] = False
                bill_cycle_prof['detectionConfidence'] = 0.0
                bill_cycle_prof['count'] = 0

                if bill_cycle_prof.get('attributes') is not None:
                    bill_cycle_prof['attributes']['ppConsumption'] = 0.0
                    bill_cycle_prof['attributes']['appType'] = None
                    bill_cycle_prof['attributes']['fuelType'] = None
                    bill_cycle_prof['attributes']['numberOfRuns'] = None
                    bill_cycle_prof['attributes']['amplitude'] = None
                    bill_cycle_prof['attributes']['schedule'] = None
                    bill_cycle_prof['attributes']['timeOfUsage'] = None

                appliance_profile[bill_cycle].get('profileList')[0]['2'][0] = bill_cycle_prof

        item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_deleted_wh_appliance_profile(item_input_object, item_output_object, logger):

    """
    update appliance profile for cases where we remove wh appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    if (item_input_object.get("item_input_params") is not None) and \
            (item_input_object.get("item_input_params").get("wh_removed") is not None) and \
            item_input_object.get("item_input_params").get("wh_removed") == 1:
        appliance_profile = item_output_object.get('appliance_profile')
        bc_list = list(appliance_profile.keys())

        logger.info('updating wh appliance profile since wh is removed in disagg postprocessing | ')

        for bill_cycle in bc_list:

            if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['7'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['7']) > 0):

                bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['7'][0]

                bill_cycle_prof['isPresent'] = False
                bill_cycle_prof['detectionConfidence'] = 0.0
                bill_cycle_prof['count'] = 0

                if bill_cycle_prof.get('attributes') is not None:
                    bill_cycle_prof['attributes']['whConsumption'] = 0.0
                    bill_cycle_prof['attributes']['appType'] = None
                    bill_cycle_prof['attributes']['fuelType'] = None
                    bill_cycle_prof['attributes']['runsCount'] = 0
                    bill_cycle_prof['attributes']['amplitude'] =  0.0
                    bill_cycle_prof['attributes']['amplitudeConfidence'] = 0.0
                    bill_cycle_prof['attributes']['dailyThinPulseCount'] = 0
                    bill_cycle_prof['attributes']['passiveUsageFraction'] = 0.0
                    bill_cycle_prof['attributes']['activeUsageFraction'] = 0.0
                    bill_cycle_prof['attributes']['timeOfUsage'] = None

                appliance_profile[bill_cycle].get('profileList')[0]['7'][0] = bill_cycle_prof

        item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_added_wh_appliance_profile(item_input_object, item_output_object, logger):
    """
    update appliance profile for cases where we add wh appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """
    if (item_input_object.get("item_input_params") is not None) and \
            (item_input_object.get("item_input_params").get("wh_added") is not None) and \
            item_input_object.get("item_input_params").get("wh_added") == 1:
        appliance_profile = item_output_object.get('appliance_profile')
        bc_list = list(appliance_profile.keys())

        logger.info('updating wh appliance profile since wh is added in disagg postprocessing | ')

        final_output = item_output_object.get("final_itemization").get("tou_itemization")
        app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
        bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX], axis=1)
        wh_idx = np.where(app_list == 'wh')[0][0] + 1

        wh_bc_output = final_output[wh_idx]
        wh_amp = 0

        samples = int(final_output[wh_idx].shape[1] / 24)

        wh_type = item_input_object.get('item_input_params').get("wh_added_type")

        if np.sum(wh_bc_output):
            wh_amp = np.percentile(wh_bc_output[wh_bc_output > 0], 95) * samples

        for bill_cycle in bc_list:

            if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['7'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['7']) > 0):

                bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['7'][0]

                bill_cycle_prof['isPresent'] = True
                bill_cycle_prof['detectionConfidence'] = 0.8
                bill_cycle_prof['count'] = 1

                if bill_cycle_prof.get('attributes') is not None:
                    bill_cycle_prof['attributes']['whConsumption'] = 0.0
                    bill_cycle_prof['attributes']['appType'] = None
                    bill_cycle_prof['attributes']['fuelType'] = "Electric"
                    bill_cycle_prof['attributes']['runsCount'] = 0
                    bill_cycle_prof['attributes']['amplitude'] =  0.0
                    bill_cycle_prof['attributes']['amplitudeConfidence'] = 0.0
                    bill_cycle_prof['attributes']['dailyThinPulseCount'] = 0
                    bill_cycle_prof['attributes']['passiveUsageFraction'] = 0.0
                    bill_cycle_prof['attributes']['activeUsageFraction'] = 0.0
                    bill_cycle_prof['attributes']['timeOfUsage'] = None

                    update_wh_prof_attributes(item_input_object, bill_cycle_prof, final_output[wh_idx], bill_cycle_data, bill_cycle, wh_amp, wh_type)

                appliance_profile[bill_cycle].get('profileList')[0]['7'][0] = bill_cycle_prof

        item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_added_pp_appliance_profile(item_input_object, item_output_object, logger):

    """
    update appliance profile for cases where we add pp appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """
    pp_added_from_hybrid = \
        (item_input_object.get("item_input_params") is not None) and \
        (item_input_object.get("item_input_params").get("pp_added") is not None) and \
        item_input_object.get("item_input_params").get("pp_added") == 1

    if not pp_added_from_hybrid:
        return item_output_object

    appliance_profile = item_output_object.get('appliance_profile')
    bc_list = list(appliance_profile.keys())

    logger.info('updating pp appliance profile since pp is added in disagg postprocessing | ')

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    pp_idx = np.where(app_list == 'pp')[0][0] + 1

    pp_bc_output = final_output[pp_idx]
    pp_amp = 0

    samples = int(final_output[pp_idx].shape[1] / 24)

    if np.sum(pp_bc_output):
        pp_amp = np.round(np.percentile(pp_bc_output[pp_bc_output > 0], 95)  * samples, 2)

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
            (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
            (appliance_profile[bill_cycle].get('profileList')[0]['2'] is not None) and \
            (len(appliance_profile[bill_cycle].get('profileList')[0]['2']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['2'][0]

            bill_cycle_prof['isPresent'] = True
            bill_cycle_prof['detectionConfidence'] = 0.85
            bill_cycle_prof['count'] = 1

            if item_input_object.get("item_input_params").get("backup_pp") is not None and \
                    item_input_object.get("item_input_params").get("backup_pp") > 0:
                bill_cycle_prof['detectionConfidence'] = 0.55

            if bill_cycle_prof.get('attributes') is not None:

                bill_cycle_prof['attributes']['ppConsumption'] = 0.0
                bill_cycle_prof['attributes']['appType'] = 'Single'
                bill_cycle_prof['attributes']['fuelType'] = 'Electric'
                bill_cycle_prof['attributes']['numberOfRuns'] = [1.0]
                bill_cycle_prof['attributes']['amplitude'] = None
                bill_cycle_prof['attributes']['schedule'] = None
                bill_cycle_prof['attributes']['timeOfUsage'] = None

                update_pp_prof_attributes(bill_cycle_prof, final_output[pp_idx], bill_cycle_data, bill_cycle, pp_amp)

            appliance_profile[bill_cycle].get('profileList')[0]['2'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

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


def get_ev_attributes(ev_output):

    """
    prepare ev attributes

    Parameters:
        ev_output           (np.ndarray)      : disagg output

    Returns:
        ev_amp              (int)             : ev  amp
        type                (str)             : ev  type
        duration            (int)             : ev  length

    """

    ev_amp = 0

    samples = int(ev_output.shape[1] / 24)

    type = None

    duration = 0

    if np.sum(ev_output):
        ev_amp = np.percentile(ev_output[ev_output > 0], 95) * samples

        if ev_amp > 3000 / samples:
            type = 'L2'
        else:
            type = 'L1'

        ev_1d = (ev_output > 0).flatten()

        ev_usage_seq = find_seq(ev_1d, np.zeros_like(ev_1d), np.zeros_like(ev_1d))

        duration = np.median(ev_usage_seq[ev_usage_seq[:, 0] > 0, 3])

    return ev_amp, type, duration


def update_pp_prof_attributes(bill_cycle_prof, pp_output, bill_cycle_data, bill_cycle, pp_amp):

    """
    update profile with pp attributes

    Parameters:
        bill_cycle_prof     (dict)             : appliance profile
        pp_output           (np.ndarray)       : pp output
        bill_cycle_data     (np.darray)        : list of bc of all days
        bill_cycle          (int)              : current bill cycle
        pp_amp              (int)              : pp amp

    Returns:
        bill_cycle_prof     (dict)             : updated profile

    """

    if np.any(bill_cycle_data == bill_cycle):
        pp_cons = pp_output[bill_cycle_data == bill_cycle].sum()
        pp_tou = prepare_bc_tou_for_profile(pp_output[bill_cycle_data == bill_cycle].sum(axis=0))

        bill_cycle_prof['attributes']['ppConsumption'] = float(pp_cons)
        bill_cycle_prof['attributes']['timeOfUsage'] = pp_tou
        bill_cycle_prof['attributes']['amplitude'] = [float(pp_amp)]

    else:
        bill_cycle_prof['attributes']['amplitude'] = [float(pp_amp)]

    return bill_cycle_prof


def update_added_ev_appliance_profile(item_input_object, item_output_object, logger):

    """
    update appliance profile for cases where we add pp appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        logger                    (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    ev_added = (item_input_object.get("item_input_params") is not None) and \
               (item_input_object.get("item_input_params").get("ev_added") is not None) and \
               item_input_object.get("item_input_params").get("ev_added") == 1

    if not ev_added:
        return item_output_object

    appliance_profile = item_output_object.get('appliance_profile')
    bc_list = list(appliance_profile.keys())

    logger.info('updating ev appliance profile since ev is added in disagg postprocessing | ')

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    ev_idx = np.where(app_list == 'ev')[0][0] + 1

    ev_bc_output = final_output[ev_idx]

    ev_amp, type, duration = get_ev_attributes(ev_bc_output)

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
            (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
            (appliance_profile[bill_cycle].get('profileList')[0]['18'] is not None) and \
            (len(appliance_profile[bill_cycle].get('profileList')[0]['18']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['18'][0]

            bill_cycle_prof['isPresent'] = True

            score = 0.85

            bill_cycle_prof['detectionConfidence'] = score
            bill_cycle_prof['count'] = 1

            if bill_cycle_prof.get('attributes') is not None:
                bill_cycle_prof['attributes']['evConsumption'] = None
                bill_cycle_prof['attributes']['chargerType'] = None
                bill_cycle_prof['attributes']['amplitude'] =  0.0
                bill_cycle_prof['attributes']['chargingInstanceCount'] = 0
                bill_cycle_prof['attributes']['averageChargingDuration'] = 0.0
                bill_cycle_prof['attributes']['timeOfUsage'] = None

                update_ev_prof_attributes(bill_cycle_prof, final_output[ev_idx], bill_cycle_data,
                                          bill_cycle, ev_amp, duration, type)

            appliance_profile[bill_cycle].get('profileList')[0]['18'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_ev_prof_attributes(bill_cycle_prof, ev_output, bill_cycle_data, bill_cycle, ev_amp, duration, ev_type):

    """
    update profile with ev attributes

    Parameters:
        bill_cycle_prof     (dict)             : appliance profile
        ev_output           (np.ndarray)       : wh output
        bill_cycle_data     (np.darray)        : list of bc of all days
        bill_cycle          (int)              : current bill cycle
        ev_amp              (int)              : ev amp
        duration            (int)              : ev length
        ev_type             (str)              : ev type

    Returns:
        bill_cycle_prof     (dict)             : updated profile

    """

    if np.any(bill_cycle_data == bill_cycle):
        ev_cons = ev_output[bill_cycle_data == bill_cycle].sum()
        ev_tou = prepare_bc_tou_for_profile(ev_output[bill_cycle_data == bill_cycle].sum(axis=0))

        ev_1d = (ev_output[bill_cycle_data == bill_cycle] > 0).flatten()

        ev_usage_seq = find_seq(ev_1d, np.zeros_like(ev_1d), np.zeros_like(ev_1d))

        count = np.sum(ev_usage_seq[:, 0] > 0)

        bill_cycle_prof['attributes']['evConsumption'] = float(ev_cons)
        bill_cycle_prof['attributes']['timeOfUsage'] = ev_tou
        bill_cycle_prof['attributes']['amplitude'] = float(ev_amp)
        bill_cycle_prof['attributes']['chargerType'] = ev_type
        bill_cycle_prof['attributes']['chargingInstanceCount'] = int(count)
        bill_cycle_prof['attributes']['averageChargingDuration'] = float(duration)

    else:
        bill_cycle_prof['attributes']['amplitude'] = float(ev_amp)
        bill_cycle_prof['attributes']['chargerType'] = ev_type
        bill_cycle_prof['attributes']['averageChargingDuration'] = float(duration)

    return bill_cycle_prof


def update_ev_appliance_profile(item_input_object, item_output_object, logger):
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
    disagg_output = item_output_object.get("inference_engine_dict").get("output_data")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    ev_idx = np.where(app_list == 'ev')[0][0] + 1

    ev_added = 0

    if (item_input_object.get("item_input_params") is not None) and \
            (item_input_object.get("item_input_params").get("ev_added") is not None):
        ev_added = item_input_object.get("item_input_params").get("ev_added")

    ev_removed = (np.sum(final_output[ev_idx]) == 0) and (np.sum(disagg_output[ev_idx-1]) > 0)

    if ev_added or (np.sum(final_output[ev_idx] + disagg_output[ev_idx-1]) == 0):
        item_output_object['appliance_profile'] = appliance_profile
        logger.info("Not updating ev appliance profile in disagg postprocessing | ")
        return item_output_object

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['18'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['18']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['18'][0]

            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle)):
                ev_output = final_output[ev_idx]
                ev_cons = ev_output[bill_cycle_data == bill_cycle].sum()
                bill_cycle_prof['attributes']['evConsumption'] = float(ev_cons)

                ev_tou = prepare_bc_tou_for_profile(ev_output[bill_cycle_data == bill_cycle].sum(axis=0))
                bill_cycle_prof['attributes']['timeOfUsage'] = ev_tou

                ev_1d = (ev_output[bill_cycle_data == bill_cycle] > 0).flatten()
                ev_usage_seq = find_seq(ev_1d, np.zeros_like(ev_1d), np.zeros_like(ev_1d))
                count = np.sum(ev_usage_seq[:, 0] > 0)

                bill_cycle_prof['attributes']['chargingInstanceCount'] = int(count)

                if ev_removed:
                    bill_cycle_prof['isPresent'] = False
                    bill_cycle_prof['detectionConfidence'] = 0.0
                    bill_cycle_prof['count'] = 0
                    bill_cycle_prof['attributes']['evConsumption'] = None
                    bill_cycle_prof['attributes']['chargerType'] = None
                    bill_cycle_prof['attributes']['amplitude'] = 0.0
                    bill_cycle_prof['attributes']['chargingInstanceCount'] = 0
                    bill_cycle_prof['attributes']['averageChargingDuration'] = 0.0
                    bill_cycle_prof['attributes']['timeOfUsage'] = None

            appliance_profile[bill_cycle].get('profileList')[0]['18'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_wh_prof_attributes(item_input_object, bill_cycle_prof, wh_output, bill_cycle_data, bill_cycle, wh_amp, wh_type):

    """
    update profile with wh attributes

    Parameters:
        item_input_object   (dict)             : pipeline input dict
        bill_cycle_prof     (dict)             : appliance profile
        wh_output           (np.ndarray)       : wh output
        bill_cycle_data     (np.darray)        : list of bc of all days
        bill_cycle          (int)              : current bill cycle
        wh_amp              (int)              : wh amp
        wh_type             (str)              : wh type

    Returns:
        bill_cycle_prof     (dict)             : updated profile

    """

    if np.any(bill_cycle_data == bill_cycle):
        wh_cons = wh_output[bill_cycle_data == bill_cycle].sum()
        wh_tou = prepare_bc_tou_for_profile(wh_output[bill_cycle_data == bill_cycle].sum(axis=0))

        bill_cycle_prof['attributes']['whConsumption'] = float(wh_cons)
        bill_cycle_prof['attributes']['timeOfUsage'] = wh_tou
        bill_cycle_prof['attributes']['amplitude'] = float(wh_amp)
        bill_cycle_prof['attributes']['appType'] = wh_type

        if (wh_type is not None) and (wh_type == 'thermostat') and item_input_object.get("item_input_params").get("hybrid_thin_pulse") is not None:
            passive_usage_frac = item_input_object["item_input_params"]["hybrid_thin_pulse"].sum() / wh_output.sum()
            passive_usage_frac = float(min(1, np.round(passive_usage_frac, 3)))

            bill_cycle_prof['attributes']["passiveUsageFraction"] = passive_usage_frac
            bill_cycle_prof['attributes']["activeUsageFraction"] = 1 - passive_usage_frac

    else:
        bill_cycle_prof['attributes']['amplitude'] = float(wh_amp)
        bill_cycle_prof['attributes']['appType'] = wh_type

    return bill_cycle_prof


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


def update_wh_appliance_profile(item_input_object, item_output_object, logger):
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
    wh_idx = np.where(app_list == 'wh')[0][0] + 1

    wh_added = 0
    wh_removed = 0

    if (item_input_object.get("item_input_params") is not None) and (
            item_input_object.get("item_input_params").get("wh_added") is not None):
        wh_added = item_input_object.get("item_input_params").get("wh_added")

    if (item_input_object.get("item_input_params") is not None) and (
            item_input_object.get("item_input_params").get("wh_removed") is not None):
        wh_removed = item_input_object.get("item_input_params").get("wh_removed")

    if np.sum(final_output[wh_idx]) == 0 or wh_added or wh_removed:
        item_output_object['appliance_profile'] = appliance_profile
        logger.info("Not updating wh appliance profile in disagg postprocessing | ")
        return item_output_object

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['7'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['7']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['7'][0]

            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle)):
                wh_output = final_output[wh_idx]
                wh_cons = wh_output[bill_cycle_data == bill_cycle].sum()
                bill_cycle_prof['attributes']['whConsumption'] = float(wh_cons)
                bill_cycle_prof['isPresent'] = True

                bill_cycle_prof = update_wh_prof_attributes_without_hld_change(item_input_object, bill_cycle_prof, wh_output, bill_cycle_data, bill_cycle)

            appliance_profile[bill_cycle].get('profileList')[0]['7'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_wh_prof_attributes_without_hld_change(item_input_object, bill_cycle_prof, wh_output, bill_cycle_data, bill_cycle):

    """
    update profile with wh attributes

    Parameters:
        item_input_object   (dict)             : pipeline input dict
        bill_cycle_prof     (dict)             : appliance profile
        wh_output           (np.ndarray)       : wh output
        bill_cycle_data     (np.darray)        : list of bc of all days
        bill_cycle          (int)              : current bill cycle
        wh_amp              (int)              : wh amp
        wh_type             (str)              : wh type

    Returns:
        bill_cycle_prof     (dict)             : updated profile

    """

    if np.any(bill_cycle_data == bill_cycle):
        wh_tou = prepare_bc_tou_for_profile(wh_output[bill_cycle_data == bill_cycle].sum(axis=0))

        bill_cycle_prof['attributes']['timeOfUsage'] = wh_tou

        if ((bill_cycle_prof['attributes']["passiveUsageFraction"] is None) or not (bill_cycle_prof['attributes']["passiveUsageFraction"] >= 0))\
                and item_input_object.get("item_input_params").get("hybrid_thin_pulse") is not None:
            passive_usage_frac = item_input_object["item_input_params"]["hybrid_thin_pulse"].sum() / wh_output.sum()
            passive_usage_frac = float(min(1, np.round(passive_usage_frac, 3)))

            bill_cycle_prof['attributes']["passiveUsageFraction"] = passive_usage_frac
            bill_cycle_prof['attributes']["activeUsageFraction"] = 1 - passive_usage_frac

    return bill_cycle_prof


def update_pp_appliance_profile(item_input_object, item_output_object, logger):
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
    pp_idx = np.where(app_list == 'pp')[0][0] + 1

    pp_added = 0
    pp_removed = 0

    if (item_input_object.get("item_input_params") is not None) and (
            item_input_object.get("item_input_params").get("pp_added") is not None):
        pp_added = item_input_object.get("item_input_params").get("pp_added")

    if (item_input_object.get("item_input_params") is not None) and (
            item_input_object.get("item_input_params").get("pp_removed") is not None):
        pp_removed = item_input_object.get("item_input_params").get("pp_removed")

    if np.sum(final_output[pp_idx]) == 0 or pp_added or pp_removed:
        item_output_object['appliance_profile'] = appliance_profile
        logger.info("Not updating pp appliance profile in disagg postprocessing | ")
        return item_output_object

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['2'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['2']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['2'][0]

            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle)):
                pp_output = final_output[pp_idx]
                pp_cons = pp_output[bill_cycle_data == bill_cycle].sum()
                bill_cycle_prof['attributes']['ppConsumption'] = float(pp_cons)

                pp_tou = prepare_bc_tou_for_profile(pp_output[bill_cycle_data == bill_cycle].sum(axis=0))
                bill_cycle_prof['attributes']['timeOfUsage'] = pp_tou

            appliance_profile[bill_cycle].get('profileList')[0]['2'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_cooling_appliance_profile(item_input_object, item_output_object):
    """
    update appliance profile for cases where we add wh appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """
    appliance_profile = item_output_object.get('appliance_profile')
    bc_list = list(appliance_profile.keys())

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    disagg_output = item_output_object.get("inference_engine_dict").get("output_data")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    cool_idx = np.where(app_list == 'cooling')[0][0] + 1
    hvac_removed = 0

    if (np.sum(final_output[cool_idx]) == 0) and (np.sum(disagg_output[cool_idx-1]) > 0):
        hvac_removed = 1

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['4'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['4']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['4'][0]

            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle)):
                cool_output = final_output[cool_idx]
                cool_cons = cool_output[bill_cycle_data == bill_cycle].sum()
                bill_cycle_prof['attributes']['coolingConsumption'] = float(cool_cons)

                cool_tou = prepare_bc_tou_for_profile(cool_output[bill_cycle_data == bill_cycle].sum(axis=0))
                bill_cycle_prof['attributes']['timeOfUsage'] = cool_tou

                if hvac_removed:
                    bill_cycle_prof['isPresent'] = False

            appliance_profile[bill_cycle].get('profileList')[0]['4'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_heating_appliance_profile(item_input_object, item_output_object):
    """
    update appliance profile for cases where we add wh appliance in hybrid module

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """
    appliance_profile = item_output_object.get('appliance_profile')
    bc_list = list(appliance_profile.keys())

    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    disagg_output = item_output_object.get("inference_engine_dict").get("output_data")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    heat_idx = np.where(app_list == 'heating')[0][0] + 1

    hvac_removed = 0

    if (np.sum(final_output[heat_idx]) == 0) and (np.sum(disagg_output[heat_idx-1]) > 0):
        hvac_removed = 1

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['3'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['3']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['3'][0]

            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle)):
                heat_output = final_output[heat_idx]
                heat_cons = heat_output[bill_cycle_data == bill_cycle].sum()
                bill_cycle_prof['attributes']['heatingConsumption'] = float(heat_cons)

                heat_tou = prepare_bc_tou_for_profile(heat_output[bill_cycle_data == bill_cycle].sum(axis=0))
                bill_cycle_prof['attributes']['timeOfUsage'] = heat_tou

                if hvac_removed:
                    bill_cycle_prof['isPresent'] = False

            appliance_profile[bill_cycle].get('profileList')[0]['3'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile

    return item_output_object


def update_ao_appliance_profile(item_input_object, item_output_object, logger):
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
    disagg_output = item_output_object.get("inference_engine_dict").get("output_data")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    ao_idx = np.where(app_list == 'ao')[0][0] + 1

    if np.sum(final_output[ao_idx] + disagg_output[ao_idx-1]) == 0:
        item_output_object['appliance_profile'] = appliance_profile
        logger.info("Not updating ao appliance profile in disagg postprocessing | ")
        return item_output_object

    for bill_cycle in bc_list:

        if (appliance_profile[bill_cycle].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle].get('profileList')[0]['8'] is not None) and \
                (len(appliance_profile[bill_cycle].get('profileList')[0]['8']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle].get('profileList')[0]['8'][0]

            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle)):
                ao_output = final_output[ao_idx]
                ao_cons = ao_output[bill_cycle_data == bill_cycle].sum()
                bill_cycle_prof['attributes']['aoConsumption'] = float(ao_cons)

            appliance_profile[bill_cycle].get('profileList')[0]['8'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile
    return item_output_object


def update_ref_appliance_profile(item_input_object, item_output_object, logger):
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
    final_output = item_output_object.get("final_itemization").get("tou_itemization")
    app_list = np.array(item_output_object.get('inference_engine_dict').get('appliance_list'))
    bill_cycle_data = np.max(item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX],  axis=1)
    ref_idx = np.where(app_list == 'ref')[0][0] + 1
    out_bill_cycles = item_input_object.get('out_bill_cycles')
    bc_start_col = 0
    bc_end_col = 1
    if np.sum(final_output[ref_idx]) == 0:
        item_output_object['appliance_profile'] = appliance_profile
        logger.info("Not updating REF appliance profile in disagg postprocessing | ")
        return item_output_object

    for bill_cycle_idx in range(out_bill_cycles.shape[0]):
        # Extract the bill cycle to populate the profile for ref
        bill_cycle_start = out_bill_cycles[bill_cycle_idx, bc_start_col]
        bill_cycle_end = out_bill_cycles[bill_cycle_idx, bc_end_col]

        if (appliance_profile[bill_cycle_start].get('profileList') is not None) and \
                (len(appliance_profile[bill_cycle_start].get('profileList')) > 0) and \
                (appliance_profile[bill_cycle_start].get('profileList')[0]['9'] is not None) and \
                (len(appliance_profile[bill_cycle_start].get('profileList')[0]['9']) > 0):

            bill_cycle_prof = appliance_profile[bill_cycle_start].get('profileList')[0]['9'][0]
            if (bill_cycle_prof.get('attributes') is not None) and (np.any(bill_cycle_data == bill_cycle_start)):
                ref_output = final_output[ref_idx]
                ref_cons = ref_output[bill_cycle_data == bill_cycle_start].sum()
                bill_cycle_prof['attributes']['refConsumption'] = float(ref_cons)

            bill_cycle_prof['isPresent'] = True
            bill_cycle_prof['detectionConfidence'] = 1.0
            bill_cycle_prof['count'] = 1
            bill_cycle_prof['validity'] = dict()
            bill_cycle_prof['validity']['start'] = int(bill_cycle_start)
            bill_cycle_prof['validity']['end'] = int(bill_cycle_end)
            appliance_profile[bill_cycle_start].get('profileList')[0]['9'][0] = bill_cycle_prof

    item_output_object['appliance_profile'] = appliance_profile
    return item_output_object
