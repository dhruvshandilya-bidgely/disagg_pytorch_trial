

"""
Author - Nisha Agarwal
Date - 7th Sep 2020
Prepare HSM data required in itemization pipeline
"""

# Import python packages

import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def check_validity_of_hsm(valid_flag, app_hsm, app_key):

    """
    checking the present of particular key in the appliance hsm

    Parameters:
        valid_flag               (bool)      : true if app hsm can be used
        app_hsm                  (dict)      : Dict containing app hsm
        app_key                  (str)       : app code

    Returns:
        valid_app_hsm            (bool)      : true if app hsm can be used
    """

    valid_app_hsm = False

    if valid_flag and (app_hsm is not None) and (app_hsm.get(app_key) is not None):
        valid_app_hsm = True

    return valid_app_hsm


def get_hsm(item_input_object, app_name):

    """
    Fetch HSM payload

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        app_name                  (str)       : target app

    Returns:
        hsm_in                    (dict)      : hsm input
    """

    try:
        hsm_dic = item_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get(app_name)
    except KeyError:
        hsm_in = None

    return hsm_in


def fetch_pp_hsm(item_input_object, day_input_data, samples_per_hour):

    """
    Fetch hybrid pp attributes

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        day_input_data            (np.ndarray): input data
        samples_per_hour          (int)       : samples in an hour

    Returns:
        valid_pp_hsm             (bool)      : true if pp hsm can be used
        pp_hsm                   (dict)      : dict containing hsm
    """

    hsm_in = get_hsm(item_input_object, 'pp')

    pp_hsm = None
    valid_pp_hsm = True

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (item_input_object.get('config').get('disagg_mode') == 'historical')

    if (hsm_in != {}) and (hsm_fail or (hsm_in is None or len(hsm_in.get('attributes')) == 0 or
                                        hsm_in.get('attributes') is None or hsm_in.get('attributes').get('item_tou') is None or
                                        hsm_in.get('attributes').get('item_amp') is None or
                                        hsm_in.get('attributes').get('item_hld') is None or
                                        hsm_in.get('attributes').get('item_conf') is None or
                                        hsm_in.get('attributes').get('final_item_amp') is None)):
        valid_pp_hsm = False

    elif hsm_in == {} and ((item_input_object.get('config').get('disagg_mode') not in ['incremental', 'historical']) or
                           (item_input_object.get('config').get('disagg_mode') == 'incremental') and (len(day_input_data) < 70)):
        valid_pp_hsm = True

    elif hsm_in != {}:
        valid_pp_hsm, pp_hsm = check_pp_hsm_validity(hsm_in, samples_per_hour)

    return valid_pp_hsm, pp_hsm


def check_pp_hsm_validity(hsm_in, samples_per_hour):

    """
    Fetch validity of pp hsm

    Parameters:
        hsm_in                    (dict)      : hsm input
        samples_per_hour          (int)       : samples in an hour

    Returns:
        valid_pp_hsm             (bool)      : true if pp hsm can be used
        pp_hsm                   (dict)      : dict containing hsm
    """

    pp_hsm = hsm_in.get('attributes')

    if isinstance(pp_hsm, list):
        pp_hsm = pp_hsm[0]

    final_amplitude = pp_hsm.get('final_item_amp')
    conf = pp_hsm.get('item_conf')
    hld = pp_hsm.get('item_hld')
    pp_tou = pp_hsm.get('item_tou')
    pp_extension_tou = pp_hsm.get('item_extend_tou')

    if hld is None or conf is None or final_amplitude is None:
        valid_pp_hsm = False
    else:
        valid_pp_hsm = True

        if int(len(pp_tou) / Cgbdisagg.HRS_IN_DAY) != samples_per_hour:
            pp_hsm['item_tou'] = resample_1d_data(np.array(pp_tou), samples_per_hour * Cgbdisagg.HRS_IN_DAY)
            pp_hsm['item_extend_tou'] = resample_1d_data(np.array(pp_extension_tou), samples_per_hour * Cgbdisagg.HRS_IN_DAY)

    return valid_pp_hsm, pp_hsm


def resample_1d_data(data, total_samples):

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

    elif samples_in_an_hour > 1:

        samples_in_an_hour = int(samples_in_an_hour)

        aggregated_data = np.zeros(data.shape)

        for sample in range(samples_in_an_hour):

            aggregated_data = aggregated_data + np.roll(data, sample)

        resampled_data = aggregated_data[np.arange(samples_in_an_hour-1, len(data[0]), samples_in_an_hour)]

    # Upsample data

    else:

        resampled_data = np.zeros(total_samples)

        for sample in range(int(1/samples_in_an_hour)):

            resampled_data[np.arange(sample, total_samples, int(1/samples_in_an_hour))] = data

        resampled_data = resampled_data * samples_in_an_hour

    resampled_data[resampled_data > 0] = 1

    return resampled_data


def fetch_wh_hsm(item_input_object, day_input_data, samples_per_hour):

    """
    Fetch hybrid wh attributes

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        day_input_data            (np.ndarray): input data
        samples_per_hour          (int)       : samples in an hour

    Returns:
        valid_wh_hsm             (bool)      : true if wh hsm can be used
        wh_hsm                   (dict)      : dict containing hsm
    """

    wh_hsm = None
    valid_wh_hsm = True

    hsm_in = get_hsm(item_input_object, 'wh')

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (item_input_object.get('config').get('disagg_mode') == 'historical')

    if (hsm_in != {}) and (hsm_fail or (hsm_in is None or len(hsm_in.get('attributes')) == 0 or
                                        hsm_in.get('attributes') is None or hsm_in.get('attributes').get('item_tou') is None or
                                        hsm_in.get('attributes').get('item_amp') is None or
                                        hsm_in.get('attributes').get('item_hld') is None or
                                        hsm_in.get('attributes').get('item_type') is None)):
        valid_wh_hsm = False

    elif hsm_in == {} and ((item_input_object.get('config').get('disagg_mode') not in ['incremental', 'historical']) or
                           (item_input_object.get('config').get('disagg_mode') == 'incremental') and (len(day_input_data) < 70)):
        valid_wh_hsm = True
    elif hsm_in != {}:
        valid_wh_hsm, wh_hsm = check_wh_hsm_validity(hsm_in, samples_per_hour)

    return valid_wh_hsm, wh_hsm


def check_wh_hsm_validity(hsm_in, samples_per_hour):

    """
    check validity of wh hsm

    Parameters:
        hsm_in                    (dict)      : hsm input
        samples_per_hour          (int)       : samples in an hour

    Returns:
        valid_wh_hsm              (bool)      : true if wh hsm can be used
        wh_hsm                    (dict)      : dict containing hsm

    """

    wh_hsm = hsm_in.get('attributes')

    if isinstance(wh_hsm, list):
        wh_hsm = wh_hsm[0]

    wh_tou = wh_hsm.get('item_tou')
    hld = wh_hsm.get('item_hld')

    valid_wh_hsm = True

    if hld is None or wh_tou is None:
        valid_wh_hsm = False
    elif int(len(wh_tou) / Cgbdisagg.HRS_IN_DAY) != samples_per_hour:
        wh_hsm['item_tou'] = resample_1d_data(np.array(wh_tou), samples_per_hour * Cgbdisagg.HRS_IN_DAY)

    return valid_wh_hsm, wh_hsm


def fetch_ev_hsm(item_input_object, day_input_data):

    """
    Fetch hybrid ev attributes

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        day_input_data            (np.ndarray): input data

    Returns:
        valid_ev_hsm             (bool)      : true if ev hsm can be used
        ev_hsm                   (dict)      : dict containing hsm
    """

    hsm_in = get_hsm(item_input_object, 'ev')

    valid_ev_hsm = False
    ev_hsm = None

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (item_input_object.get('config').get('disagg_mode') == 'historical')

    if (hsm_in != {}) and (hsm_fail or (hsm_in is None or len(hsm_in.get('attributes')) == 0 or
                                        hsm_in.get('attributes') is None or hsm_in.get('attributes').get('item_type') is None or
                                        hsm_in.get('attributes').get('item_amp') is None)):
        valid_ev_hsm = False
    elif hsm_in == {} and ((item_input_object.get('config').get('disagg_mode') not in ['incremental', 'historical']) or
                           (item_input_object.get('config').get('disagg_mode') == 'incremental') and (len(day_input_data) < 70)):
        valid_ev_hsm = True
    elif hsm_in != {}:
        ev_hsm = hsm_in.get('attributes')

        if isinstance(ev_hsm, list):
            type = ev_hsm[0].get('item_type')
            amplitude = ev_hsm[0].get('item_amp')
        else:
            type = ev_hsm.get('item_type')
            amplitude = ev_hsm.get('item_amp')

        valid_ev_hsm = True

        if type is None or amplitude is None:
            valid_ev_hsm = False

    return valid_ev_hsm, ev_hsm


def fetch_ref_hsm(item_input_object, day_input_data):

    """
    Fetch hybrid ref attributes

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        day_input_data            (np.ndarray): input data

    Returns:
        valid_ref_hsm             (bool)      : true if ref hsm can be used
        ref_hsm                   (dict)      : dict containing hsm
    """

    hsm_in = get_hsm(item_input_object, 'ref')

    ref_hsm = None
    valid_ref_hsm = True

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (item_input_object.get('config').get('disagg_mode') == 'historical')

    if ((hsm_in != {}) and (hsm_fail or (hsm_in is None or len(hsm_in.get('attributes')) == 0 or
                                         hsm_in.get('attributes') is None or hsm_in.get('attributes').get('hybrid_ref') is None))) or \
            (hsm_in == {} and (hsm_in.get('attributes') is None)):
        valid_ref_hsm = False

    elif hsm_in == {} and ((item_input_object.get('config').get('disagg_mode') not in ['incremental', 'historical']) or
                           (item_input_object.get('config').get('disagg_mode') == 'incremental') and (len(day_input_data) < 70)):
        valid_ref_hsm = True

    elif hsm_in != {}:
        ref_hsm, valid_ref_hsm = check_ref_hsm_validity(hsm_in, valid_ref_hsm)

    return valid_ref_hsm, ref_hsm


def check_ref_hsm_validity(hsm_in, valid_ref_hsm):

    """
    check validity of ref hsm

    Parameters:
        hsm_in                    (dict)      : hsm input
        valid_ref_hsm             (bool)      : true if ref hsm can be used

    Returns:
        ref_hsm                   (dict)      : dict containing hsm
        valid_ref_hsm             (bool)      : true if ref hsm can be used
    """

    ref_hsm = hsm_in.get('attributes')

    if (ref_hsm is None) or \
            ((ref_hsm is not None) and (isinstance(ref_hsm, list) and ref_hsm[0].get('hybrid_ref') is None)) or \
            ((ref_hsm is not None) and ref_hsm.get('hybrid_ref') is None):
        valid_ref_hsm = False

    return ref_hsm, valid_ref_hsm
