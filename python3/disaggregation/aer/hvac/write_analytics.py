"""
Author - Abhinav Srivastava
Date - 22/10/18
Call the hvac disaggregation module and get results
"""

# Import python packages

import copy
import os
import pandas as pd
import numpy as np

# Import functions from within the project

from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def analytics_cooling_2nd_mode(analyse):

    """Populate analytics for cooling 2nd mode"""

    cooling_2nd_mode_info = []

    # noinspection PyBroadException
    try:
        cooling_second_mean = analyse['cooling']['detection']['means'][1]
        cooling_second_std = analyse['cooling']['detection']['std'][1]

    except (IndexError, KeyError):

        cooling_second_mean = 0
        cooling_second_std = 0

    cooling_second_mode_reg_info = analyse['cooling']['estimation'][1]['cluster_info']
    cooling_second_mode_dbscan_cluster_validity = [cooling_second_mode_reg_info[key]['validity'] for key in
                                                   cooling_second_mode_reg_info.keys()]
    cooling_second_mode_validity = any(cooling_second_mode_dbscan_cluster_validity)

    if cooling_second_mode_validity:

        cooling_second_mode_coefficient = \
            [cooling_second_mode_reg_info[key]['coefficient'] for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key]['validity']][0][0][0]

        cooling_second_mode_intercept = \
            [cooling_second_mode_reg_info[key]['intercept'] for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key]['validity']][0][0]

        cooling_second_mode_reg_kind = \
            [cooling_second_mode_reg_info[key]['regression_kind'] for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key]['validity']][0]

        cooling_second_mode_reg_r2 = \
            [cooling_second_mode_reg_info[key]['r_square'] for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key]['validity']][0]
    else:
        cooling_second_mode_coefficient = 0
        cooling_second_mode_intercept = 0
        cooling_second_mode_reg_kind = 0
        cooling_second_mode_reg_r2 = 0

    cooling_2nd_mode_info.append(cooling_second_mean)
    cooling_2nd_mode_info.append(cooling_second_std)
    cooling_2nd_mode_info.append(cooling_second_mode_coefficient)
    cooling_2nd_mode_info.append(cooling_second_mode_intercept)
    cooling_2nd_mode_info.append(cooling_second_mode_reg_kind)
    cooling_2nd_mode_info.append(cooling_second_mode_reg_r2)

    return cooling_2nd_mode_info


def analytics_heating_2nd_mode(analyse):

    """Populate analytics for heating 2nd mode"""

    heating_2nd_mode_info = []

    # noinspection PyBroadException
    try:
        heating_second_mean = analyse['heating']['detection']['means'][1]
        heating_second_std = analyse['heating']['detection']['std'][1]
    except (IndexError, KeyError):
        heating_second_mean = 0
        heating_second_std = 0

    heating_second_mode_reg_info = analyse['heating']['estimation'][1]['cluster_info']
    heating_second_mode_dbscan_cluster_validity = [heating_second_mode_reg_info[key]['validity'] for key in
                                                   heating_second_mode_reg_info.keys()]
    heating_second_mode_validity = any(heating_second_mode_dbscan_cluster_validity)

    if heating_second_mode_validity:

        heating_second_mode_coefficient = \
            [heating_second_mode_reg_info[key]['coefficient'] for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key]['validity']][0][0][0]

        heating_second_mode_intercept = \
            [heating_second_mode_reg_info[key]['intercept'] for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key]['validity']][0][0]

        heating_second_mode_reg_kind = \
            [heating_second_mode_reg_info[key]['regression_kind'] for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key]['validity']][0]

        heating_second_mode_reg_r2 = \
            [heating_second_mode_reg_info[key]['r_square'] for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key]['validity']][0]
    else:
        heating_second_mode_coefficient = 0
        heating_second_mode_intercept = 0
        heating_second_mode_reg_kind = 0
        heating_second_mode_reg_r2 = 0

    heating_2nd_mode_info.append(heating_second_mean)
    heating_2nd_mode_info.append(heating_second_std)
    heating_2nd_mode_info.append(heating_second_mode_coefficient)
    heating_2nd_mode_info.append(heating_second_mode_intercept)
    heating_2nd_mode_info.append(heating_second_mode_reg_kind)
    heating_2nd_mode_info.append(heating_second_mode_reg_r2)

    return heating_2nd_mode_info


def convert_analytics_hsm_to_float(hvac_analytics_hsm):
    '''
    Function to ensure hsm values are acceptable

    Parameters:

        hvac_analytics_hsm      (list)      : List containing analytics related parameters

    Returns:
        hvac_analytics_hsm      (list)      : List containing analytics related sanitized parameters
    '''

    static_params = hvac_static_params()

    for idx in range(len(hvac_analytics_hsm)):

        if hvac_analytics_hsm[idx] == np.inf:
            hvac_analytics_hsm[idx] = static_params.get('hsm_large_num')

        hvac_analytics_hsm[idx] = float(hvac_analytics_hsm[idx])

    return hvac_analytics_hsm


def get_ac_first_mode_validity(analyse):

    """
    Function to write hvac analytics csv

    Parameters:

        analyse (dict)                     : Dictionary containing all hvac related important analytics parameters

    Returns:
        cooling_first_mode_reg_info (dict) : Dictionary of regression info
        cooling_first_mode_validity (bool) : Boolean of first mode validity
    """

    if 0 in list(analyse['cooling']['estimation'].keys()):
        cooling_first_mode_reg_info = analyse['cooling']['estimation'][0]['cluster_info']
        cooling_first_mode_dbscan_cluster_validity = [cooling_first_mode_reg_info[key]['validity'] for key in cooling_first_mode_reg_info.keys()]
        cooling_first_mode_validity = any(cooling_first_mode_dbscan_cluster_validity)
    else:
        cooling_first_mode_reg_info = {}
        cooling_first_mode_validity = False

    return cooling_first_mode_reg_info, cooling_first_mode_validity


def get_sh_first_mode_validity(analyse):

    """
    Function to write hvac analytics csv

    Parameters:

        analyse (dict)                     : Dictionary containing all hvac related important analytics parameters

    Returns:
        heating_first_mode_reg_info (dict) : Dictionary of regression info
        heating_first_mode_validity (bool) : Boolean of first mode validity
    """

    if 0 in list(analyse['heating']['estimation'].keys()):
        heating_first_mode_reg_info = analyse['heating']['estimation'][0]['cluster_info']
        heating_first_mode_dbscan_cluster_validity = [heating_first_mode_reg_info[key]['validity'] for key in heating_first_mode_reg_info.keys()]
        heating_first_mode_validity = any(heating_first_mode_dbscan_cluster_validity)
    else:
        heating_first_mode_reg_info = {}
        heating_first_mode_validity = False

    return heating_first_mode_reg_info, heating_first_mode_validity


def write_hvac_analytics(disagg_input_object, analyse, disagg_output_object):

    """
    Function to write hvac analytics csv

    Parameters:

        disagg_input_object (dict)  : Dictionary containing all inputs
        analyse (dict)              : Dictionary containing all hvac related important analytics parameters
        disagg_output_object (dict) : Dictionary containing all user outputs

    Returns:
        None
    """

    global_config = disagg_input_object['config']
    config = copy.deepcopy(global_config)
    meta_data = disagg_input_object['home_meta_data']

    # Cooling related Analytics
    cooling_modes_count = analyse['cooling']['detection']['number_of_modes']
    cooling_first_mean = analyse['cooling']['detection']['means'][0]
    cooling_first_std = analyse['cooling']['detection']['std'][0]

    cooling_first_mode_reg_info, cooling_first_mode_validity = get_ac_first_mode_validity(analyse)

    if cooling_first_mode_validity:

        cooling_first_mode_coefficient = \
        [cooling_first_mode_reg_info[key]['coefficient'] for key in cooling_first_mode_reg_info.keys() if
         cooling_first_mode_reg_info[key]['validity']][0][0][0]
        cooling_first_mode_intercept = \
        [cooling_first_mode_reg_info[key]['intercept'] for key in cooling_first_mode_reg_info.keys() if
         cooling_first_mode_reg_info[key]['validity']][0][0]
        cooling_first_mode_reg_kind = \
            [cooling_first_mode_reg_info[key]['regression_kind'] for key in cooling_first_mode_reg_info.keys() if
             cooling_first_mode_reg_info[key]['validity']][0]
        cooling_first_mode_reg_r2 = \
            [cooling_first_mode_reg_info[key]['r_square'] for key in cooling_first_mode_reg_info.keys() if
             cooling_first_mode_reg_info[key]['validity']][0]
    else:
        cooling_first_mode_coefficient = 0
        cooling_first_mode_intercept = 0
        cooling_first_mode_reg_kind = 0
        cooling_first_mode_reg_r2 = 0

    cooling_second_mean = 0
    cooling_second_std = 0
    cooling_second_mode_validity = False

    # noinspection PyBroadException
    try:

        cooling_2nd_mode_info = analytics_cooling_2nd_mode(analyse)

        cooling_second_mean = cooling_2nd_mode_info[0]
        cooling_second_std = cooling_2nd_mode_info[1]
        cooling_second_mode_coefficient = cooling_2nd_mode_info[2]
        cooling_second_mode_intercept = cooling_2nd_mode_info[3]
        cooling_second_mode_reg_kind = cooling_2nd_mode_info[4]
        cooling_second_mode_reg_r2 = cooling_2nd_mode_info[5]

    except (IndexError, KeyError):

        cooling_second_mode_validity = 0
        cooling_second_mode_coefficient = 0
        cooling_second_mode_intercept = 0
        cooling_second_mode_reg_kind = 0
        cooling_second_mode_reg_r2 = 0

    # Heating related Analytics
    heating_modes_count = analyse['heating']['detection']['number_of_modes']
    heating_first_mean = analyse['heating']['detection']['means'][0]
    heating_first_std = analyse['heating']['detection']['std'][0]

    heating_first_mode_reg_info, heating_first_mode_validity = get_sh_first_mode_validity(analyse)

    if heating_first_mode_validity:

        heating_first_mode_coefficient = \
            [heating_first_mode_reg_info[key]['coefficient'] for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key]['validity']][0][0][0]
        heating_first_mode_intercept = \
            [heating_first_mode_reg_info[key]['intercept'] for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key]['validity']][0][0]
        heating_first_mode_reg_kind = \
            [heating_first_mode_reg_info[key]['regression_kind'] for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key]['validity']][0]
        heating_first_mode_reg_r2 = \
            [heating_first_mode_reg_info[key]['r_square'] for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key]['validity']][0]
    else:
        heating_first_mode_coefficient = 0
        heating_first_mode_intercept = 0
        heating_first_mode_reg_kind = 0
        heating_first_mode_reg_r2 = 0

    heating_second_mean = 0
    heating_second_std = 0
    heating_second_mode_validity = False

    # noinspection PyBroadException
    try:
        heating_2nd_mode_info = analytics_heating_2nd_mode(analyse)

        heating_second_mean = heating_2nd_mode_info[0]
        heating_second_std = heating_2nd_mode_info[1]
        heating_second_mode_coefficient = heating_2nd_mode_info[2]
        heating_second_mode_intercept = heating_2nd_mode_info[3]
        heating_second_mode_reg_kind = heating_2nd_mode_info[4]
        heating_second_mode_reg_r2 = heating_2nd_mode_info[5]

    except (IndexError, KeyError):

        heating_second_mode_validity = 0
        heating_second_mode_coefficient = 0
        heating_second_mode_intercept = 0
        heating_second_mode_reg_kind = 0
        heating_second_mode_reg_r2 = 0

    meta_keys = ['ownershipType', 'numOccupants', 'state', 'city', 'country', 'zipCode', 'timezone', 'yearBuilt', 'defaultNId',
                 'livingArea', 'lotSize', 'bedrooms', 'totalRooms', 'spaceHeatingType', 'solarUser', 'latitude', 'longitude']

    meta_data = initialise_meta_data_keys(meta_data, meta_keys)

    hvac_analytics = [config['pilot_id'], config['sampling_rate'], meta_data['city'], meta_data['zipCode'],
                      meta_data['defaultNId'], meta_data['state'], meta_data['timezone'], meta_data['country'],
                      meta_data['yearBuilt'], meta_data['ownershipType'], meta_data['numOccupants'],
                      meta_data['livingArea'], meta_data['lotSize'], meta_data['totalRooms'], meta_data['bedrooms'],
                      meta_data['spaceHeatingType'], meta_data['solarUser'], meta_data['latitude'],
                      meta_data['longitude'],
                      cooling_modes_count,
                      np.around(cooling_first_mean), np.around(cooling_first_std), cooling_first_mode_validity,
                      np.around(cooling_first_mode_coefficient), np.around(cooling_first_mode_intercept),
                      cooling_first_mode_reg_kind, np.around(cooling_first_mode_reg_r2, 2),
                      np.around(cooling_second_mean), np.around(cooling_second_std), cooling_second_mode_validity,
                      np.around(cooling_second_mode_coefficient), np.around(cooling_second_mode_intercept),
                      cooling_second_mode_reg_kind, np.around(cooling_second_mode_reg_r2, 2),
                      np.around(analyse['cooling']['setpoint']['setpoint']), analyse['cooling']['setpoint']['exist'],
                      heating_modes_count,
                      np.around(heating_first_mean), np.around(heating_first_std), heating_first_mode_validity,
                      np.around(heating_first_mode_coefficient), np.around(heating_first_mode_intercept),
                      heating_first_mode_reg_kind, np.around(heating_first_mode_reg_r2, 2),
                      np.around(heating_second_mean), np.around(heating_second_std), heating_second_mode_validity,
                      np.around(heating_second_mode_coefficient), np.around(heating_second_mode_intercept),
                      heating_second_mode_reg_kind, np.around(heating_second_mode_reg_r2, 2),
                      np.around(analyse['heating']['setpoint']['setpoint']), analyse['heating']['setpoint']['exist']]

    hvac_analytics_names = ['pilot_id', 'sampling_rate', 'city', 'zipcode', 'n_id', 'state', 'timezone', 'country',
                            'year_built', 'owner_type', 'n_occupants', 'living_area', 'lot_size', 'total_rooms',
                            'bedrooms', 'heating_type', 'solar_user', 'latitude', 'longitude',
                            'cooling_modes_n', 'cooling_1m_mean', 'cooling_1m_std', 'cooling_1m_validity',
                            'cooling_1m_coefficient', 'cooling_1m_intercept', 'cooling_1m_reg_kind',
                            'cooling_1m_reg_r2', 'cooling_2m_mean', 'cooling_2m_std', 'cooling_2m_validity',
                            'cooling_2m_coefficient', 'cooling_2m_intercept', 'cooling_2m_reg_kind',
                            'cooling_2m_reg_r2', 'cooling_setpoint', 'cooling_setpoint_exist',
                            'heating_modes_n', 'heating_1m_mean', 'heating_1m_std', 'heating_1m_validity',
                            'heating_1m_coefficient', 'heating_1m_intercept', 'heating_1m_reg_kind',
                            'heating_1m_reg_r2', 'heating_2m_mean', 'heating_2m_std', 'heating_2m_validity',
                            'heating_2m_coefficient', 'heating_2m_intercept', 'heating_2m_reg_kind',
                            'heating_2m_reg_r2', 'heating_setpoint', 'heating_setpoint_exist']

    # --------------------------------------------------------------------------------------------------------------

    # getting ao cooling results
    ao_cooling = disagg_output_object.get('ao_seasonality', {}).get('epoch_cooling')
    ao_cooling_exist = len(ao_cooling[ao_cooling > 0]) > 0

    # extracting ao cooling metrics
    if ao_cooling_exist:
        ao_cooling_mean = np.mean(ao_cooling[ao_cooling > 0])
        ao_cooling_std = np.std(ao_cooling[ao_cooling > 0])
    else:
        ao_cooling_mean = 0.0
        ao_cooling_std = 0.0

    # getting ao heating results
    ao_heating = disagg_output_object.get('ao_seasonality', {}).get('epoch_heating')
    ao_heating_exist = len(ao_heating[ao_heating > 0]) > 0

    # extracting ao heating metrics
    if ao_heating_exist:
        ao_heating_mean = np.mean(ao_heating[ao_heating > 0])
        ao_heating_std = np.std(ao_heating[ao_heating > 0])
    else:
        ao_heating_mean = 0.0
        ao_heating_std = 0.0

    # creating analytics list for hsm
    hvac_analytics_hsm = [config['sampling_rate'],
                          np.around(cooling_first_mean), np.around(cooling_first_std),
                          np.around(cooling_first_mode_coefficient),
                          np.around(cooling_second_mean), np.around(cooling_second_std),
                          np.around(cooling_second_mode_coefficient),
                          np.around(ao_cooling_mean), np.around(ao_cooling_std),
                          np.around(analyse['cooling']['setpoint']['setpoint']),

                          np.around(heating_first_mean), np.around(heating_first_std),
                          np.around(heating_first_mode_coefficient),
                          np.around(heating_second_mean), np.around(heating_second_std),
                          np.around(heating_second_mode_coefficient),
                          np.around(ao_heating_mean), np.around(ao_heating_std),
                          np.around(analyse['heating']['setpoint']['setpoint'])]

    # sanitizing hsm list entries
    hvac_analytics_hsm = convert_analytics_hsm_to_float(hvac_analytics_hsm)

    # column names of analytics hsm list
    hvac_analytics_hsm_names = ['sampling_rate',
                                'cooling_1m_mean', 'cooling_1m_std',
                                'cooling_1m_coefficient',
                                'cooling_2m_mean', 'cooling_2m_std',
                                'cooling_2m_coefficient',
                                'ao_cooling_mean', 'ao_cooling_std',
                                'cooling_setpoint',

                                'heating_1m_mean', 'heating_1m_std',
                                'heating_1m_coefficient',
                                'heating_2m_mean', 'heating_2m_std',
                                'heating_2m_coefficient',
                                'ao_heating_mean', 'ao_heating_std',
                                'heating_setpoint']

    # adding analytics key in hsm
    disagg_output_object['created_hsm']['hvac']['attributes']['analytics'] = hvac_analytics_hsm

    # --------------------------------------------------------------------------------------------------------------

    if disagg_input_object['switch']['hvac']['dump_metrics']:
        df_hvac_analytics = pd.DataFrame([hvac_analytics], columns=hvac_analytics_names)
        uuid = disagg_input_object['config']['uuid']
        hvac_analytics_dir = os.path.join('../', "hvac_analytics")

        if not os.path.exists(hvac_analytics_dir):
            os.makedirs(hvac_analytics_dir)
        pd.DataFrame.to_csv(df_hvac_analytics, hvac_analytics_dir + '/' + uuid + '.csv', header=False, index=False)

        df_hvac_analytics_hsm = pd.DataFrame([hvac_analytics_hsm], columns=hvac_analytics_hsm_names)
        pd.DataFrame.to_csv(df_hvac_analytics_hsm, hvac_analytics_dir + '/' + uuid + '_hsm.csv', header=False, index=False)


def initialise_meta_data_keys(meta_data, meta_keys):
    """ Utility function to initialise the meta data object with all required keys"""
    for attribute in meta_keys:
        if attribute not in meta_data.keys():
            meta_data[attribute] = 'default'

    return meta_data
