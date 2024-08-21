"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to add smb analytics
"""

# Import python packages
import os
import copy
import numpy as np
import pandas as pd
from scipy import stats


def analytics_cooling_2nd_mode(analyse):

    """
    Function to write key attributes of cooling mode related to smb disagg

    Parameters:
        analyse                 (dict)                : Contains smb disagg related key attributes

    Returns:
        cooling_2nd_mode_info   (dict)                : Contains cooling mode related key attributes
    """

    # initializing dictionary to keep key attributes corresponding to 2nd cooling mode

    cooling_2nd_mode_info = []

    # noinspection PyBroadException
    try:

        # accessing mean and std
        cooling_second_mean = analyse.get('cooling').get('detection').get('means')[1]
        cooling_second_std = analyse.get('cooling').get('detection').get('std')[1]

    except (IndexError, KeyError):

        # failsafe mean and std
        cooling_second_mean = 0
        cooling_second_std = 0

    # getting 2nd mode specific cluster info
    cooling_second_mode_reg_info = analyse.get('cooling').get('estimation')[1].get('cluster_info')
    cooling_second_mode_dbscan_cluster_validity = [cooling_second_mode_reg_info[key].get('validity') for key in
                                                   cooling_second_mode_reg_info.keys()]
    cooling_second_mode_validity = any(cooling_second_mode_dbscan_cluster_validity)

    # checking if 2nd mode exists
    if cooling_second_mode_validity:

        # accessing second mode regression coefficient
        cooling_second_mode_coefficient = \
            [cooling_second_mode_reg_info[key].get('coefficient') for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key].get('validity')][0][0][0]

        # accessing second mode regression intercept
        cooling_second_mode_intercept = \
            [cooling_second_mode_reg_info[key].get('intercept') for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key].get('validity')][0][0]

        # accessing second mode regression kind
        cooling_second_mode_reg_kind = \
            [cooling_second_mode_reg_info[key].get('regression_kind') for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key].get('validity')][0]

        # accessing second mode regression degree of fit
        cooling_second_mode_reg_r2 = \
            [cooling_second_mode_reg_info[key].get('r_square') for key in cooling_second_mode_reg_info.keys() if
             cooling_second_mode_reg_info[key].get('validity')][0]
    else:

        # failsafe attributes of second mode
        cooling_second_mode_coefficient = 0
        cooling_second_mode_intercept = 0
        cooling_second_mode_reg_kind = 0
        cooling_second_mode_reg_r2 = 0

    # populating 2nd mode dictionary with attributes
    cooling_2nd_mode_info.append(cooling_second_mean)
    cooling_2nd_mode_info.append(cooling_second_std)
    cooling_2nd_mode_info.append(cooling_second_mode_coefficient)
    cooling_2nd_mode_info.append(cooling_second_mode_intercept)
    cooling_2nd_mode_info.append(cooling_second_mode_reg_kind)
    cooling_2nd_mode_info.append(cooling_second_mode_reg_r2)

    return cooling_2nd_mode_info


def analytics_heating_2nd_mode(analyse):

    """
    Function to write key attributes of heating mode related to smb disagg

    Parameters:
        analyse (dict)                              : Contains smb disagg related key attributes

    Returns:
        heating_2nd_mode_info (dict)                : Contains heating mode related key attributes
    """

    # initializing dictionary to keep key attributes corresponding to 2nd heating mode
    heating_2nd_mode_info = []

    # noinspection PyBroadException
    try:

        # accessing mean and std
        heating_second_mean = analyse.get('heating').get('detection').get('means')[1]
        heating_second_std = analyse.get('heating').get('detection').get('std')[1]

    except (IndexError, KeyError):

        # failsafe mean and std
        heating_second_mean = 0
        heating_second_std = 0

    # getting 2nd mode specific cluster info
    heating_second_mode_reg_info = analyse.get('heating').get('estimation')[1].get('cluster_info')
    heating_second_mode_dbscan_cluster_validity = [heating_second_mode_reg_info[key]['validity'] for key in
                                                   heating_second_mode_reg_info.keys()]
    heating_second_mode_validity = any(heating_second_mode_dbscan_cluster_validity)

    # checking if 2nd mode exists
    if heating_second_mode_validity:

        # accessing second mode regression coefficient
        heating_second_mode_coefficient = \
            [heating_second_mode_reg_info[key].get('coefficient') for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key].get('validity')][0][0][0]

        # accessing second mode regression intercept
        heating_second_mode_intercept = \
            [heating_second_mode_reg_info[key].get('intercept') for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key].get('validity')][0][0]

        # accessing second mode regression kind
        heating_second_mode_reg_kind = \
            [heating_second_mode_reg_info[key].get('regression_kind') for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key].get('validity')][0]

        # accessing second mode regression degree of fit
        heating_second_mode_reg_r2 = \
            [heating_second_mode_reg_info[key].get('r_square') for key in heating_second_mode_reg_info.keys() if
             heating_second_mode_reg_info[key].get('validity')][0]
    else:

        # failsafe attributes of second mode
        heating_second_mode_coefficient = 0
        heating_second_mode_intercept = 0
        heating_second_mode_reg_kind = 0
        heating_second_mode_reg_r2 = 0

    # populating 2nd mode dictionary with attributes
    heating_2nd_mode_info.append(heating_second_mean)
    heating_2nd_mode_info.append(heating_second_std)
    heating_2nd_mode_info.append(heating_second_mode_coefficient)
    heating_2nd_mode_info.append(heating_second_mode_intercept)
    heating_2nd_mode_info.append(heating_second_mode_reg_kind)
    heating_2nd_mode_info.append(heating_second_mode_reg_r2)

    return heating_2nd_mode_info


def map_day(off_day_num):

    """
    Function to map day of the week integer to day string

    Parameters:
        off_day_num (int)                   : Contains user level input information

    Returns:
        day (string)                        : The off day of the week
    """

    # initializing day string
    day = ''

    # overwriting day string based on day numerical value
    if off_day_num == 1:
        day = 'Sunday'
    elif off_day_num == 2:
        day = 'Monday'
    elif off_day_num == 3:
        day = 'Tuesday'
    elif off_day_num == 4:
        day = 'Wednesday'
    elif off_day_num == 5:
        day = 'Thursday'
    elif off_day_num == 6:
        day = 'Friday'
    elif off_day_num == 7:
        day = 'Saturday'

    return day


def write_smb_analytics(disagg_input_object, analyse):

    """
    Function to write key attributes related to smb disagg

    Parameters:
        disagg_input_object (dict)                  : Contains user level input information
        analyse (dict)                              : Contains smb disagg related key attributes

    Returns:
        None
    """

    # accessing config and meta information of user
    global_config = disagg_input_object.get('config')
    config = copy.deepcopy(global_config)
    meta_data = disagg_input_object.get('home_meta_data')

    # getting smb specific key attributes
    smb_info = disagg_input_object.get('switch').get('smb')
    operational_load = smb_info.get('operational_load')
    cooling_open_median = smb_info.get('cooling_open_median')
    cooling_close_median = smb_info.get('cooling_close_median')
    heating_open_median = smb_info.get('heating_open_median')
    heating_close_median = smb_info.get('heating_close_median')
    baseload_median = smb_info.get('baseload_median')
    residue_median = smb_info.get('residue_median')

    # accessing smb_info
    work_info = smb_info.get('info')

    # initializing work hours failsafe for smb analytics
    open_hour = []
    open_minute = []
    close_hour = []
    close_minute = []

    # assigning work hours from each month
    for key, value in work_info.items():
        open_hour.append(value['open'].hour)
        open_minute.append(value['open'].minute)
        close_hour.append(value['close'].hour)
        close_minute.append(value['close'].minute)

    # getting general representative work hours
    open_hour = stats.mode(open_hour)[0][0]
    open_minute = stats.mode(open_minute)[0][0]
    close_hour = stats.mode(close_hour)[0][0]
    close_minute = stats.mode(close_minute)[0][0]

    # noinspection PyBroadException
    try:

        # getting general off days
        off_days = smb_info.get('off_days')
        off_days, off_freq = np.unique(off_days, return_counts=True)
        off_day_num = int(off_days[off_freq == np.max(off_freq)][0])
        off_day = map_day(off_day_num)

    except (TypeError, IndexError, KeyError):

        # failsafe off day is sunday
        off_day = 'Sunday'

    # Cooling related Analytics
    cooling_modes_count = analyse.get('cooling').get('detection').get('number_of_modes')
    cooling_first_mean = analyse.get('cooling').get('detection').get('means')[0]
    cooling_first_std = analyse.get('cooling').get('detection').get('std')[0]
    cooling_first_mode_reg_info = analyse.get('cooling').get('estimation')[0].get('cluster_info')
    cooling_first_mode_dbscan_cluster_validity = [cooling_first_mode_reg_info[key]['validity'] for key in
                                                  cooling_first_mode_reg_info.keys()]
    cooling_first_mode_validity = any(cooling_first_mode_dbscan_cluster_validity)

    # Cooling 1st mode related analytics
    if cooling_first_mode_validity:

        cooling_first_mode_coefficient = \
            [cooling_first_mode_reg_info[key].get('coefficient') for key in cooling_first_mode_reg_info.keys() if
             cooling_first_mode_reg_info[key].get('validity')][0][0][0]
        cooling_first_mode_intercept = \
            [cooling_first_mode_reg_info[key].get('intercept') for key in cooling_first_mode_reg_info.keys() if
             cooling_first_mode_reg_info[key].get('validity')][0][0]
        cooling_first_mode_reg_kind = \
            [cooling_first_mode_reg_info[key].get('regression_kind') for key in cooling_first_mode_reg_info.keys() if
             cooling_first_mode_reg_info[key].get('validity')][0]
        cooling_first_mode_reg_r2 = \
            [cooling_first_mode_reg_info[key].get('r_square') for key in cooling_first_mode_reg_info.keys() if
             cooling_first_mode_reg_info[key].get('validity')][0]
    else:

        # Cooling 1st mode failsafe
        cooling_first_mode_coefficient = 0
        cooling_first_mode_intercept = 0
        cooling_first_mode_reg_kind = 0
        cooling_first_mode_reg_r2 = 0

    cooling_second_mean = 0
    cooling_second_std = 0
    cooling_second_mode_validity = False

    # noinspection PyBroadException
    try:

        # Cooling 2nd mode related analytics
        cooling_2nd_mode_info = analytics_cooling_2nd_mode(analyse)

        cooling_second_mean = cooling_2nd_mode_info[0]
        cooling_second_std = cooling_2nd_mode_info[1]
        cooling_second_mode_coefficient = cooling_2nd_mode_info[2]
        cooling_second_mode_intercept = cooling_2nd_mode_info[3]
        cooling_second_mode_reg_kind = cooling_2nd_mode_info[4]
        cooling_second_mode_reg_r2 = cooling_2nd_mode_info[5]

    except (ValueError, IndexError, KeyError):

        # Cooling 2nd mode failsafe
        cooling_second_mode_validity = 0
        cooling_second_mode_coefficient = 0
        cooling_second_mode_intercept = 0
        cooling_second_mode_reg_kind = 0
        cooling_second_mode_reg_r2 = 0

    # Heating related Analytics
    heating_modes_count = analyse.get('heating').get('detection').get('number_of_modes')
    heating_first_mean = analyse.get('heating').get('detection').get('means')[0]
    heating_first_std = analyse.get('heating').get('detection').get('std')[0]
    heating_first_mode_reg_info = analyse.get('heating').get('estimation')[0].get('cluster_info')
    heating_first_mode_dbscan_cluster_validity = [heating_first_mode_reg_info[key].get('validity') for key in
                                                  heating_first_mode_reg_info.keys()]
    heating_first_mode_validity = any(heating_first_mode_dbscan_cluster_validity)

    # Heating 1st mode regression analytics
    if heating_first_mode_validity:

        heating_first_mode_coefficient = \
            [heating_first_mode_reg_info[key].get('coefficient') for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key].get('validity')][0][0][0]
        heating_first_mode_intercept = \
            [heating_first_mode_reg_info[key].get('intercept') for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key].get('validity')][0][0]
        heating_first_mode_reg_kind = \
            [heating_first_mode_reg_info[key].get('regression_kind') for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key].get('validity')][0]
        heating_first_mode_reg_r2 = \
            [heating_first_mode_reg_info[key].get('r_square') for key in heating_first_mode_reg_info.keys() if
             heating_first_mode_reg_info[key].get('validity')][0]
    else:

        # Heating 1st mode regression failsafe
        heating_first_mode_coefficient = 0
        heating_first_mode_intercept = 0
        heating_first_mode_reg_kind = 0
        heating_first_mode_reg_r2 = 0

    heating_second_mean = 0
    heating_second_std = 0
    heating_second_mode_validity = False

    # noinspection PyBroadException
    try:

        # Heating 2nd mode regression analytics
        heating_2nd_mode_info = analytics_heating_2nd_mode(analyse)

        heating_second_mean = heating_2nd_mode_info[0]
        heating_second_std = heating_2nd_mode_info[1]
        heating_second_mode_coefficient = heating_2nd_mode_info[2]
        heating_second_mode_intercept = heating_2nd_mode_info[3]
        heating_second_mode_reg_kind = heating_2nd_mode_info[4]
        heating_second_mode_reg_r2 = heating_2nd_mode_info[5]

    except (ValueError, IndexError, KeyError):

        # Heating 2nd mode failsafe
        heating_second_mode_validity = 0
        heating_second_mode_coefficient = 0
        heating_second_mode_intercept = 0
        heating_second_mode_reg_kind = 0
        heating_second_mode_reg_r2 = 0

    # meta key format used in analytics frame population
    meta_keys = ['ownershipType', 'numOccupants', 'state', 'city', 'country', 'zipCode', 'timezone', 'yearBuilt',
                 'defaultNId',
                 'livingArea', 'lotSize', 'bedrooms', 'totalRooms', 'spaceHeatingType', 'solarUser', 'latitude',
                 'longitude']

    for attribute in meta_keys:
        if attribute not in meta_data.keys():
            meta_data[attribute] = 'default'

    # creating detailed hvac analytics frame
    hvac_analytics = [config.get('pilot_id'), config.get('sampling_rate'), meta_data.get('city'), meta_data.get('zipCode'),
                      meta_data.get('defaultNId'), meta_data.get('state'), meta_data.get('timezone'), meta_data.get('country'),
                      meta_data.get('yearBuilt'), meta_data.get('ownershipType'), meta_data.get('numOccupants'),
                      meta_data.get('livingArea'), meta_data.get('lotSize'), meta_data.get('totalRooms'), meta_data.get('bedrooms'),
                      meta_data.get('spaceHeatingType'), meta_data.get('solarUser'), meta_data.get('latitude'),
                      meta_data.get('longitude'),
                      cooling_modes_count,
                      np.around(cooling_first_mean), np.around(cooling_first_std), cooling_first_mode_validity,
                      np.around(cooling_first_mode_coefficient), np.around(cooling_first_mode_intercept),
                      cooling_first_mode_reg_kind, np.around(cooling_first_mode_reg_r2, 2),
                      np.around(cooling_second_mean), np.around(cooling_second_std), cooling_second_mode_validity,
                      np.around(cooling_second_mode_coefficient), np.around(cooling_second_mode_intercept),
                      cooling_second_mode_reg_kind, np.around(cooling_second_mode_reg_r2, 2),
                      np.around(analyse.get('cooling').get('setpoint').get('setpoint')), analyse.get('cooling').get('setpoint').get('exist'),
                      heating_modes_count,
                      np.around(heating_first_mean), np.around(heating_first_std), heating_first_mode_validity,
                      np.around(heating_first_mode_coefficient), np.around(heating_first_mode_intercept),
                      heating_first_mode_reg_kind, np.around(heating_first_mode_reg_r2, 2),
                      np.around(heating_second_mean), np.around(heating_second_std), heating_second_mode_validity,
                      np.around(heating_second_mode_coefficient), np.around(heating_second_mode_intercept),
                      heating_second_mode_reg_kind, np.around(heating_second_mode_reg_r2, 2),
                      np.around(analyse.get('heating').get('setpoint').get('setpoint')), analyse.get('heating').get('setpoint').get('exist'),
                      operational_load, cooling_open_median, cooling_close_median, heating_open_median,
                      heating_close_median, baseload_median, residue_median, off_day, open_hour, open_minute, close_hour,
                      close_minute]

    # naming the features
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
                            'heating_2m_reg_r2', 'heating_setpoint', 'heating_setpoint_exist',
                            'operational_load', 'cooling_open_median', 'cooling_close_median', 'heating_open_median',
                            'heating_close_median', 'baseload_median', 'residue_median', 'off_day', 'open_hour',
                            'open_minute', 'close_hour','close_minute']

    # creating analytics frame
    df_hvac_analytics = pd.DataFrame([hvac_analytics], columns=hvac_analytics_names)
    uuid = disagg_input_object['config']['uuid']
    hvac_analytics_dir = os.path.join('../', "hvac_analytics")

    # dumping analytics frame
    if not os.path.exists(hvac_analytics_dir):
        os.makedirs(hvac_analytics_dir)
    pd.DataFrame.to_csv(df_hvac_analytics, hvac_analytics_dir + '/' + uuid + '.csv', header=False, index=False)
