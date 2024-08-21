"""
Author - Abhinav
Date - 10/10/2018

Getting HSM Attributes
"""

import numpy as np


def get_mode_limits(mode_limits_list):
    """
    Function to get mode limits of ac and sh

    Parameters:
        mode_limits_list (list)     : List of ac or sh mode limits encoded

    Returns:
        mode_limits (list)          : List of ac or sh mode limits decoded
    """

    first_mode_limits = mode_limits_list[:2]
    second_mode_limits = mode_limits_list[2:]
    mode_limits = [first_mode_limits, second_mode_limits]

    return mode_limits


def fill_ac_cluster_info(number_of_clusters, ac_cluster_info, attributes):
    """

    Args:
        number_of_clusters:
        ac_cluster_info:
        attributes:

    Returns:

    """
    for i in range(number_of_clusters):

        ac_cluster_info[attributes['ac_cluster_info_keys'][i]] = {}

        if attributes['ac_cluster_info_keys_kind'][i] == 1:
            ac_cluster_info[attributes['ac_cluster_info_keys'][i]]['regression_kind'] = 'linear'
        elif attributes['ac_cluster_info_keys_kind'][i] == 0.5:
            ac_cluster_info[attributes['ac_cluster_info_keys'][i]]['regression_kind'] = 'root'

        ac_cluster_info[attributes['ac_cluster_info_keys'][i]]['validity'] = bool(
            attributes['ac_cluster_info_keys_validity'][i])

        if attributes['ac_cluster_info_keys_coefficient'][i] == 0:
            continue
        ac_cluster_info[attributes['ac_cluster_info_keys'][i]]['coefficient'] = np.array(
            [[attributes['ac_cluster_info_keys_coefficient'][i]]])

        if attributes['ac_cluster_info_keys_intercept'][i] == 0:
            continue
        ac_cluster_info[attributes['ac_cluster_info_keys'][i]]['intercept'] = np.array(
            [[attributes['ac_cluster_info_keys_intercept'][i]]])


def fill_sh_cluster_info(number_of_clusters, sh_cluster_info, attributes):
    """
    Args:
        number_of_clusters:
        sh_cluster_info:
        attributes:

    Returns:

    """

    for i in range(number_of_clusters):

        sh_cluster_info[attributes['sh_cluster_info_keys'][i]] = {}

        if attributes['sh_cluster_info_keys_kind'][i] == 1:
            sh_cluster_info[attributes['sh_cluster_info_keys'][i]]['regression_kind'] = 'linear'
        elif attributes['sh_cluster_info_keys_kind'][i] == 0.5:
            sh_cluster_info[attributes['sh_cluster_info_keys'][i]]['regression_kind'] = 'root'

        sh_cluster_info[attributes['sh_cluster_info_keys'][i]]['validity'] = bool(
            attributes['sh_cluster_info_keys_validity'][i])

        if attributes['sh_cluster_info_keys_coefficient'][i] == 0:
            continue
        sh_cluster_info[attributes['sh_cluster_info_keys'][i]]['coefficient'] = np.array(
            [[attributes['sh_cluster_info_keys_coefficient'][i]]])

        if attributes['sh_cluster_info_keys_intercept'][i] == 0:
            continue
        sh_cluster_info[attributes['sh_cluster_info_keys'][i]]['intercept'] = np.array(
            [[attributes['sh_cluster_info_keys_intercept'][i]]])


def prepare_cluster_info(attributes):

    """
    Function to prepare cluster info

    Parameters:

        attributes (dict)      : Dictionary of hvac attributes from hsm

    Returns:
        ac_cluster_info (dict) : Dictionary of cooling cluster info
        sh_cluster_info (dict) : Dictionary of heating cluster info
    """

    keys = attributes.keys()

    ac_cluster_info = {}
    sh_cluster_info = {}

    if 'ac_cluster_info_keys' in keys:

        number_of_clusters = len(attributes['ac_cluster_info_keys'])

        fill_ac_cluster_info(number_of_clusters, ac_cluster_info, attributes)

    if 'sh_cluster_info_keys' in keys:

        number_of_clusters = len(attributes['sh_cluster_info_keys'])

        fill_sh_cluster_info(number_of_clusters, sh_cluster_info, attributes)

    return ac_cluster_info, sh_cluster_info


def prepare_analytics_dict(hvac_debug, disagg_output_object):

    """
    Function to prepare analytics dictionary

    Parameters:

        hvac_debug (dict)           : Dictionary containing all hvac related key debug attributes
        disagg_output_object (dict) : Dictionary containing all the disagg inputs

    Returns:
        None
    """

    disagg_output_object['analytics']['values']['cooling']= {}
    disagg_output_object['analytics']['values']['heating'] = {}

    disagg_output_object['analytics']['values']['cooling']['setpoint'] = hvac_debug['estimation']['cdd']
    disagg_output_object['analytics']['values']['heating']['setpoint'] = hvac_debug['estimation']['hdd']

    disagg_output_object['analytics']['values']['cooling']['detection'] = hvac_debug['detection']['cdd']['amplitude_cluster_info']
    disagg_output_object['analytics']['values']['heating']['detection'] = hvac_debug['detection']['hdd']['amplitude_cluster_info']


def get_hsm_attributes(disagg_input_object, disagg_output_object):

    """
    Function to get hsm attributes from last saved hsm

    Parameters:
        disagg_input_object     (dict)      : Dictionary containing all inputs
        disagg_output_object    (dict)      : Dictionary containing all outputs
    Returns:
        hvac_debug              (dict)      : Dictionary containing hvac detection stage attributes read from hsm
    """

    attributes = disagg_input_object['appliances_hsm']['hvac']['attributes']
    user_type = disagg_input_object['config']['user_type'].lower()

    if attributes.get('ac_setpoint') is None:

        # Handles hsm populated using MATLAB
        hvac_debug = {
            'detection': {
                'cdd': {
                    'found': attributes.get('newalgo_cddfound'),
                    'setpoint': attributes.get('csetpoint'),
                    'coeff': attributes.get('ccoeff'),
                    'mu': attributes.get('cmu'),
                    'sigma': attributes.get('csigma'),
                },

                'hdd': {
                    'found': attributes.get('newalgo_hddfound'),
                    'setpoint': attributes.get('hsetpoint'),
                    'mu': attributes.get('hmu'),
                    'coeff': attributes.get('hcoeff'),
                    'sigma': attributes.get('hsigma'),
                },
                'pastBaseLoad': attributes.get('pastBaseLoad'),
                'minBaseLoad': attributes.get('minBaseLoad'),
                'baseLoadFromNewAlgo': attributes.get('baseloadFromNewAlgo'),
            },
        }

    else:

        # Handles hsm populated using python

        ac_mode_limits = get_mode_limits(attributes.get('ac_mode_limits_limits'))
        sh_mode_limits = get_mode_limits(attributes.get('sh_mode_limits_limits'))
        ac_cluster_info, sh_cluster_info = prepare_cluster_info(attributes)

        if user_type != 'smb':
            hvac_debug = {
                'pre_pipeline': {
                    'all_flags': {
                        'hot_cold_normal_user_flag': attributes.get('hot_cold_normal_user_flag', [3])[0],
                        'ac_low_consumption_user': attributes.get('ac_low_consumption_user', [0])[0],
                        'sh_low_consumption_user': attributes.get('sh_low_consumption_user', [0])[0],
                        'is_not_ac': attributes.get('no_ac_user', [0])[0],
                        'is_night_ac': attributes.get('night_ac_user', [0])[0]
                    },
                },
                'detection': {
                    'cdd': {
                        'amplitude_cluster_info': {
                            'means': attributes.get('ac_means'),
                            'std': attributes.get('ac_std'),
                            'cluster_limits': ac_mode_limits,
                            'cluster_overlap': 'mtd',
                            'bin_centers': 'mtd',
                            'all_gaussians': 'mtd',
                            'mode_idx_for_plotting': 'mtd',
                            'number_of_modes': attributes.get('ac_number_of_modes')[0]},

                        'found': bool(attributes.get('ac_found')[0]),
                        'mu': attributes.get('ac_mu')[0]},

                    'hdd': {
                        'amplitude_cluster_info': {
                            'means': attributes.get('sh_means'),
                            'std': attributes.get('sh_std'),
                            'cluster_limits': sh_mode_limits,
                            'cluster_overlap': 'mtd',
                            'bin_centers': 'mtd',
                            'all_gaussians': 'mtd',
                            'mode_idx_for_plotting': 'mtd',
                            'number_of_modes': attributes.get('sh_number_of_modes')[0]},

                        'found': bool(attributes.get('sh_found')[0]),
                        'mu': attributes.get('sh_mu')[0]},
                },

                'estimation': {

                    'cdd': {
                        'aggregation_factor': attributes.get('ac_aggregation_factor')[0],
                        'setpoint': attributes.get('ac_setpoint')[0],
                        'exist': bool(attributes.get('ac_setpoint_exist')[0]),
                        'cluster_info': {'hvac' : ac_cluster_info}},

                    'hdd': {
                        'aggregation_factor': attributes.get('sh_aggregation_factor')[0],
                        'setpoint': attributes.get('sh_setpoint')[0],
                        'exist': bool(attributes.get('sh_setpoint_exist')[0]),
                        'cluster_info': {'hvac' : sh_cluster_info}},},
            }

        else:
            hvac_debug = {
                'detection': {

                    'cdd': {
                        'amplitude_cluster_info': {
                            'means': attributes.get('ac_means'),
                            'std': attributes.get('ac_std'),
                            'cluster_limits': ac_mode_limits,
                            'cluster_overlap': 'mtd',
                            'bin_centers': 'mtd',
                            'all_gaussians': 'mtd',
                            'mode_idx_for_plotting': 'mtd',
                            'number_of_modes': attributes.get('ac_number_of_modes')[0]},

                        'found': bool(attributes.get('ac_found')[0]),
                        'mu': attributes.get('ac_mu')[0]},

                    'hdd': {
                        'amplitude_cluster_info': {
                            'means': attributes.get('sh_means'),
                            'std': attributes.get('sh_std'),
                            'cluster_limits': sh_mode_limits,
                            'cluster_overlap': 'mtd',
                            'bin_centers': 'mtd',
                            'all_gaussians': 'mtd',
                            'mode_idx_for_plotting': 'mtd',
                            'number_of_modes': attributes.get('sh_number_of_modes')[0]},

                        'found': bool(attributes.get('sh_found')[0]),
                        'mu': attributes.get('sh_mu')[0]},
                },

                'estimation': {

                    'cdd': {
                        'setpoint': attributes.get('ac_setpoint')[0],
                        'exist': bool(attributes.get('ac_setpoint_exist')[0]),
                        'cluster_info': {'hvac': ac_cluster_info}},

                    'hdd': {
                        'setpoint': attributes.get('sh_setpoint')[0],
                        'exist': bool(attributes.get('sh_setpoint_exist')[0]),
                        'cluster_info': {'hvac': sh_cluster_info}}, },
            }

        ac_means = np.array(hvac_debug['detection']['cdd']['amplitude_cluster_info']['means'], dtype=float)
        ac_means[ac_means == 123456789] = np.Inf
        hvac_debug['detection']['cdd']['amplitude_cluster_info']['means'] = list(ac_means)

        sh_means = np.array(hvac_debug['detection']['hdd']['amplitude_cluster_info']['means'], dtype=float)
        sh_means[sh_means == 123456789] = np.Inf
        hvac_debug['detection']['hdd']['amplitude_cluster_info']['means'] = list(sh_means)

        ac_mode_limits = np.array(hvac_debug['detection']['cdd']['amplitude_cluster_info']['cluster_limits'], dtype=float)
        ac_mode_limits[ac_mode_limits == 123456789] = np.Inf
        hvac_debug['detection']['cdd']['cluster_limits'] = ((ac_mode_limits[0][0], ac_mode_limits[0][1]), (ac_mode_limits[1][0], ac_mode_limits[1][1]))

        sh_mode_limits = np.array(hvac_debug['detection']['hdd']['amplitude_cluster_info']['cluster_limits'], dtype=float)
        sh_mode_limits[sh_mode_limits == 123456789] = np.Inf
        hvac_debug['detection']['hdd']['cluster_limits'] = ((sh_mode_limits[0][0], sh_mode_limits[0][1]), (sh_mode_limits[1][0], sh_mode_limits[1][1]))

        prepare_analytics_dict(hvac_debug, disagg_output_object)

    return hvac_debug
