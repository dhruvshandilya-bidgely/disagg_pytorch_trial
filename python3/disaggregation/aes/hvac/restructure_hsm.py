"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to structure and prepare hsm
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.disaggregation.aes.hvac.get_smb_params import get_smb_params


def restructure_atrributes_for_hsm(attributes):

    """
    Function restructures the attributes into format that is desired for hsm storage in cassandra

    Parameters:

        attributes               (dict) : Dictionary containing attributes to be saved in hsm, but in nested form

    Returns:

         restructured_attributes (dict) : Dictionary containing attributes to be saved in hsm, but in restructured form
    """

    smb_params = get_smb_params()

    # base of restructuring attributes
    restructured_attributes = copy.deepcopy(attributes)

    # dictionaries to restructure
    dict_to_structure = ['ac_cluster_info', 'sh_cluster_info']
    array_to_structure = ['ac_mode_limits', 'sh_mode_limits']

    # ac restructured
    ac_means = np.array(restructured_attributes['ac_means'])
    ac_means[ac_means == np.Inf] = smb_params.get('utility').get('large_number_constant')
    restructured_attributes['ac_means'] = list(ac_means)

    # sh restructured
    sh_means = np.array(restructured_attributes['sh_means'])
    sh_means[sh_means == np.Inf] = smb_params.get('utility').get('large_number_constant')
    restructured_attributes['sh_means'] = list(sh_means)

    # ac modes restructured
    ac_mode_limits = np.array(restructured_attributes['ac_mode_limits'])
    ac_mode_limits[ac_mode_limits == np.Inf] = smb_params.get('utility').get('large_number_constant')
    restructured_attributes['ac_mode_limits'] = ((ac_mode_limits[0][0],ac_mode_limits[0][1]), (ac_mode_limits[1][0],ac_mode_limits[1][1]))

    # sh modes restructured
    sh_mode_limits = np.array(restructured_attributes['sh_mode_limits'])
    sh_mode_limits[sh_mode_limits == np.Inf] = smb_params.get('utility').get('large_number_constant')
    restructured_attributes['sh_mode_limits'] = ((sh_mode_limits[0][0],sh_mode_limits[0][1]), (sh_mode_limits[1][0],sh_mode_limits[1][1]))

    # restructuring for each key in attributes
    for main_key in dict_to_structure:

        sub_keys = list(attributes[main_key].keys())

        restructured_attributes[main_key +'_keys'] = sub_keys

        # initializing attributes empty list
        kind_list = []
        validity_list = []
        coefficient_list = []
        intercept_list = []

        # populating attributes
        for cluster in sub_keys:

            if attributes[main_key][cluster]['regression_kind'] == 'root':
                kind_list.append(0.5)
            elif attributes[main_key][cluster]['regression_kind'] == 'linear':
                kind_list.append(1)

            validity_list.append(int(attributes[main_key][cluster]['validity']))

            # noinspection PyBroadException
            try:
                coefficient_list.append(attributes[main_key][cluster]['coefficient'][0][0])
            except (IndexError, KeyError):
                coefficient_list.append(0)

            # noinspection PyBroadException
            try:
                intercept_list.append(attributes[main_key][cluster]['intercept'][0])
            except (IndexError, KeyError):
                intercept_list.append(0)

        # populating restructured attributes
        restructured_attributes[main_key + '_keys' + '_kind'] = kind_list
        restructured_attributes[main_key + '_keys' + '_validity'] = validity_list
        restructured_attributes[main_key + '_keys' + '_coefficient'] = coefficient_list
        restructured_attributes[main_key + '_keys' + '_intercept'] = intercept_list

        restructured_attributes.pop(main_key, None)

    # populating limits
    for main_key in array_to_structure:

        limits = np.array(attributes[main_key])
        restructured_attributes[main_key + '_limits'] = list(limits.flatten())

        restructured_attributes.pop(main_key, None)

    # ac mode limits structured
    ac_mode_limits_limits = np.array(restructured_attributes['ac_mode_limits_limits'])
    ac_mode_limits_limits[ac_mode_limits_limits == np.Inf] = smb_params.get('utility').get('large_number_constant')
    restructured_attributes['ac_mode_limits_limits'] = list(ac_mode_limits_limits)

    # sh mode limits structured
    sh_mode_limits_limits = np.array(restructured_attributes['sh_mode_limits_limits'])
    sh_mode_limits_limits[sh_mode_limits_limits == np.Inf] = smb_params.get('utility').get('large_number_constant')
    restructured_attributes['sh_mode_limits_limits'] = list(sh_mode_limits_limits)

    return restructured_attributes
