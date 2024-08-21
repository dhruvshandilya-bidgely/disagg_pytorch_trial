"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to prepare appliance data array
"""

# Import python packages

import numpy as np
import pandas as pd


def prepare_appliance_df(input_df, disagg_output_object, epoch_ao_hvac_true, column_index):
    """
    Function to suppress bad extra ao candidate values

    Parameters:

        input_df                (pd.DataFrame)     : Contains all 21 column input data for user
        disagg_output_object    (dict)             : Contains user level output information
        epoch_ao_hvac_true      (np.ndarray)       : Contains epoch level av and hvac consumption information
        column_index            (dict)             : Contains appliance identification ids

    Returns:

        appliance_df            (pd.DataFrame)     : Contains all appliance's consumption information

    """

    # getting appliance specific consumptions along with temperature and time info
    appliance_df = pd.DataFrame()
    appliance_df['consumption'] = input_df['consumption']
    appliance_df['temperature'] = input_df['temperature']
    appliance_df['date'] = input_df['date']
    appliance_df['time'] = input_df['time']

    # allocating external_lighting
    appliance_df['external_lighting'] = disagg_output_object['epoch_estimate'][:, disagg_output_object.get('output_write_idx_map').get('li_smb')]

    # allocating baseload
    appliance_df['ao_baseload'] = disagg_output_object.get('ao_seasonality').get('epoch_baseload')

    # allocating ac - on demand and ao
    appliance_df['ac_od'] = epoch_ao_hvac_true[:, column_index['ac']]
    appliance_df['ao_cooling'] = disagg_output_object.get('ao_seasonality').get('epoch_cooling')
    appliance_df['ac_net'] = appliance_df['ac_od'] + appliance_df['ao_cooling']

    # allocating sh - on demand and ao
    appliance_df['sh_od'] = epoch_ao_hvac_true[:, column_index['sh']]
    appliance_df['ao_heating'] = disagg_output_object.get('ao_seasonality').get('epoch_heating')
    appliance_df['sh_net'] = appliance_df['sh_od'] + appliance_df['ao_heating']

    # allocating grey-load
    appliance_df['ao_grey'] = disagg_output_object.get('ao_seasonality').get('epoch_grey')

    # getting and allocating residual load
    appliance_df['residue'] = input_df['consumption'] - (appliance_df['external_lighting'] +
                                                         appliance_df['ao_baseload'] + appliance_df['ac_net'] +
                                                         appliance_df['sh_net'] + appliance_df['ao_grey'])

    # Ensuring residue Sanity
    appliance_df['residue'] = np.round(appliance_df['residue'], 2)
    negative_residue = appliance_df['residue'] < 0
    appliance_df['ac_od'][negative_residue] = appliance_df['ac_od'][negative_residue] + \
                                              appliance_df['residue'][negative_residue]
    appliance_df['ac_od'][appliance_df['ac_od'] < 0] = 0
    appliance_df['sh_od'][negative_residue] = appliance_df['sh_od'][negative_residue] + \
                                              appliance_df['residue'][negative_residue]
    appliance_df['sh_od'][appliance_df['sh_od'] < 0] = 0
    appliance_df['ac_net'] = appliance_df['ac_od'] + appliance_df['ao_cooling']
    appliance_df['sh_net'] = appliance_df['sh_od'] + appliance_df['ao_heating']
    appliance_df['residue'] = input_df['consumption'] - (appliance_df['ao_baseload'] + appliance_df['ac_net'] +
                                                         appliance_df['sh_net']) - appliance_df['ao_grey']
    appliance_df['residue'] = np.around(appliance_df['residue'], 2)

    return appliance_df
