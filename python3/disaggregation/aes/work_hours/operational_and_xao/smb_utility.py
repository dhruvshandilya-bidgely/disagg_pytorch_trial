"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to get smb specific helper utility functions
"""

# Import python packages

import numpy as np
import pandas as pd

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aes.hvac.get_smb_params import get_smb_params


def apply_continuity_filter(label, width=3):

    """
    Function to remove discontinuous work hours from smb

    Parameters:
        label   (np.ndarray)                : Array containing integer data
        width   (int)                       : Carries window information for applying continuity

    Returns:
        label   (np.ndarray)                : Array containing integer data
    """

    # appending zeros to maintain same length array
    append_zeros = np.zeros(width//2)
    filter = np.ones(width)

    # valid epoch should contain more than half of non-zero values
    filter_threshold = (width//2) + 1

    # getting labels in required shape
    n_label = label.shape[0]
    label = np.append(append_zeros, label)
    label = np.append(label, append_zeros)

    # getting 2d stack of labels, day-by-day
    a_hstack = np.hstack(label[i:i + width] for i in range(0, n_label))
    a_vstack = a_hstack.reshape(int(len(a_hstack)/width),width)

    # applying filter for consistency check
    filtered = a_vstack*filter
    filtered_sum = filtered.sum(axis=1)

    # checking qualification of an epoch based on consistency score
    label = (filtered_sum >= filter_threshold).astype(int)

    return label


def prepare_input_df(epoch_input_data, disagg_input_object):

    """
    Function to suppress bad extra ao candidate values

    Parameters:

        epoch_input_data        (np.ndarray)        : Contains epoch level input attributes information
        disagg_input_object     (dict)              : Contains user level input information

    Returns:

        input_df                (pd.DataFrame)      : Contains user level input attributes information

    """

    # initializing data
    input_df = pd.DataFrame(epoch_input_data)
    columns = Cgbdisagg.INPUT_COLUMN_NAMES
    input_df.columns = columns

    # getting timestamps in local time
    input_df['timestamp'] = pd.to_datetime(input_df['epoch'], unit='s')
    timezone = disagg_input_object.get('home_meta_data').get('timezone')
    try:
        input_df['timestamp'] = input_df.timestamp.dt.tz_localize('UTC', ambiguous='infer').dt.tz_convert(
            timezone)
    except (IndexError, KeyError, TypeError):
        input_df['timestamp'] = input_df.timestamp.dt.tz_localize('UTC', ambiguous='NaT').dt.tz_convert(
            timezone)

    # getting ket time related parameters
    input_df['date'] = input_df['timestamp'].dt.date
    input_df['time'] = input_df['timestamp'].dt.time
    input_df['year'] = input_df['timestamp'].dt.year
    input_df['month_'] = input_df['timestamp'].dt.month

    return input_df


def get_base_maps(appliance_df_deepcopy):

    """
    Function to suppress bad extra ao candidate values

    Parameters:

        appliance_df_deepcopy (pd.DataFrame)    : Contains appliance level consumption information

    Returns:

        energy_heatmap      (pd.DataFrame)                  : Contains net consumption data
        temperature_heatmap (pd.Dataframe)                  : Contains temperature data
        bl_heatmap          (pd.Dataframe)                  : Contains baseload consumption data
        grey_heatmap        (pd.DataFrame)                  : Contains extra AO initial consumption information

    """

    # getting energy matrix
    energy_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='consumption',
                                                       aggfunc=sum)
    energy_heatmap = energy_heatmap.fillna(0)
    energy_heatmap = energy_heatmap.astype(int)

    # getting temperature matrix
    temperature_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='temperature',
                                                            aggfunc=sum)
    temperature_heatmap = temperature_heatmap.fillna(0)
    temperature_heatmap = temperature_heatmap.astype(int)

    # getting baseload matrix
    bl_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='ao_baseload', aggfunc=sum)
    bl_heatmap = bl_heatmap.fillna(0)
    bl_heatmap = bl_heatmap.astype(int)

    # getting external_lighting matrix
    ex_l_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='external_lighting',
                                                     aggfunc=sum)
    ex_l_heatmap = ex_l_heatmap.fillna(0)
    ex_l_heatmap = ex_l_heatmap.astype(int)

    # getting grey-data matrix
    grey_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='ao_grey', aggfunc=sum)
    grey_heatmap = grey_heatmap.fillna(0)
    grey_heatmap = grey_heatmap.astype(int)

    return energy_heatmap, temperature_heatmap, bl_heatmap, ex_l_heatmap, grey_heatmap


def get_sanitized_grey_data(grey_map, bl_map):

    """
    Function to suppress bad extra ao candidate values

    Parameters:

        grey_map    (pd.DataFrame)                  : Contains extra AO initial consumption information
        bl_map      (pd.DataFrame)                  : Contains extra baseload consumption information

    Returns:

        grey_heatmap    (pd.DataFrame)                  : Contains extra AO initial consumption information

    """

    smb_params = get_smb_params()

    # qualifying only valid extra ao, based on the extent it is different from baseload
    x_ao_factor_grey = smb_params.get('extra-ao').get('min_factor_grey')
    suppress_grey = np.array(grey_map.sum(axis=1) < bl_map.sum(axis=1) * x_ao_factor_grey)
    grey_values = grey_map.values
    grey_values[suppress_grey, :] = 0

    # qualifying only valid extra ao, based on the extent it is different from grey values
    grey_median_factor = smb_params.get('extra-ao').get('grey_median_factor')
    grey_day_value = grey_values.sum(axis=1)
    grey_median = np.median(grey_day_value[grey_day_value > 0])
    suppress_x_ao = grey_day_value < grey_median_factor * grey_median
    grey_values[suppress_x_ao, :] = 0

    # updating grey_map
    grey_map[:] = grey_values

    return grey_map


def get_hvac_maps(appliance_df_deepcopy):

    """
    Function to get hvac consumption maps

    Parameters:

        appliance_df_deepcopy (pd.DataFrame)                    : Contains appliance level consumption information

    Returns:

        ac_heatmap              (pd.DataFrame)                  : Contains ac consumption
        sh_heatmap              (pd.DataFrame)                  : Contains sh consumption
        residue_heatmap         (pd.DataFrame)                  : Contains residue consumption

    """

    # getting ac data
    ac_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='ac_net', aggfunc=sum)
    ac_heatmap = ac_heatmap.fillna(0)
    ac_heatmap = ac_heatmap.astype(int)

    # getting sh data
    sh_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='sh_net', aggfunc=sum)
    sh_heatmap = sh_heatmap.fillna(0)
    sh_heatmap = sh_heatmap.astype(int)

    # getting residue data
    residue_heatmap = appliance_df_deepcopy.pivot_table(index='date', columns=['time'], values='residue', aggfunc=sum)
    residue_heatmap = residue_heatmap.fillna(0)
    residue_heatmap = residue_heatmap.astype(int)

    return ac_heatmap, sh_heatmap, residue_heatmap


def getting_others_sanity_by_hvac(others_heatmap, ac_map_open, sh_map_open):

    """
    Function to ensure there are no negative values in others because of hvac

    Parameters:

        others_heatmap  (pd.DataFrame)                  : Contains others consumption data
        ac_map_open     (pd.Dataframe)                  : Contains ac consumption data
        sh_map_open     (pd.Dataframe)                  : Contains sh consumption data

    Returns:

        ac_map_open (pd.Dataframe)                  : Contains ac consumption data
        sh_map_open (pd.Dataframe)                  : Contains sh consumption data
    """

    # checking negative residues exist or not
    neg_others = others_heatmap < 0

    # checking if can be handled by ac or sh
    ac_handled = ac_map_open >= abs(others_heatmap)
    sh_handled = sh_map_open >= abs(others_heatmap)

    # handling residues negative values
    ac_cared = ac_handled * neg_others
    sh_cared = sh_handled * neg_others

    # updating hvac
    ac_map_open[ac_cared] = ac_map_open[ac_cared] + others_heatmap[ac_cared]
    sh_map_open[sh_cared] = ac_map_open[sh_cared] + others_heatmap[sh_cared]

    return ac_map_open, sh_map_open


def getting_others_sanity_by_op(others_map, operational_map):

    """
    Function to ensure there are no negative values in others because of operational load

    Parameters:

        others_map      (pd.DataFrame)                  : Contains others consumption data
        operational_map (pd.Dataframe)                  : Contains operational consumption data

    Returns:

        operational_map (pd.Dataframe)                  : Contains operational consumption data
    """

    # checking if negative residues exists of not
    neg_others = others_map < 0

    # checking if operational loads can handle negative residues
    op_handled = operational_map >= abs(others_map)

    # updating operational values
    op_cared = op_handled * neg_others
    operational_map[op_cared] = operational_map[op_cared] + others_map[op_cared]

    return operational_map
