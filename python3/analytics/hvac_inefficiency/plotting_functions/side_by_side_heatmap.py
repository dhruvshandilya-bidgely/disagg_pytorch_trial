"""
Author - Anand Kumar Singh
Date - 12th March 2021
Function to plot multiple  heatmaps
"""

import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt

from python3.utils.maths_utils.matlab_utils import superfast_matlab_percentile as super_percentile


def get_cmaps(cmaps, idx):

    """
    Function to get cmaps
    """

    if cmaps is not None:
        cmap = cmaps[idx]
    else:
        cmap = 'hot'

    return cmap


def get_emax_emin(arr, mins, idx):

    """
    Function to get emax and emin

    Parameters:
        arr     (np.ndarray)    : plot array
        mins    (np.ndarray)    : mins
        idx     (int)           : indexes

    Returns:
        emax    (float)         : Max energy
        emin    (float)         : Min energy
    """

    if arr.shape[0] == 0:
        emax = 1
        emin = -1
    else:
        emax = super_percentile(arr, 98)

        if mins is None:
            emin = super_percentile(arr, 5)
            emax = super_percentile(arr, 98)
        else:
            if mins[idx] is None:
                emin = super_percentile(arr, 5)
            else:
                emin = mins[idx]

    return emax, emin


def plot_columns_of_heatmap(list_of_dataframes, titles=[], cmaps=[], sup_titles='all heatmaps', save_path=None, mins=None):

    """
    Function to plot heatmaps

    Parameters:
        list_of_dataframes   (list): List_of_dataframes

    Returns:
        None
    """

    fig = plt.figure(figsize=(16, 8))
    fig_grid_shape = (1, len(list_of_dataframes))

    first_axis = None

    for idx in range(0, len(list_of_dataframes)):
        # Get data frame from list
        dataframe = list_of_dataframes[idx]

        dataframe = dataframe.fillna(0)

        arr = dataframe.values
        valid_idx = (arr != 0)
        arr = arr[valid_idx]

        emax, emin = get_emax_emin(arr, mins, idx)

        if first_axis is None:
            ax = plt.subplot2grid(fig_grid_shape, (0, idx), colspan=1, rowspan=1)
        else:
            ax = plt.subplot2grid(fig_grid_shape, (0, idx), colspan=1, rowspan=1, sharey=first_axis)

        cmap = get_cmaps(cmaps, idx)

        if cmap in ['RdBu', 'RdBu_r']:
            emin = min(emin, 0)
            emax = max(emax, 0)
            sns.heatmap(dataframe, vmax=emax, vmin=emin, cmap=cmap, ax=ax, center=0)
        else:
            sns.heatmap(dataframe, vmax=emax, vmin=emin, cmap=cmap, ax=ax)

        if titles is not None:
            ax.set_title(titles[idx])

        ax.tick_params(axis='x', which='major', labelsize=7)

        if first_axis is not None:
            ax.axes.yaxis.set_visible(False)
        else:
            ax.axes.yaxis.set_visible(True)

        first_axis = ax

    fig.suptitle(sup_titles)

    if save_path is not None:
        plt.savefig(save_path)

    plt.clf()
    plt.cla()
    plt.close()
