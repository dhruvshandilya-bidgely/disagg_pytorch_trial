"""
Author - Paras Tehria
Date - 12/11/19
This module converts the data to suitable format for solar detection
"""

# Import python packages

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn import preprocessing


def dump_heatmap_raw(y_signal_raw, y_signal_sun, so_config, probability_solar, conf_arr, plot_dir, confidence, start_date, end_date, kind):
    """
    Function to save local csv files

        Parameters:
            y_signal_raw        (np.ndarray)      : 2-d raw consumption data where each row is a day
            y_signal_sun        (np.ndarray)      : 2-d sunlight presence data where each row is a day
            so_config            (dict)           : module config parameters
            probability_solar   (list)            : instance probability list
            conf_arr            (list)            : Confidence array of all the runs
            plot_dir            (str)             : Directory to save plots
            confidence          (float)           : LGBM confidence_output
            start_date          (int)             : Start date of solar panel presence
            end_date            (int)             : End date of solar panel presence
            kind                (str)             : Solar panel present throughout or installation/removal

        Returns:
    """
    # Capping negative values to zero helps in capturing solar signals on normalised data

    y_signal_raw[y_signal_raw < 0] = 0

    # Generating solar presence pivot table

    percentile_cap = so_config.get('prep_solar_data').get('percentile_cap')
    percentile_array = np.tile(np.percentile(deepcopy(y_signal_raw), percentile_cap, axis=1).reshape(-1, 1),
                               y_signal_raw.shape[1])
    y_signal_normalised = np.round(np.minimum(deepcopy(y_signal_raw), percentile_array), 2)
    y_signal_normalised[np.isnan(y_signal_normalised)] = 0
    y_signal_normalised = preprocessing.minmax_scale(y_signal_normalised.T).T

    list_hmap_arr = [y_signal_raw, y_signal_normalised, y_signal_sun]
    cmap_list = ['jet', 'hot', 'jet']
    title_list = ['Raw Data', 'Normalised', 'Sunlight']

    plt.figure(figsize=(7.0 * len(list_hmap_arr), 15.0))
    plt.xticks(rotation=90)
    plt.rcParams['font.size'] = 10

    for i in range(1, len(list_hmap_arr) + 1):
        plt.subplot(1, len(list_hmap_arr), i)
        plt.gca().set_title(title_list[i - 1])
        sns.heatmap(list_hmap_arr[i - 1], cmap=cmap_list[i - 1], cbar=True, xticklabels=8, yticklabels=30)
        plt.axhline(start_date, color='black', lw=2)
        plt.axhline(end_date, color='black', lw=2)

    plt.xticks(rotation=90)
    plt.yticks(rotation=90)

    conf_string_c = str(round(confidence, 2))
    probability_string_c = ", ".join([str(x) for x in probability_solar[-12:]])

    kind = "Installation" if kind==1 else "Removal" if kind==2 else "Present Throughout" if kind==0 else "No Solar"
    plt.suptitle(
        "UUID: " + so_config.get('uuid') + ", Pilot: " + str(
            so_config.get('pilot')) + "\n" + "CNN Raw_data_probabilities: " + probability_string_c + "\n" +
        "LGB Confidence: " + conf_string_c + "\n" + "Kind: " + kind, fontsize=17)

    # Saving Plots

    # Check if the target directory exists (if not, create one)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Dump the plot and close the file

    plt.savefig(plot_dir + '/' + so_config.get('uuid') + '_heatmap.png')
    plt.close()

    return None
