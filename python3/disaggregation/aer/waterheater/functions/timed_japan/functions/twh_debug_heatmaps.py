"""
Author - Sahana M
Date - 07/06/2021
Perform Timed water heater detection
"""

# Import python packages
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def band_plots(debug, uuid, sampling_rate, user_dir, logger_base, pilot='NA'):
    """
    This function is used to plot feature level information for each timed wh run/band
    Parameters:
        debug               (dict)          : Dictionary containing algorithmic outputs
        uuid                (str)           : User id
        sampling_rate       (np.ndarray)    : Sampling rate
        user_dir            (str)           : Path to save the plots
        logger_base         (dict)          : Logger
        pilot               (int)           : Pilot id of the user
    """

    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('band_plots')
    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    # Get the timed water heater debug object

    timed_debug = debug.get('timed_debug')

    # If the twh detection module was run get all the bands

    if timed_debug.get('bands_info'):

        # Extract bands information

        bands_info = timed_debug.get('bands_info')

        # For each band dump its features plot

        for band in bands_info:

            idx = band.split('_')[1]

            # Extract all the data, features and score

            auc_wh_pot_corr = bands_info.get(band).get('features_' + idx).get('auc_wh_pot_corr')
            one_sided_seasonality_score = bands_info.get(band).get('features_' + idx).get('one_sided_seasonality_score')
            reverse_seasonality_score = bands_info.get(band).get('features_' + idx).get('reverse_seasonality_score')
            double_dip_score = bands_info.get(band).get('features_' + idx).get('double_dip_score')
            continuity_score = bands_info.get(band).get('features_' + idx).get('continuity_score')
            auc_std = bands_info.get(band).get('features_' + idx).get('auc_std')
            dur_std = bands_info.get(band).get('features_' + idx).get('dur_std')
            amp_std = bands_info.get(band).get('features_' + idx).get('amp_std')
            final_tb_prob = bands_info.get(band).get('features_' + idx).get('final_tb_prob')
            max_consistency = bands_info.get(band).get('features_' + idx).get('max_consistency')
            max_median_consistency = bands_info.get(band).get('features_' + idx).get('max_median_consistency')
            score = bands_info.get(band).get('score_' + idx)

            original_data_matrix = timed_debug.get('original_data_matrix')
            ytick_labels = timed_debug.get('ytick_labels')
            s_label = timed_debug.get('season_label')
            scored_data_matrix = timed_debug.get('scored_data_matrix')
            plot_scored_matrix = bands_info.get(band).get('features_' + idx).get('plot_scored_matrix')
            box_fit_matrix = bands_info.get(band).get('features_' + idx).get('box_fit_matrix')
            auc_line = bands_info.get(band).get('features_' + idx).get('area_under_curve_line')
            duration_line = bands_info.get(band).get('features_' + idx).get('duration_line')
            amplitude_line_box = bands_info.get(band).get('features_' + idx).get('amplitude_line')
            wh_potential = timed_debug.get('wh_potential')

            fig = plt.figure(figsize=(36, 18))
            fig.suptitle('UUID : ' + str(uuid) + ' Pilot : ' + str(pilot)
                         + ' Sampling Rate : ' + str(sampling_rate) + 's'
                         + ' Band idx : ' + str(idx)
                         + '\n\nauc_wh_pot_score = ' + str(auc_wh_pot_corr)
                         + ', one_sided_seasonality_score = ' + str(one_sided_seasonality_score)
                         + ', reverse_seasonality_score = ' + str(reverse_seasonality_score)
                         + ', double_dip_score = ' + str(double_dip_score)
                         + ', continuity_score = ' + str(continuity_score)
                         + ', auc_std = ' + str(auc_std)
                         + ', dur_std = ' + str(dur_std)
                         + ', amp_std = ' + str(amp_std)
                         + ', final_tb_prob = ' + str(final_tb_prob)
                         + ', max_consistency = ' + str(max_consistency)
                         + ', max_median_consistency = ' + str(max_median_consistency)
                         + '\n\n Score = ' + str(score)
                         , fontsize=15)

            grid = plt.GridSpec(4, 8, hspace=0.2, wspace=0.2)
            raw_data_heatmap_axn = fig.add_subplot(grid[0:, 0])
            sn_label_axn = fig.add_subplot(grid[0:, 1])
            score_whole_heatmap_axn = fig.add_subplot(grid[0:, 2])
            score_heatmap_axn = fig.add_subplot(grid[0:, 3])
            box_heatmap_axn = fig.add_subplot(grid[0:, 4])
            auc_axn = fig.add_subplot(grid[0, 5:])
            dur_axn = fig.add_subplot(grid[1, 5:])
            amp_axn = fig.add_subplot(grid[2, 5:])
            wh_pot_axn = fig.add_subplot(grid[3, 5:])

            original_data_matrix_df = pd.DataFrame(original_data_matrix)
            original_data_matrix_df.index = ytick_labels
            raw_data_heatmap_axn.set_ylabel(ytick_labels)

            sns.heatmap(original_data_matrix_df, cmap='jet', ax=raw_data_heatmap_axn, yticklabels=10)
            sns.heatmap(s_label.reshape(-1, 1), cmap='jet', ax=sn_label_axn)
            sns.heatmap(scored_data_matrix, cmap='jet', ax=score_whole_heatmap_axn)
            sns.heatmap(plot_scored_matrix, cmap='jet', ax=score_heatmap_axn)
            sns.heatmap(box_fit_matrix, cmap='jet', ax=box_heatmap_axn)
            auc_axn.plot(auc_line, linewidth=2)
            dur_axn.plot(duration_line, linewidth=2)
            amp_axn.plot(amplitude_line_box, linewidth=2)
            wh_pot_axn.plot(wh_potential.reshape(-1, 1), linewidth=2)
            wh_pot_axn.set_xticklabels([ytick_labels[i] for i in range(0, len(ytick_labels) - 30, 30)])
            plt.yticks(np.round(np.arange(0, 1.1, 0.1), 1))
            auc_axn.grid()
            dur_axn.grid()
            amp_axn.grid()
            wh_pot_axn.grid()
            raw_data_heatmap_axn.tick_params(axis='y', labelrotation=0)
            raw_data_heatmap_axn.set_title('Input data')
            sn_label_axn.set_title('Season label')
            score_whole_heatmap_axn.set_title('Scored chunks')
            score_heatmap_axn.set_title('Instance')
            box_heatmap_axn.set_title('Box fit data')
            auc_axn.set_title('Area under the curve')
            dur_axn.set_title('Duration')
            amp_axn.set_title('Amplitude')
            wh_pot_axn.set_title('WH Potential')

            plot_dir = user_dir
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            # Dump the plot and close the file

            plt.savefig(plot_dir + '/' + uuid + '_' + str(pilot) + '_' + str(idx) + '.png')
            plt.close()

    else:
        logger.info('Not dumping band plots for timed wh debugging since no bands detected | ')


def estimation_plots(debug, uuid, sampling_rate, user_dir, pilot='NA'):
    """
    This function is used to plot estimation level information for each timed wh run/band
    Parameters:
        debug               (dict)          : Dictionary containing algorithmic outputs
        uuid                (str)           : User id
        sampling_rate       (np.ndarray)    : Sampling rate
        user_dir            (str)           : Path to save the plots
        pilot               (int)           : Pilot id of the user
    """

    # Get the timed water heater debug object

    timed_debug = debug.get('timed_debug')
    original_data_matrix = timed_debug.get('original_data_matrix')

    if debug.get('timed_hld') == 0:
        final_twh_matrix = np.full_like(original_data_matrix, fill_value=0.0)
    else:
        final_twh_matrix = timed_debug.get('final_twh_matrix')

    # Extract all the required data

    confidence = debug.get('timed_confidence_score')
    twh_amplitude = debug.get('timed_wh_amplitude')
    num_runs = timed_debug.get('num_runs')
    original_data_matrix = timed_debug.get('original_data_matrix')
    twh_data_matrix = timed_debug.get('twh_data_matrix')
    scored_data_matrix = timed_debug.get('scored_data_matrix')
    ytick_labels = timed_debug.get('ytick_labels')

    # Plotting

    fig_heatmap, axn = plt.subplots(1, 4)
    fig_heatmap.set_size_inches(24, 24)
    fig_heatmap.suptitle('UUID : ' + str(uuid) + ' Pilot : ' + str(pilot)
                         + ' Sampling Rate: ' + str(sampling_rate) + 's'
                         + '\n Confidence : ' + str(confidence)
                         + ', TWH Amp : ' + str(np.round(twh_amplitude, 2))
                         + ', Num runs : ' + str(num_runs), fontsize=24)

    original_data_matrix_df = pd.DataFrame(original_data_matrix)
    original_data_matrix_df.index = ytick_labels

    sns.heatmap(original_data_matrix_df, cmap='jet', ax=axn[0], yticklabels=10)
    sns.heatmap(twh_data_matrix, cmap='jet', ax=axn[1])
    sns.heatmap(scored_data_matrix, cmap='jet', ax=axn[2])
    sns.heatmap(final_twh_matrix, cmap='jet', ax=axn[3])

    axn.flat[0].set_title('Input data')
    axn.flat[1].set_title('TOI matrix')
    axn.flat[2].set_title('Scored chunks matrix')
    axn.flat[3].set_title('Final WH estimation')

    axn[0].tick_params(axis='y', labelrotation=0)

    plot_dir = user_dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Dump the plot and close the file

    plt.savefig(plot_dir + '/' + uuid + '_ep_' + str(pilot) + '.png')
    plt.close()
