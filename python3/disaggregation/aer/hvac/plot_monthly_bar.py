"""
Author - Abhinav Srivastava
Date - 22/10/18
Call the hvac disaggregation module and get results
"""

# Import python packages
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
matplotlib.use('Agg')

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params


def plot_monthly_bar(disagg_input_object, disagg_output_object, month_ao_hvac_res_net, column_index, stage):
    """
    Function to generate month level hvac consumtion bar plots

    Parameters:
        disagg_input_object     (dict)         : Dictionary containing all input attributes
        disagg_output_object    (dict)         : Dictionary containing all output attributes
        month_ao_hvac_res_net   (np.ndarray)   : Array containing | month-ao-ac-sh-residue-net energies
        column_index            (dict)         : Dictionary containing column identifier indices of ao-ac-sh
        stage                   (str)          : String to identify in bar plot is being made before/after post-processing

    Returns:
        None
    """

    if stage == 'processed':

        static_params = hvac_static_params()

        features = disagg_output_object['analytics']['values']
        residual_stability = features['residual_stability']
        residual_rsquare = features['residual_rsquare']

        global_config = disagg_input_object.get('config')

        month_labels = [datetime.utcfromtimestamp(month_ao_hvac_res_net[i, 0]).strftime('%b-%Y') for i in
                        range(month_ao_hvac_res_net.shape[0])]

        ao_cooling = disagg_output_object['ao_seasonality']['cooling'] / Cgbdisagg.WH_IN_1_KWH
        ao_heating = disagg_output_object['ao_seasonality']['heating'] / Cgbdisagg.WH_IN_1_KWH
        ao_baseload = disagg_output_object['ao_seasonality']['baseload'] / Cgbdisagg.WH_IN_1_KWH

        ac_consumption = month_ao_hvac_res_net[:, column_index['ac_od']] / Cgbdisagg.WH_IN_1_KWH
        sh_consumption = month_ao_hvac_res_net[:, column_index['sh_od']] / Cgbdisagg.WH_IN_1_KWH

        residual = month_ao_hvac_res_net[:, 4] / Cgbdisagg.WH_IN_1_KWH

        if disagg_input_object['switch']['plot_level'] >= 1:

            figure, axis_array = plt.subplots(figsize=(10, 7))
            array_twinx = axis_array.twinx()

            width = 0.85
            index = np.arange(len(month_labels))
            p1 = axis_array.bar(index, ao_baseload, width, color='limegreen', alpha=1, edgecolor='green')
            p8 = axis_array.bar(index, ao_cooling, width, bottom=ao_baseload,
                                color='blue', alpha=1, edgecolor='green')
            p2 = axis_array.bar(index, ac_consumption, width, bottom=ao_baseload + ao_cooling,
                                color='dodgerblue', alpha=1, edgecolor='blue')
            p9 = axis_array.bar(index, ao_heating, width, bottom=ao_baseload + ao_cooling + ac_consumption,
                                color='red', alpha=1, edgecolor='green')
            p3 = axis_array.bar(index, sh_consumption, width,
                                bottom=ao_baseload + ao_cooling + ac_consumption + ao_heating,
                                color='orangered', alpha=1, edgecolor='red')
            p4 = axis_array.bar(index, residual, width,
                                bottom=ao_baseload + ao_cooling + ac_consumption + ao_heating + sh_consumption,
                                color='black', alpha=0.85, edgecolor='black')

            axis_array.set_ylabel('Monthly Consumption (kwh)', fontsize=9)
            axis_array.set_title('Monthly Disagg   |    Pilot id : {}   |   Sampling rate : {}   |   '
                                 'Stability : {}  |  Residue R2 : {} \n \n '
                                 'AC : {} {}F : mu {} : std {}        '
                                 'SH : {} {}F : mu {} : std {} '.format(global_config['pilot_id'],
                                                                        global_config['sampling_rate'],
                                                                        residual_stability, residual_rsquare,
                                                                        features['cooling']['setpoint']['exist'],
                                                                        features['cooling']['setpoint']['setpoint'],
                                                                        features['cooling']['detection']['means'],
                                                                        features['cooling']['detection']['std'],
                                                                        features['heating']['setpoint']['exist'],
                                                                        features['heating']['setpoint']['setpoint'],
                                                                        features['heating']['detection']['means'],
                                                                        features['heating']['detection']['std']),
                                 fontsize=9)

            axis_array.set_xticks(index)

            epoch_input_data = disagg_input_object['input_data']

            columns = Cgbdisagg.INPUT_COLUMN_NAMES

            epoch_input_data = epoch_input_data[:, :len(columns)]

            input_df = pd.DataFrame(epoch_input_data)

            input_df.columns = columns

            df_min_mean_max = input_df.groupby('month')['temperature'].agg([pd.np.min, pd.np.max, pd.np.mean])

            p5 = array_twinx.plot(index, df_min_mean_max['amin'], 'b:', label='Min Temperature', linewidth=1.5)
            p6 = array_twinx.plot(index, df_min_mean_max['mean'], 'y:', label='Mean Temperature', linewidth=2)
            p7 = array_twinx.plot(index, df_min_mean_max['amax'], 'r:', label='Max Temperature', linewidth=1.5)
            array_twinx.set_ylabel('Temperature (F)', fontsize=7)
            array_twinx.set_yticks([10 * i for i in range(13)])

            axis_array.set_xticklabels(month_labels, fontdict={'fontsize': 7, 'verticalalignment': 'top'}, minor=False,
                                       rotation=90)

            plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]),
                       ('AO', 'AC', 'SH', 'OTHER', 'Min Temperature', 'Mean Temperature', 'Max Temperature',
                        'AO-Cooling', 'AO-Heating'),
                       fontsize=7, loc='upper center', ncol=7, framealpha=0.1)

            plt.tight_layout()

            plot_dir = static_params.get('path').get('hvac_plots')
            start, end = np.nanmin(disagg_input_object.get('input_data')[:, Cgbdisagg.INPUT_EPOCH_IDX]), np.nanmax(
                disagg_input_object.get('input_data')[:, Cgbdisagg.INPUT_EPOCH_IDX])
            mode = disagg_input_object['config'].get('disagg_mode', 'historical').lower()

            plot_dir = plot_dir + '/' + global_config['uuid']
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            image_location = plot_dir + "/{disagg_mode}_bar_plot_".format(disagg_mode=mode) + \
                "{user}_{start}_{end}_".format(user=global_config['uuid'], start=start, end=end) + ".png"

            if os.path.isfile(image_location):
                os.remove(image_location)

            if plot_dir:
                plt.savefig(image_location, dpi=250)

            plt.close()

            del figure
