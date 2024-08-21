"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to plot smb monthly consumptions
"""

# Import python packages

import os
import datetime
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params

matplotlib.use('Agg')


def plot_monthly_bar_smb(month_ao_hvac_res_net, disagg_input_object, disagg_output_object, monthly_info):
    """
    Function to generatue month level consumption bar plots

    Parameters:

        month_ao_hvac_res_net   (np.ndarray)       : Contains month level ao and hvac consumption
        disagg_input_object     (dict)             : Contains user level input information
        disagg_output_object    (dict)             : Contains user level output information
        monthly_info            (dict)             : Contains month level appliance consumption information


    Returns:

        None

    """

    static_params = hvac_static_params()

    month_baseload = monthly_info.get('month_baseload')
    month_extra_ao = monthly_info.get('month_extra_ao')
    month_operational = monthly_info.get('month_operational')
    month_ac = monthly_info.get('month_ac')
    month_ac_close = monthly_info.get('month_ac_close')
    month_sh = monthly_info.get('month_sh')
    month_sh_close = monthly_info.get('month_sh_close')
    month_others = monthly_info.get('month_others')

    features = disagg_output_object.get('analytics').get('values')
    residual_stability = features.get('residual_stability')
    residual_rsquare = features.get('residual_rsquare')

    global_config = disagg_input_object.get('config')

    month_labels = [datetime.utcfromtimestamp(month_ao_hvac_res_net[i, 0]).strftime('%b-%Y') for i in
                    range(month_ao_hvac_res_net.shape[0])]

    # Initializing figure
    figure, axis_array = plt.subplots(figsize=(10, 7))
    array_twinx = axis_array.twinx()
    width = 0.85
    index = np.arange(len(month_labels))

    p1 = axis_array.bar(index, month_baseload, width, color='limegreen', alpha=1, edgecolor='green')
    p10 = axis_array.bar(index, month_extra_ao, width, bottom=month_baseload, color='green', alpha=1, edgecolor='green')
    p8 = axis_array.bar(index, month_operational, width, bottom=month_baseload + month_extra_ao,
                        color='mediumorchid', alpha=1, edgecolor='darkviolet')
    p2 = axis_array.bar(index, month_ac, width, bottom=month_baseload + month_extra_ao + month_operational,
                        color='dodgerblue', alpha=1, edgecolor='blue')
    p11 = axis_array.bar(index, month_ac_close, width,
                         bottom=month_baseload + month_extra_ao + month_operational + month_ac,
                         color='blue', alpha=1, edgecolor='blue')
    p9 = axis_array.bar(index, month_sh, width,
                        bottom=month_baseload + month_extra_ao + month_operational + month_ac + month_ac_close,
                        color='orangered', alpha=1, edgecolor='red')
    p12 = axis_array.bar(index, month_sh_close, width,
                         bottom=month_baseload + month_extra_ao + month_operational + month_ac + month_ac_close + month_sh,
                         color='red', alpha=1, edgecolor='red')
    p3 = axis_array.bar(index, month_others, width,
                        bottom=month_baseload + month_extra_ao + month_operational + month_ac + month_sh + month_ac_close + month_sh_close,
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

    epoch_input_data = disagg_input_object.get('input_data')
    input_df = pd.DataFrame(epoch_input_data)
    input_df.columns = Cgbdisagg.INPUT_COLUMN_NAMES

    df_min_mean_max = input_df.groupby('month')['temperature'].agg([pd.np.min, pd.np.max, pd.np.mean])

    p5 = array_twinx.plot(index, df_min_mean_max['amin'], 'b:', label='Min Temperature', linewidth=1.5)
    p6 = array_twinx.plot(index, df_min_mean_max['mean'], 'y:', label='Mean Temperature', linewidth=2)
    p7 = array_twinx.plot(index, df_min_mean_max['amax'], 'r:', label='Max Temperature', linewidth=1.5)
    array_twinx.set_ylabel('Temperature (F)', fontsize=7)
    array_twinx.set_yticks([10 * i for i in range(13)])

    axis_array.set_xticklabels(month_labels, fontdict={'fontsize': 7, 'verticalalignment': 'top'}, minor=False,
                               rotation=90)

    plt.legend((p1[0], p2[0], p3[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0]),
               ('AO', 'AC', 'OTHER', 'Min Temperature', 'Mean Temperature', 'Max Temperature', 'OP', 'SH', 'Xtra_AO',
                'Close AC', 'Close SH'),
               fontsize=7, loc='upper center', ncol=7, framealpha=0.1)

    plt.tight_layout()

    plot_dir = static_params.get('path').get('hvac_plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_dir = plot_dir + '/' + global_config['uuid']
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = \
        plot_dir + '/bar_plot_' + 'smb' + '_' + global_config['disagg_mode'] + '_' + global_config['uuid'] + \
        '.png'

    if os.path.isfile(image_location):
        os.remove(image_location)

    if plot_dir:
        plt.savefig(image_location, dpi=250)

    plt.close()

    del figure
