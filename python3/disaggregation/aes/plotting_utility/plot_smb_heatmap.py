"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to plot smb specific heatmaps
"""

# Import python packages
import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import functions from within the project
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import hvac_static_params

matplotlib.use('Agg')


def generate_app_heatmap_smb(disagg_input_object, appliance_df_deepcopy, plot_info, smb_type=False):
    """
    Function to generate epoch level heatmaps

    Parameters:

        disagg_input_object     (dict)              : Contains user level input information
        appliance_df_deepcopy   (pd.DataFrame)      : Contains appliance level consumption information
        plot_info               (dict)              : Contains appliance consumption information
        smb_type                (Bool)              : Bool to identify if extra appliances are to be plotted

    Returns:

        None

    """

    static_params = hvac_static_params()
    alt_open_close = plot_info.get('alt_open_close')
    energy_heatmap = plot_info.get('energy_heatmap')
    temperature_heatmap = plot_info.get('temperature_heatmap')
    bl_heatmap = plot_info.get('bl_heatmap')
    grey_heatmap = plot_info.get('ao_grey')
    operational_heatmap = plot_info.get('operational_heatmap')
    ac_heatmap_open = plot_info.get('ac_heatmap')
    sh_heatmap_open = plot_info.get('sh_heatmap')
    others_heatmap = plot_info.get('others_heatmap')
    open_close = plot_info.get('open_close')
    external_lighting = plot_info.get('external_lighting')
    energy_external_removed = energy_heatmap - bl_heatmap - external_lighting

    if smb_type:
        statistical_input = plot_info.get('statistical_input_data')
        refrigerator = plot_info.get('refrigerator')
        cooking = plot_info.get('cooking')
        equipment = plot_info.get('equipments')
        internal_lighting = plot_info.get('internal_light')
        final_residue = statistical_input - refrigerator - bl_heatmap - internal_lighting - equipment - cooking
        statistical_input[statistical_input > np.nanpercentile(statistical_input, 99)] = np.nanpercentile(
            statistical_input, 99)
        final_residue[final_residue > np.nanpercentile(final_residue, 99)] = np.nanpercentile(final_residue, 99)

    else:
        others_heatmap = energy_heatmap - bl_heatmap - grey_heatmap - operational_heatmap - ac_heatmap_open - \
                         sh_heatmap_open - external_lighting

    # To generate one day SMB
    one_day_smb = False

    if one_day_smb:
        day_arr = np.array([entry.day for entry in energy_heatmap.index])
        month_arr = np.array([entry.month for entry in energy_heatmap.index])
        year_arr = np.array([entry.year for entry in energy_heatmap.index])

        df_temp = pd.DataFrame()
        day = 5
        month = 7
        year = 2017
        df_temp['Net Cons'] = energy_heatmap.loc[(day_arr == day) & (month_arr == month) & (year_arr == year)].iloc[0]
        df_temp['Always On'] = bl_heatmap.loc[(day_arr == day) & (month_arr == month) & (year_arr == year)].iloc[0]
        df_temp['Operational Load'] = \
            operational_heatmap.loc[(day_arr == day) & (month_arr == month) & (year_arr == year)].iloc[0]
        df_temp['Cooling'] = ac_heatmap_open.loc[(day_arr == day) & (month_arr == month) & (year_arr == year)].iloc[0]
        df_temp['Heating'] = sh_heatmap_open.loc[(day_arr == day) & (month_arr == month) & (year_arr == year)].iloc[0]
        df_temp['Work Hours'] = open_close.loc[(day_arr == day) & (month_arr == month) & (year_arr == year)].iloc[0]
        df_temp['time'] = df_temp.index.astype(str)

        df_temp.to_csv('/Users/abhi/Documents/2020/smb/1_day_smb.csv', index=False)

    if smb_type:
        fig_heatmap, axn = plt.subplots(1, 17, sharey=True)
    else:
        fig_heatmap, axn = plt.subplots(1, 12, sharey=True)
    fig_heatmap.set_size_inches(32, 12)

    e_max = np.nanmax(energy_heatmap)
    t_max = np.nanpercentile(temperature_heatmap, 99)
    t_min = np.nanmin(temperature_heatmap)

    if t_min < 80:
        t_max = 80
    elif t_max > 110:
        t_max = 110

    if t_min > 10:
        t_min = 10

    count = 0
    sns.heatmap(energy_heatmap, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    axn.flat[count].set_title("Raw Energy")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(temperature_heatmap, ax=axn.flat[count], cmap='jet', cbar=True, xticklabels=4, yticklabels=30,
                vmin=t_min, vmax=t_max)
    axn.flat[count].set_title("Temperature (F)")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(bl_heatmap, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)
    axn.flat[count].set_title("AO")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(external_lighting, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)
    axn.flat[count].set_title("External_Lighting")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(energy_external_removed, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30,
                vmin=0, vmax=e_max)
    axn.flat[count].set_title("Raw - (External+AO)")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(open_close, ax=axn.flat[count], cmap='Reds', cbar=False, xticklabels=4, yticklabels=30, vmin=0, vmax=2)
    axn.flat[count].set_title("Work Hours")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(alt_open_close, ax=axn.flat[count], cmap='Reds', cbar=False, xticklabels=4, yticklabels=30, vmin=0,
                vmax=2)
    axn.flat[count].set_title("Alternate Work Hours")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(grey_heatmap, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)
    axn.flat[count].set_title("Extra AO")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(operational_heatmap, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)
    axn.flat[count].set_title("Operational")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(ac_heatmap_open, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)
    axn.flat[count].set_title("AC")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(sh_heatmap_open, ax=axn.flat[count], cmap='hot', cbar=False, xticklabels=4, yticklabels=30, vmin=0)
    axn.flat[count].set_title("SH")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    count += 1
    sns.heatmap(others_heatmap, ax=axn.flat[count], cmap='hot', cbar=True, xticklabels=4, yticklabels=30, vmin=0)
    axn.flat[count].set_title("Others")
    axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    if smb_type:
        count += 1
        sns.heatmap(statistical_input, ax=axn.flat[count], cmap='hot', cbar=True, xticklabels=4, yticklabels=30, vmin=0)
        axn.flat[count].set_title("Statistical_Input")
        axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

        count += 1
        sns.heatmap(internal_lighting, ax=axn.flat[count], cmap='hot', cbar=True, xticklabels=4, yticklabels=30)
        axn.flat[count].set_title("Internal Lighting")
        axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

        count += 1
        sns.heatmap(refrigerator, ax=axn.flat[count], cmap='hot', cbar=True, xticklabels=4, yticklabels=30)
        axn.flat[count].set_title("Ref_out")
        axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

        count += 1
        sns.heatmap(equipment, ax=axn.flat[count], cmap='hot', cbar=True, xticklabels=4, yticklabels=30)
        axn.flat[count].set_title("Equipments")
        axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

        count += 1
        sns.heatmap(final_residue, ax=axn.flat[count], cmap='hot', cbar=True, xticklabels=4, yticklabels=30)
        axn.flat[count].set_title("Final Residue")
        axn.flat[count].tick_params(axis='x', which='major', labelsize=7)

    # Turning off axes labels

    for axis_idx in range(len(axn.flat)):
        x_axis = axn.flat[axis_idx].get_xaxis()
        x_label = x_axis.get_label()
        x_label.set_visible(False)

        y_axis = axn.flat[axis_idx].get_yaxis()
        y_label = y_axis.get_label()
        y_label.set_visible(False)

    plt.yticks(rotation=0)

    plot_dir = static_params.get('path').get('hvac_plots')

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    uuid = disagg_input_object.get('config').get('uuid')
    mode = disagg_input_object.get('config').get('disagg_mode')
    plot_dir = plot_dir + '/' + uuid

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    image_location = plot_dir + '/smb_v2_appliance_' + uuid + '_' + mode

    if os.path.isfile(image_location):
        os.remove(image_location)

    plt.savefig(image_location + '.png', dpi=250)
    plt.close('all')

    file_location = image_location + '.csv'
    pd.DataFrame(appliance_df_deepcopy).to_csv(file_location)

    file_2_location = image_location + '_alt_open_close.csv'
    pd.DataFrame(alt_open_close).to_csv(file_2_location)
