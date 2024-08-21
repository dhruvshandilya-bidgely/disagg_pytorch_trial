"""
Author - Abhinav Srivastava / Mirambika Sikdar
Date - 09/01/24
Wrapper function for post-processing in MTD mode with utility functions defined in the postprocess_hvac file
"""

# Import python packages
import copy
import scipy
import numpy as np

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.hvac.postprocess_hvac import impose_duty_cycle_on_hvac
from python3.disaggregation.aer.hvac.postprocess_hvac import avoid_overshoot_net_consumption
from python3.disaggregation.aer.hvac.postprocess_hvac import remove_fp_hvac
from python3.disaggregation.aer.hvac.postprocess_hvac import suppress_ao_seasonal


def postprocess_hvac_mtd(month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_input_object, disagg_output_object,
                         column_index):
    """
    Function to check overshoot of net consumption and FP addition at bill cycle for MTD mode
    No hvac extraction or addition from residue for MTD mode

    Parameters:

        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies
        disagg_input_object     (dict)          : Dictionary containing all input attributes
        disagg_output_object    (dict)          : Dictionary containing all output attributes
        column_index            (dict)          : Dictionary containing column identifier indices of ao-ac-sh

    Returns:
        month_ao_hvac_res_net   (np.ndarray)    : Array containing | month-ao-ac-sh-residue-net energies (Processed)
        epoch_ao_hvac_true      (np.ndarray)    : Array containing | epoch-ao-ac-sh energies (Processed)
        disagg_output_object    (dict)          : Dictionary containing all output attributes
    """

    std_residual_col = -2
    month_ao_hvac_res_net[:, column_index['ao']] = disagg_output_object['ao_seasonality']['baseload']
    epoch_ao_hvac_true[:, column_index['ao']] = disagg_output_object['ao_seasonality']['epoch_baseload']

    epoch_input_data = copy.deepcopy(disagg_input_object.get('switch').get('hvac_input_data_timed_removed'))
    sampling_rate = disagg_input_object['config']['sampling_rate']
    epochs_per_hour = Cgbdisagg.SEC_IN_HOUR / sampling_rate
    low_consumption_ac_flag = disagg_output_object.get('hvac_debug', {}).get('pre_pipeline', {}).\
        get('all_flags', {}).get('adjust_ac_detection_range_flag', 0)

    _, _, month_idx = scipy.unique(epoch_input_data[:, Cgbdisagg.INPUT_BILL_CYCLE_IDX],
                                   return_index=True, return_inverse=True)
    days_in_bc = np.bincount(month_idx) * (sampling_rate / Cgbdisagg.SEC_IN_DAY)
    days_in_bc_scaled = days_in_bc / Cgbdisagg.DAYS_IN_MONTH
    days_in_bc_scaled = np.clip(days_in_bc_scaled, a_min=None, a_max=2)
    # Duty Cycle HVAC -----------------------
    net_consumption = epoch_input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    ac_od_consumption = epoch_ao_hvac_true[:, column_index['ac']]
    sh_od_consumption = epoch_ao_hvac_true[:, column_index['sh']]
    epoch_ao_hvac_true[:, column_index['ac']] = impose_duty_cycle_on_hvac(ac_od_consumption, net_consumption)
    epoch_ao_hvac_true[:, column_index['sh']] = impose_duty_cycle_on_hvac(sh_od_consumption, net_consumption)

    # Block Handler 4 ----------------------------------- >

    # Ensuring net HVAC consumption do not exceed monthly total consumption
    epoch_ao_hvac_true, month_ao_hvac_res_net = avoid_overshoot_net_consumption(epoch_ao_hvac_true,
                                                                                column_index, disagg_input_object,
                                                                                disagg_output_object)

    # Removal of FP bill cycles of low monthly consumption
    info_carrier_fp_removal = {
        'epoch_ao_hvac_true': epoch_ao_hvac_true,
        'month_ao_hvac_res_net': month_ao_hvac_res_net,
        'epoch_input_data': epoch_input_data,
        'month_idx': month_idx,
        'days_in_bc_scaled': days_in_bc_scaled,
        'low_consumption_ac_flag': low_consumption_ac_flag,
        'low_consumption_sh_flag': 0
    }

    residue_from_cooling_fp = remove_fp_hvac(info_carrier_fp_removal, column_index, epochs_per_hour,
                                             disagg_output_object, 'ac')

    epoch_ao_hvac_true[:, column_index.get('ac')] = epoch_ao_hvac_true[:, column_index.get('ac')] - residue_from_cooling_fp[:, 0]

    month_residue_from_cooling_fp = np.bincount(month_idx, residue_from_cooling_fp[:, 0])
    month_residue_from_cooling_fp = month_residue_from_cooling_fp / Cgbdisagg.WH_IN_1_KWH

    month_od_cooling = np.bincount(month_idx, epoch_ao_hvac_true[:, column_index['ac']])
    month_ao_hvac_res_net[:, column_index['ac']] = month_od_cooling / Cgbdisagg.WH_IN_1_KWH

    month_od_heating = np.bincount(month_idx, epoch_ao_hvac_true[:, column_index['sh']])
    month_ao_hvac_res_net[:, column_index['sh']] = month_od_heating / Cgbdisagg.WH_IN_1_KWH

    month_ao_hvac_res_net[:, std_residual_col] = month_ao_hvac_res_net[:, std_residual_col] + month_residue_from_cooling_fp

    # Adjust AO for no OD HVAC consumption
    disagg_output_object = suppress_ao_seasonal(month_ao_hvac_res_net, disagg_output_object,
                                                epoch_input_data,
                                                days_in_bc_scaled, month_idx, column_index, 'ac',
                                                low_consumption_ac_flag)

    month_ao_hvac_res_net = np.c_[month_ao_hvac_res_net, disagg_output_object['ao_seasonality']['cooling']]
    month_ao_hvac_res_net = np.c_[
        month_ao_hvac_res_net, month_ao_hvac_res_net[:, column_index['ac']] * Cgbdisagg.WH_IN_1_KWH]
    month_ao_hvac_res_net = np.c_[month_ao_hvac_res_net, disagg_output_object['ao_seasonality']['heating']]
    month_ao_hvac_res_net = np.c_[
        month_ao_hvac_res_net, month_ao_hvac_res_net[:, column_index['sh']] * Cgbdisagg.WH_IN_1_KWH]

    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, disagg_output_object['ao_seasonality']['epoch_cooling']]
    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, epoch_ao_hvac_true[:, column_index['ac']]]
    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, disagg_output_object['ao_seasonality']['epoch_heating']]
    epoch_ao_hvac_true = np.c_[epoch_ao_hvac_true, epoch_ao_hvac_true[:, column_index['sh']]]

    # Add AO HVAC into HVAC Columns
    month_ao_hvac_res_net[:, column_index['ac']] = \
        month_ao_hvac_res_net[:, column_index['ac']] * Cgbdisagg.WH_IN_1_KWH + disagg_output_object['ao_seasonality']['cooling']

    epoch_ao_hvac_true[:, column_index['ac']] = \
        epoch_ao_hvac_true[:, column_index['ac']] + disagg_output_object['ao_seasonality']['epoch_cooling']

    month_ao_hvac_res_net[:, column_index['sh']] = \
        month_ao_hvac_res_net[:, column_index['sh']] * Cgbdisagg.WH_IN_1_KWH + disagg_output_object['ao_seasonality']['heating']

    epoch_ao_hvac_true[:, column_index['sh']] = \
        epoch_ao_hvac_true[:, column_index['sh']] + disagg_output_object['ao_seasonality']['epoch_heating']

    return month_ao_hvac_res_net, epoch_ao_hvac_true, disagg_output_object
