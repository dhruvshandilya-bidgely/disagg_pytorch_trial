"""
Author - Mayank Sharan
Date - 12/03/19
We try to reconstruct signal from a rounded one for TEPCO to help disagg run properly
"""

# Import python packages

import copy
import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.maths_utils.find_seq import find_seq
from python3.config.pilot_constants import PilotConstants


def reconstruct_signal_section(input_cons, rounding_value):

    """Utility to reconstruct signal for a given section"""

    input_cons = copy.deepcopy(input_cons)

    # Get plateaus in the data

    seq_of_plateaus = find_seq(input_cons, min_seq_length=0)
    seq_of_plateaus = seq_of_plateaus.astype(int)

    plateau_height_diff = np.diff(seq_of_plateaus[:, 0])
    non_negative_idx = np.where(plateau_height_diff > 0)[0]

    for idx in non_negative_idx:

        delta_value = np.round(rounding_value / (seq_of_plateaus[idx, 3] + 1), 1)

        input_cons[seq_of_plateaus[idx, 1]: seq_of_plateaus[idx, 2] + 1] += delta_value
        total_delta_value = delta_value * seq_of_plateaus[idx, 3]
        input_cons[seq_of_plateaus[idx, 2] + 1] -= total_delta_value

    return input_cons


def reconstruct_rounded_signal(disagg_input_object, logger):

    """
    Parameters:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
        logger              (logger)            : Logger object to write logging statements

    Returns:
        disagg_input_object (dict)              : Contains all inputs required to run the pipeline
    """

    # Extract variables to make the decision on

    pilot_id = disagg_input_object.get('config').get('pilot_id')

    disagg_input_object['input_data_without_rounding_correction'] = copy.deepcopy(disagg_input_object.get('input_data'))

    # Do not perform reconstruction in not a Japan pilot

    if not(pilot_id in PilotConstants.HVAC_JAPAN_PILOTS):
        return disagg_input_object

    # Initialize variables to be used
    # IMPORTANT: Any new entries to the below list should maintain the list's overall ascending order.
    possible_rounding_values = [1000, 600, 100, 10]

    input_data = copy.deepcopy(disagg_input_object.get('input_data'))
    rounded_cons = input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    reconstructed_cons = copy.deepcopy(rounded_cons)

    # Detect the rounding values part of the data, Split the data if there are multiple rounding values

    cons_diff = np.diff(rounded_cons)
    rounding_value = np.full(shape=(cons_diff.shape[0],), fill_value=-1)

    # For each of the possible rounding values find sequences in input data where it applies
    # We do this in decreasing order of rounding values since a 100 W rounding also satisfies conditions of 10 W
    # rounding we want the 100 W rounding to be identified first correctly

    for idx in range(len(possible_rounding_values)):

        # Get the current rounding value being considered

        rounding_value_considered = possible_rounding_values[idx]

        avail_idx = rounding_value == -1

        if np.sum(avail_idx) < 30:
            break

        mod_val = np.mod(cons_diff, rounding_value_considered)

        # Get consecutive sequences in the data, merge consecutive sequences with same difference

        rounding_sequences = find_seq(mod_val, min_seq_length=30)

        for idx_round_seq in range(rounding_sequences.shape[0] - 1):

            if rounding_sequences[idx_round_seq, 0] == rounding_sequences[idx_round_seq + 1, 0]:

                rounding_sequences[idx_round_seq + 1, 1] = rounding_sequences[idx_round_seq, 1]
                rounding_sequences[idx_round_seq + 1, 3] = rounding_sequences[idx_round_seq + 1, 2] - \
                    rounding_sequences[idx_round_seq + 1, 1] + 1

                rounding_sequences[idx_round_seq, 0] = np.nan

        if rounding_sequences.shape[0] > 0:
            rounding_sequences = rounding_sequences[np.logical_not(np.isnan(rounding_sequences[:, 0])), :]
            rounding_sequences = rounding_sequences[rounding_sequences[:, 0] == 0, :]

        # Get indices at which we have to do the reconstruction and subsequently perform reconstruction

        rounded_val_bool = np.full(shape=(len(avail_idx,)), fill_value=False)
        rounding_sequences = rounding_sequences.astype(int)

        for idx_round_seq in range(rounding_sequences.shape[0]):
            rounded_val_bool[rounding_sequences[idx_round_seq, 1]: rounding_sequences[idx_round_seq, 2] + 1] = True

        idx_to_reconstruct = np.logical_and(avail_idx, rounded_val_bool)
        seq_to_reconstruct = find_seq(idx_to_reconstruct)
        seq_to_reconstruct = seq_to_reconstruct.astype(int)
        seq_to_reconstruct = seq_to_reconstruct[seq_to_reconstruct[:, 0] == 1, :]

        for idx_seq_reconstruct in range(seq_to_reconstruct.shape[0]):

            idx_start = seq_to_reconstruct[idx_seq_reconstruct, 1]
            idx_end = seq_to_reconstruct[idx_seq_reconstruct, 2] + 2

            reconstructed_cons[idx_start: idx_end] = reconstruct_signal_section(reconstructed_cons[idx_start: idx_end],
                                                                                rounding_value_considered)

            rounding_value[idx_start: idx_end] = rounding_value_considered

        if seq_to_reconstruct.shape[0] > 0:
            logger.info('Rounding reconstructed | Rounding value : %d, Number of points : %d',
                        rounding_value_considered, np.sum(seq_to_reconstruct[:, 3] + 1))

    # Reassign the consumption to the reconstructed values

    input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX] = reconstructed_cons
    disagg_input_object['input_data'] = input_data

    return disagg_input_object
