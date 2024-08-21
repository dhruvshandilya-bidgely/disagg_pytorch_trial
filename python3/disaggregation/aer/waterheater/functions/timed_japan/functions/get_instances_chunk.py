"""
Author - Sahana M
Date - 07/06/2021
Get all the insatnces chunks
"""

# Import python packages
import scipy
import numpy as np

# Import packages from within the project
from python3.disaggregation.aer.waterheater.functions.timed_japan.timed_wh_config import chunk_indexes
from python3.disaggregation.aer.waterheater.functions.timed_japan.functions.twh_maths_utils import get_start_end_idx


def merge_single_points(max_consistencies_1d, single_zeros, single_point_consistency):
    """
    This function considers merging a single point squashed between 2 high consistent boxes
    Parameters:
        max_consistencies_1d            (np.ndarray) : numpy array containing the time level consistencies
        single_zeros                    (np.ndarray) : single zeros indexes
        single_point_consistency        (np.ndarray) : Single point consistency threshold
    Returns:
        max_consistencies_1d            (np.ndarray) : numpy array containing the time level consistencies
    """

    # For every single point look for combining them

    for i in single_zeros:

        if i == 0 and max_consistencies_1d[i + 1] >= single_point_consistency:
            max_consistencies_1d[i] = max_consistencies_1d[i + 1]

        elif i == len(max_consistencies_1d) - 1 and max_consistencies_1d[i - 1] >= single_point_consistency:
            max_consistencies_1d[i] = max_consistencies_1d[i - 1]

        elif (i > 0 and i != len(max_consistencies_1d) - 1) and \
                (max_consistencies_1d[i - 1] + max_consistencies_1d[i + 1]) / 2 >= single_point_consistency:
            max_consistencies_1d[i] = (max_consistencies_1d[i - 1] + max_consistencies_1d[i + 1]) / 2

    return max_consistencies_1d


def get_percentile_amplitude(amplitude_blocks_2d, min_amplitude_percentile):
    """
    This function is used to get the percentile amplitudes at each time of the day
    Parameters:
        amplitude_blocks_2d             (np.ndarray)    : numpy array containing the different ranges of amplitude blocks
                                                          present at each time of day
        min_amplitude_percentile        (int)           : Percentile amplitude capping at each time of day

    Returns:
        percentile_amplitudes           (np.ndarray)    : Array containing the probable amplitude at each time of day
    """

    # For each time do the day get the corresponding amplitude

    percentile_amplitudes = []
    for i in range(amplitude_blocks_2d.shape[1]):
        curr_amp = amplitude_blocks_2d[amplitude_blocks_2d[:, i] > 0, i]
        if len(curr_amp):
            percentile_amplitudes.append(np.percentile(curr_amp, q=min_amplitude_percentile))
        else:
            percentile_amplitudes.append(0)
    percentile_amplitudes = np.array(percentile_amplitudes)

    return percentile_amplitudes


def get_instances_chunks(start_day, overall_idx, energy_blocks_2d_toi, amplitude_range_arr, debug, wh_config):
    """
    This function is used to get the best probable timed WH instances in the given given data
    Parameters:
        start_day                   (int)       :       Start of the day index
        overall_idx                 (int)       :       The index of the start_day
        energy_blocks_2d_toi        (np.ndarray):       Contains the energy blocks within the given data
        amplitude_range_arr         (np.ndarray):       Contains the amplitude ranges of each row of energy blocks
        debug                       (dict)      :       Step wise algorithm output
        wh_config                   (dict)      :       Contains WH configurations

    Returns:
        chunk_data                  (np.ndarray):       Contains the probable timed WH instances with their features
        final_consistencies         (np.ndarray):       Contains the refined consistencies of the given data
        final_amplitudes            (np.ndarray):       Contains the refined amplitudes of the given data
    """

    # Initialise the necessary data

    cols = debug.get('cols')
    factor = debug.get('factor')
    chunk_data = np.array([])
    edge_amp = wh_config.get('edge_amp')
    centered_amp = wh_config.get('centered_amp')
    final_amplitudes = np.full(cols, fill_value=0.0)
    minimum_consistency = wh_config.get('min_consistency')
    final_consistencies = np.full(cols, fill_value=0.0)
    good_chunk_probability = wh_config.get('good_chunk_probability')
    max_num_chunks_allowed = wh_config.get('max_num_chunks_allowed')
    base_num_chunks_allowed = wh_config.get('base_num_chunks_allowed')
    single_point_consistency = wh_config.get('single_point_consistency')
    min_amplitude_percentile = wh_config.get('min_amplitude_percentile')

    # Get all the instances which has the minimum_consistency (consistency == number of days in the given data that matches
    # the amplitude of the data point)

    probable_instances_bool = energy_blocks_2d_toi > minimum_consistency

    # Get all the time stamps where instances are present

    probable_instances_bool = np.sum(probable_instances_bool, axis=0) > 0

    # Take care of single empty gaps between 2 blocks

    left_shift_arr = np.full_like(probable_instances_bool, fill_value=False)
    right_shift_arr = np.full_like(probable_instances_bool, fill_value=False)
    left_shift_arr[1:] = probable_instances_bool[:-1]
    right_shift_arr[:-1] = probable_instances_bool[1:]
    probable_instances_bool |= left_shift_arr | right_shift_arr

    if len(energy_blocks_2d_toi):

        # Get the maximum instances at each time stamp

        max_consistencies_1d = np.max(energy_blocks_2d_toi, axis=0) * probable_instances_bool

        # If a single box is squashed between 2 high consistent boxes, then consider merging it

        single_points = max_consistencies_1d == 0
        single_point_idx_diff = np.diff(np.r_[0, single_points.astype(int), 0])
        single_start_idx = np.where(single_point_idx_diff[:-1] > 0)[0]
        single_end_idx = np.where(single_point_idx_diff[1:] < 0)[0]

        single_zeros = single_start_idx[(single_start_idx - single_end_idx) == 0]

        max_consistencies_1d = merge_single_points(max_consistencies_1d, single_zeros, single_point_consistency)

        # Update the probable instances bool
        probable_instances_bool = max_consistencies_1d > 0

        # Calculate the mean amplitudes of each energy block

        mean_amplitudes = ((amplitude_range_arr[:, 0] + amplitude_range_arr[:, 1])/2).reshape(-1, 1)

        # Get the percentile amplitude at each time stamp

        amplitude_blocks_2d = (energy_blocks_2d_toi > minimum_consistency)*mean_amplitudes
        percentile_amplitudes = get_percentile_amplitude(amplitude_blocks_2d, 98)

        # Calculate the start & end of each chunk of data

        box_start_idx, box_end_idx = get_start_end_idx(probable_instances_bool, end_idx_exclusive=True)

        # For each chunk obtained identify the features - consistency, amplitude, duration

        if np.sum(probable_instances_bool) > 0:

            chunk_data = np.full(shape=(len(chunk_indexes)), fill_value=0)

            count = 1
            for index in range(len(box_start_idx)):
                start_time = int(box_start_idx[index])
                end_time = int(box_end_idx[index]+1)

                # Calculate the features

                chunk_max_consistency = np.max(max_consistencies_1d[start_time:end_time])
                chunk_amplitude = np.percentile(percentile_amplitudes[start_time: end_time], q=min_amplitude_percentile)
                chunk_duration = (end_time - start_time)/factor

                temp = np.asarray([overall_idx, start_day, count, start_time, end_time, chunk_max_consistency, chunk_amplitude,
                                   chunk_duration, 0, 0, 0, 0])

                chunk_data = np.vstack((chunk_data, temp))
                count += 1

            chunk_data = chunk_data[1:, :]

            # If the number of chunks obtained is > 4 then try discarding the erroneous chunks based on amplitude

            if len(chunk_data) >= base_num_chunks_allowed:
                temp_chunk_data_amp = chunk_data[:, chunk_indexes['amplitude_idx']] * factor
                temp_chunk_data_amp -= edge_amp
                temp_chunk_data_amp = temp_chunk_data_amp / centered_amp
                probabilities = scipy.stats.norm.pdf(temp_chunk_data_amp)
                probabilities = (0.25*np.fmin(chunk_data[:, 5] / 30, 1) + 0.75*probabilities)
                good_chunks = probabilities >= good_chunk_probability
                chunk_data = chunk_data[good_chunks, :]

            # If the number of chunks obtains is still > 5, keep the top 5 chunks based on higher consistency

            if len(chunk_data) > max_num_chunks_allowed:
                chunk_data = chunk_data[np.argsort(chunk_data[:, chunk_indexes['consistency_idx']])][-max_num_chunks_allowed:]
                chunk_data = chunk_data[np.argsort(chunk_data[:, chunk_indexes['chunk_id']])]

            # Save all the chunks info in separate arrays for future reference

            final_aoi_bool = np.full_like(probable_instances_bool, fill_value=False)
            count = 1
            for i in range(chunk_data.shape[0]):
                start_time = int(chunk_data[i, chunk_indexes['chunk_start']])
                end_time = int(chunk_data[i, chunk_indexes['chunk_end']] + 1)
                final_aoi_bool[start_time:end_time] = True
                final_consistencies[start_time: end_time] = max_consistencies_1d[start_time: end_time]
                final_amplitudes[start_time: end_time] = percentile_amplitudes[start_time: end_time]
                chunk_data[i, chunk_indexes['chunk_id']] = count
                count += 1

    return chunk_data, final_consistencies, final_amplitudes
