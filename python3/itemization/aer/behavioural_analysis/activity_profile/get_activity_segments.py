"""
Author - Nisha Agarwal
Date - 8th Oct 20
Calculate segments of activity - ["plain", "mountain", "plateau", "uphill", "downhill"]
"""

# Import python packages

import logging
import numpy as np
from datetime import datetime

# import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import get_index_array
from python3.itemization.aer.functions.itemization_utils import rolling_func

from python3.itemization.init_itemization_config import init_itemization_params

from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_profile_config import init_activity_profile_config
from python3.itemization.aer.behavioural_analysis.activity_profile.config.init_activity_segments_config import init_activity_segments_config


def get_segments_params(level_index, start_index, end_index, activity_seq_chucks, seq_config):

    """
    Calculate segments of activity - ["plain", "mountain", "plateau", "uphill", "downhill"]

    Parameters:
        level_index             (int)            : activity seq index for levels calculation
        start_index             (int)            : activity seq index for start calculation
        end_index               (int)            : activity seq index for end calculation
        activity_seq_chucks     (np.ndarray)     : activity sequences
        seq_config              (dict)           : sequence index config

    Returns:
        level                   (int)            : level of the segment
        start                   (int)            : start index of the segment
        end                     (int)            : end index of the segment
    """

    level = activity_seq_chucks[level_index, seq_config.get('mid_perc')]
    start = activity_seq_chucks[start_index, seq_config.get('start')]
    end = activity_seq_chucks[end_index, seq_config.get('end')]

    return level, start, end


def get_activity_segments(samples_per_hour, activity_curve, activity_curve_diff, activity_sequences, logger_pass):

    """
    Calculate segments of activity - ["plain", "mountain", "plateau", "uphill", "downhill"]

    Parameters:
        samples_per_hour           (int)            : samples in an hour
        activity_curve             (np.ndarray)     : living load activity profile
        activity_curve_diff        (float)          : diff in max and min of activity curve
        activity_sequences         (np.ndarray)     : labels of activity sequences of the user
        logger_pass                (dict)           : Contains the logger and the logging dictionary to be passed on

    Returns:
        activity_segments          (np.ndarray)     : array containing information for individual segments
        active_hours               (np.ndarray)     : active/nonactive mask array
    """

    # Initialize the logger

    logger_base = logger_pass.get('logger_base').getChild('get_activity_segments')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    t_activity_profile_start = datetime.now()

    config = init_activity_segments_config()
    seq_config = init_itemization_params().get('seq_config')

    activity_curve_derivative = activity_curve - np.roll(activity_curve, 1)

    activity_seq_chunks = find_seq(activity_sequences, activity_curve_derivative, activity_curve)

    seq_length = min(120, len(activity_seq_chunks))

    activity_sequences_labels = activity_seq_chunks[:, 0]

    interval = int(config.get('segments_config').get('smoothing_window') * samples_per_hour / 2)

    activity_curve_rolling_avg = rolling_func(activity_curve, interval)

    baseload_consumption = np.min(activity_curve_rolling_avg)

    # segment_types : [1, 2, 3, 4, 5]
    # segment_names : ["plain", "mountain", "plateau", "uphill", "downhill"]

    # initialize active hours
    active_hours = np.zeros(len(activity_curve))

    activity_segments = np.zeros((len(activity_curve), init_activity_profile_config(samples_per_hour).get('segments_config').get("total_keys")))

    index = 0

    count = 0

    while index < seq_length:

        variation = 0
        pattern = 0
        slope = 0

        index_copy = index

        # Calculate parameters for each segment type

        segment, level, start, end, index = \
            calculate_activity_segments_params(activity_sequences_labels, activity_curve, activity_seq_chunks, index, seq_length, seq_config)

        if not segment:
            continue

        index_array = get_index_array(start, end, len(activity_curve))

        # post process steps to update activity level

        if segment in config.get('segments_config').get('levels_segment'):
            level = np.mean(activity_curve[index_array])

        if segment == config.get('segments_config').get('top_levels_segment'):
            array = activity_curve[index_array]
            limit = np.min(array) + config.get('segments_config').get('diff_multiplier')*(np.max(array) - np.min(array))
            array = array[array > limit]
            level = np.mean(array)

        # check if the current segment is morning segment

        location = (start + end) / 2

        seq_params = [start, end, location, level]

        morning = get_morning_flag(seq_params, baseload_consumption, activity_curve_diff, samples_per_hour, config)

        # mark all non constant segments as active segments

        if segment in config.get('segments_config').get('non_active_segments'):
            variation = np.max(activity_curve[index_array]) - np.min(activity_curve[index_array])
            pattern = int((activity_seq_chunks[(index_copy-1) % seq_length, 0] == -1) and
                          (activity_seq_chunks[(index_copy+1) % seq_length, 0] == 1))
            pattern = int(pattern or int((activity_seq_chunks[(index_copy-1) % seq_length, 0] == -1) and
                                         (activity_seq_chunks[(index_copy+1) % seq_length, 0] == -1)))
            slope = activity_curve[index_array[-1]] - activity_curve[index_array[0]]

        else:
            active_hours[index_array] = 1

        activity_segments[count] = np.array([segment, level, start, end, morning, int(location), pattern, slope, variation])
        count = count + 1

    activity_segments = activity_segments[~np.all(activity_segments == 0, axis=1)]

    logger.debug("Calculated activity segments")

    logger.info("Segments found | %s", activity_segments[:, 0])

    t_activity_profile_end = datetime.now()

    logger.info("Calculation of actvity segments took | %.3f s",
                get_time_diff(t_activity_profile_start, t_activity_profile_end))

    return activity_segments, active_hours


def get_morning_flag(seq_params, baseload_consumption, activity_curve_diff, samples_per_hour, config):

    """
    Check whether the particular segments lies in the morning region

    Parameters:
        seq_params              (list)            : list of start, end, location and level of the sequence
        baseload_consumption    (flaot)           : minimum level of the activity profile
        activity_curve_diff     (float)           : range of activity curve
        samples_per_hour        (int)             : number of samples in an hour
        config                  (dict)            : sequence index config

    Returns:
        morning                 (bool)            : If true, the segment belongs to the morning tod

    """

    start = seq_params[0]
    end = seq_params[1]
    location = seq_params[2]
    level = seq_params[3]

    morning1 = start <= end and location < config.get('segments_config').get('morning_end') * samples_per_hour
    morning1 = morning1 and location > config.get('segments_config').get('morning_start') * samples_per_hour
    morning1 = morning1 and level < baseload_consumption + config.get('segments_config').get(
        'morning_segment_multiplier') * activity_curve_diff

    morning2 = start > end and end / 2 < config.get('segments_config').get('morning_end') * samples_per_hour
    morning2 = morning2 and end / 2 > config.get('segments_config').get('morning_start') * samples_per_hour
    morning2 = morning2 and level < baseload_consumption + config.get('segments_config').get(
        'morning_segment_multiplier') * activity_curve_diff

    morning = morning1 or morning2

    return morning


def get_mountain_params(activity_curve, activity_seq_chunks, index, seq_length, seq_config):

    """
    Calculate mountain type segment parameters

    Parameters:
        activity_curve              (np.ndarray)    : activity curve of the user
        activity_seq_chunks         (np.ndarray)    : chunks of activity seqs (1, 0, -1)
        index                       (int)           : index of the activity seq chunk
        seq_length                  (int)           : length of current activity seq chunk
        seq_config                  (dict)          : dict containing seq config params

    Returns:
        segment                     (int)           : type of segment
        level                       (int)           : level of the segment
        start                       (int)           : start of the segment
        end                         (int)           : end of the segment
        index                       (int)           : updated index of the activity seq chunk

    """

    segment = 2

    level, start, end = get_segments_params((index + 1) % seq_length, index % seq_length,
                                            (index + 1) % seq_length, activity_seq_chunks, seq_config)

    level = (activity_curve[int(activity_seq_chunks[index % seq_length, seq_config.get('end')])] +
             activity_curve[int(activity_seq_chunks[(index + 1) % seq_length, seq_config.get('start')])]) / 2

    index = index + 2

    return segment, level, start, end, index


def get_plateau_params(activity_seq_chunks, index, seq_length, seq_config):

    """
    Calculate plateau type segment parameters

    Parameters:
        activity_curve              (np.ndarray)    : activity curve of the user
        activity_seq_chunks         (np.ndarray)    : chunks of activity seqs (1, 0, -1)
        index                       (int)           : index of the activity seq chunk
        seq_length                  (int)           : length of current activity seq chunk
        seq_config                  (dict)          : dict containing seq config params

    Returns:
        segment                     (int)           : type of segment
        level                       (int)           : level of the segment
        start                       (int)           : start of the segment
        end                         (int)           : end of the segment
        index                       (int)           : updated index of the activity seq chunk
    """

    level, start, end = get_segments_params((index + 1) % seq_length, index % seq_length,
                                            (index + 2) % seq_length, activity_seq_chunks, seq_config)

    segment = 3

    index = index + 3

    return segment, level, start, end, index


def get_plain_params(activity_seq_chunks, index, seq_length, seq_config):

    """
    Calculate plain type segment parameters

    Parameters:
        activity_curve              (np.ndarray)    : activity curve of the user
        activity_seq_chunks         (np.ndarray)    : chunks of activity seqs (1, 0, -1)
        index                       (int)           : index of the activity seq chunk
        seq_length                  (int)           : length of current activity seq chunk
        seq_config                  (dict)          : dict containing seq config params

    Returns:
        segment                     (int)           : type of segment
        level                       (int)           : level of the segment
        start                       (int)           : start of the segment
        end                         (int)           : end of the segment
        index                       (int)           : updated index of the activity seq chunk

    """

    segment = 1

    level, start, end = get_segments_params(index % seq_length, index % seq_length,
                                            index % seq_length, activity_seq_chunks, seq_config)
    index = index + 1

    return segment, level, start, end, index


def get_uphill_params(activity_seq_chunks, index, seq_length, seq_config):

    """
    Calculate uphill type segment parameters

    Parameters:
        activity_curve              (np.ndarray)    : activity curve of the user
        activity_seq_chunks         (np.ndarray)    : chunks of activity seqs (1, 0, -1)
        index                       (int)           : index of the activity seq chunk
        seq_length                  (int)           : length of current activity seq chunk
        seq_config                  (dict)          : dict containing seq config params

    Returns:
        segment                     (int)           : type of segment
        level                       (int)           : level of the segment
        start                       (int)           : start of the segment
        end                         (int)           : end of the segment
        index                       (int)           : updated index of the activity seq chunk
    """

    segment = 4

    level, start, end = get_segments_params(index % seq_length, index % seq_length,
                                            index % seq_length, activity_seq_chunks, seq_config)

    index = index + 1

    return segment, level, start, end, index


def get_downhill_params(activity_seq_chunks, index, seq_length, seq_config):

    """
    Calculate downhill type segment parameters

    Parameters:
        activity_curve              (np.ndarray)    : activity curve of the user
        activity_seq_chunks         (np.ndarray)    : chunks of activity seqs (1, 0, -1)
        index                       (int)           : index of the activity seq chunk
        seq_length                  (int)           : length of current activity seq chunk
        seq_config                  (dict)          : dict containing seq config params

    Returns:
        segment                     (int)           : type of segment
        level                       (int)           : level of the segment
        start                       (int)           : start of the segment
        end                         (int)           : end of the segment
        index                       (int)           : updated index of the activity seq chunk
    """

    segment = 5

    level, start, end = get_segments_params(index % seq_length, index % seq_length,
                                            index % seq_length, activity_seq_chunks, seq_config)

    index = index + 1

    return segment, level, start, end, index


def calculate_activity_segments_params(activity_sequences_labels, activity_curve, activity_seq_chunks, index, seq_length, seq_config):

    """
    Calculate attributes for each activity segments

    Parameters:
        activity_sequences_labels   (np.ndarray)    : labels of all the activity seq chunks
        activity_curve              (np.ndarray)    : activity curve of the user
        activity_seq_chunks         (np.ndarray)    : chunks of activity seqs (1, 0, -1)
        index                       (int)           : index of the activity seq chunk
        seq_length                  (int)           : length of current activity seq chunk
        seq_config                  (dict)          : dict containing seq config params

    Returns:
        segment                     (int)           : type of segment
        level                       (int)           : level of the segment
        start                       (int)           : start of the segment
        end                         (int)           : end of the segment
        index                       (int)           : updated index of the activity seq chunk
    """

    if activity_sequences_labels[index] == 1:

        # Calculate segment type and segment attributes

        # mountain type

        if activity_sequences_labels[(index + 1) % seq_length] == -1:

            segment, level, start, end, index = get_mountain_params(activity_curve, activity_seq_chunks, index,
                                                                    seq_length, seq_config)

        # Plateau type

        elif activity_sequences_labels[(index + 1) % seq_length] == 0 and \
                activity_sequences_labels[(index + 2) % seq_length] == -1:

            segment, level, start, end, index = get_plateau_params(activity_seq_chunks, index, seq_length, seq_config)

        # Uphill type

        else:

            segment, level, start, end, index = get_uphill_params(activity_seq_chunks, index, seq_length, seq_config)

        # Downhill type

    elif activity_sequences_labels[index] == -1:

        if activity_sequences_labels[(index - 1) % len(activity_seq_chunks)] == 1:
            index = index + 1
            return 0, 0, 0, 0, index

        else:

            segment, level, start, end, index = get_downhill_params(activity_seq_chunks, index, seq_length, seq_config)

        # plain type

    else:

        segment, level, start, end, index = get_plain_params(activity_seq_chunks, index, seq_length, seq_config)

    return segment, level, start, end, index
