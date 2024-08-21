"""
Author - Nisha Agarwal
Date - 3rd Jan 21
This file contain config values for calculating activity segments
"""

# No imports


def init_activity_segments_config():

    """
    Initialize config used for active segments calculation

    Returns:
        config                   (dict)         : active hours config dict
    """

    config = dict()

    # 'smoothing_window': window size used for smoothing average array,
    # 'levels_segment': segments with valid levels ,
    # 'top_levels_segment': mountain segment,
    # 'non_active_segments': non active segments,
    # 'active_segments': active segments,
    # 'morning_segment_multiplier': factor used for locating morning segments

    segments_config = {
        'smoothing_window': 3,
        'levels_segment': [1, 4, 5],
        'top_levels_segment': 2,
        'diff_multiplier': 0.8,
        'non_active_segments': [1],
        'active_segments': [2, 3, 4, 5],
        'morning_segment_multiplier': 0.6,
        'morning_start': 3,
        'morning_end': 9
    }

    config.update({
        "segments_config": segments_config
    })

    return config
