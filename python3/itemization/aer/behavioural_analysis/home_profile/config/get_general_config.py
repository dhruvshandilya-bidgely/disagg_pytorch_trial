
"""
Author - Nisha Agarwal
Date - 3rd Feb 21
Config file for home profile calculation
"""


def get_general_config():

    """
       Initialize config dict containing generic parameters config values for weekday weekend profile calculation

       Returns:
           config               (dict)              : prepared config dictionary
    """

    config = dict()

    config.update({
        "non_clean_day_label": -1,
        "weekends": [7, 1],
        "weekdays": [2, 3, 4, 5, 6],
        'min_days_req_to_prepare_avg_based_prof': 10
    })

    return config
