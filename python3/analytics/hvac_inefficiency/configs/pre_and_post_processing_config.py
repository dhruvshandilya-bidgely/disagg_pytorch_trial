"""
Author  -   Anand Kumar Singh
Date    -   22th Feb 2021
This file Pre processing and Post Processing Configs
"""


def get_pre_processing_config():
    """
        This function returns pre processing config

        Returns:
            pre_processing_config                    (dict)          Pre processing config
    """

    pre_processing_config = {
        'hvac_days_threshold': 45,
        'hvac_hours_threshold': 450
    }

    return pre_processing_config


def get_post_processing_config():
    """
        This function returns post processing config

        Returns:
            post_processing_config                    (dict)          Pre processing config
    """

    post_processing_config = {
        'min_clusters_limit': 2,
        'fcc_15_min_limit': 350,
        'fcc_30_min_limit': 400,
        'minimum_hours_amplitude': 400,
        'minimum_hours_of_hvac': 4,
        'complete_mask': False
    }

    return post_processing_config
