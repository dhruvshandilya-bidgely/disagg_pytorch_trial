"""
Author - Sahana M
Date - 30th March 2021
Classes for different constants and config parameters used throughout the pipeline
"""


class GlobalConfigParams:

    """
    Contains parameters and constants required to initialise the global config
    """

    # This dictionary contains the different data types available as input as per sampling frequency
    data_type = {
        'ami': 'ami',
        'nsm': 'nsm',
        'amr': 'nsm'
    }

    # This dictionary contains the different fuel types available
    fuel_type = {
        'electric': 'electric',
        'gas': 'gas'
    }

    # This dictionary contains the different user types
    user_type = {
        'residential': 'residential',
        'smb': 'smb'
    }

    # This dictionary contains the different sequences of pipeline run
    pipeline_seq = {
        'disagg': ['disagg'],
        'disagg_analytics': ['disagg', 'analytics'],
        'disagg_itemization_analytics': ['disagg', 'itemization', 'analytics']
    }

    # This dictionary contains all the module sequences to be run for their corresponding pipelines

    disagg_aer_seq = ['solar', 'va', 'ao', 'ref', 'pp', 'wh', 'ev', 'va', 'hvac', 'li']

    disagg_aes_seq = ['ao_smb', 'li_smb', 'work_hours', 'hvac_smb']

    analytics_aer_seq = ['life', 'so_propensity', 'ev_propensity', 'hvac_ineff']
    analytics_aes_seq = ['life_smb']

    itemization_seq = ["ref", "li", "wh", "cook", "ent", "ld"]

    disagg_postprocess_enabled_app = ['pp', 'ev', 'hvac', 'wh', 'li', 'others']

    residential_true_disagg_app_list = ['pp', 'ev', 'hvac', 'wh', 'ao', 'ref', 'li']

    smb_true_disagg_app_list = ['ao_smb', 'op', 'x-ao', 'hvac_smb']
    item_aer_seq = ['pp', 'ev', 'hvac', 'wh', 'li', 'ao', 'ref', 'cook', 'ld', 'ent']

    hybrid_v2_additional_app = ['cook', 'ent', 'ld']

