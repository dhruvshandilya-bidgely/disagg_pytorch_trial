
"""
Author - Nisha Agarwal
Date - 10th Mar 2023
Initialize general config required in hybrid v2 pipeline
"""


def get_hybrid_v2_generic_config(samples_per_hour=1):

    """
    Initialize general config required in hybrid v2 pipeline

    Parameters:
        samples_per_hour       (int)            : samples in an hour
    Returns:
        config                 (dict)           : Dict containing all laundry detection related parameters
    """

    config = dict({

        'min_bc_required_for_consistency_check': 4,
        'min_points_required': 5*samples_per_hour,
        'cons_thres': 3000,
        'min_days_required': 3,
        'min_li_points_required': 3*samples_per_hour,
        'cons_thres_for_li': 2000,
        'min_days_required_for_max_limit': 5,
        'cook_limit': 400,
        'ld_limit': 400,
        'ent_limit': 400,

    })

    return config
