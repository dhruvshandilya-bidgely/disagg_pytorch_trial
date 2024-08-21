"""
Author - Mayank Sharan
Date - 9th Jun 2020
Returns the disagg mode mapping for BE disagg modes
"""

# No imports here


def get_disagg_mode(be_disagg_mode):

    """
    Return the pipeline mode mapped to a given BE pipeline mode

    Parameters:
        be_disagg_mode      (string)            : The be pipeline mode we need a disagg mode for

    Returns:
        disagg_mode         (string)            : The pipeline mode corresponding to the BE pipeline mode
    """

    disagg_mode_map = {
        'HISTORICAL': 'historical',
        'SUPERVISED_DURATION_RERUN': 'historical',
        'COMPLETE_BILLING_CYCLE': 'incremental',
        'ONCE_IN_BILLING_CYCLE': 'mtd',
        'MTD': 'mtd',
    }

    return disagg_mode_map.get(str.upper(be_disagg_mode))
