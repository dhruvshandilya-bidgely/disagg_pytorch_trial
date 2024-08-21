"""
Author - Nikhil Singh Chauhan
Date - 16/10/19
Function to initialize the debug object that contains output of all intermediate algo steps
"""

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg


def initialize_debug(global_config, hsm_in, out_bill_cycles, models):
    """
    Parameters:
        global_config           (dict)          : Global config params
        hsm_in                  (dict)          : Input hsm
        out_bill_cycles         (np.ndarray)    : Bill cycles for which output to be written
        models                  (dict)          : Model objects relevant to EV

    Returns:
        debug                   (dict)          : Algorithm intermediate steps output
    """

    # Get the run mode from config

    disagg_mode = global_config.get('disagg_mode')

    # Based on run mode decide if hsm to be used in algo

    use_hsm = True if disagg_mode == 'mtd' else False

    # Based on run mode decide if hsm to be created in algo

    make_hsm = False if disagg_mode == 'mtd' else True

    # If HSM is none

    if hsm_in is None:
        hsm_in = {}

    # Create the sampling rate factor

    factor = Cgbdisagg.SEC_IN_HOUR // global_config.get('sampling_rate')

    # Initialize debug object to store every step output
    # 'hsm_in'              : Input hsm
    # 'use_hsm'             : Whether to use HSM or not
    # 'make_hsm'            : Whether to push new HSM or not
    # 'models'              : Models required in the module
    # 'out_bill_cycles'     : The bill cycles to be updated
    # 'l1'                  : Debug object for L1 chargers

    if disagg_mode is not None:
        # If valid disagg mode

        debug = {
            'factor': factor,
            'use_hsm': use_hsm,
            'make_hsm': make_hsm,
            'disagg_mode': disagg_mode,
            'hsm_in': hsm_in.get('attributes'),
            'out_bill_cycles': out_bill_cycles,
            'models': models,
            'l1': {},
        }
    else:
        # If invalid disagg mode, return empty

        debug = {}

    return debug
