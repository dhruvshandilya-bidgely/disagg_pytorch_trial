"""
Author - Sahana M
Date - 9/3/2021
Extracts and creates hsm for the seasonal water heater module
"""

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg


def extract_hsm(item_input_object, global_config):
    """Utility to extract hsm"""

    # noinspection PyBroadException
    try:
        hsm_dic = item_input_object.get('appliances_hsm')
        hsm_in = hsm_dic.get('wh')
    except KeyError:
        hsm_in = None

    hsm_fail = (hsm_in is None or len(hsm_in) == 0 or len(hsm_in.get('attributes')) == 0) and \
               (global_config.get("disagg_mode") == "mtd")

    return hsm_in, hsm_fail


def create_hsm(debug):
    """Creating hsm attributes"""

    if debug.get('swh_hld') == 1:
        attributes = {
            'swh_low_amp': debug.get('final_low_amp'),
            'swh_high_amp': debug.get('final_high_amp'),
            'swh_hld': debug.get('swh_hld'),
            'swh_start_time_zones': debug.get('start_time_zones'),
            'swh_end_time_zones': debug.get('end_time_zones'),
            'swh_band_scores': debug.get('time_zone_band_scores'),
            'local_max_idx': debug.get('local_max_idx'),
        }
    else:
        attributes = {
            'swh_hld': debug.get('swh_hld')
        }

    return attributes


def make_hsm_from_debug(debug, logger):
    """
    This function is used to make hsm for seasonal wh
    Parameters:
        debug           (dict)                  : The dictionary containing all module level output
        logger          (logger)                : The logger object to write logs

    Returns:
        debug           (dict)                  : Updated debug object
    """

    # If the mode is historical/incremental, make hsm

    if debug['make_hsm']:
        # Extract the relevant values from debug dict to create hsm
        input_data = debug.get('input_data')
        wh_hsm = {
            'timestamp': input_data[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': create_hsm(debug)
        }

        # Saving new HSM to the debug object

        debug['hsm_in']['timestamp'] = wh_hsm.get('timestamp')
        if debug.get('hsm_in').get('attributes'):
            debug['hsm_in']['attributes'].update(wh_hsm.get('attributes'))
        else:
            debug['hsm_in']['attributes'] = wh_hsm.get('attributes')

        logger.info('Writing new HSM for the run | ')
    else:
        logger.info('Not writing HSM for this mode | ')

    return debug


def update_hsm(pipeline_output_object, item_output_object, debug):
    """
    This function is used to update the WH hsm in disagg output and hybrid output object
    Parameters:
        pipeline_output_object        (dict)      : Dictionary containing all disagg outputs
        item_output_object         (dict)      : Dictionary containing all hybrid inputs
        debug                       (dict)      : Debug object

    Returns:
        pipeline_output_object        (dict)      : Dictionary containing all disagg outputs
        item_output_object         (dict)      : Dictionary containing all hybrid inputs
    """

    # If WH hsm already exists then update the hsm

    if pipeline_output_object.get('created_hsm').get('wh') is not None:
        hsm = debug.get('hsm_in')
        item_output_object['created_hsm']['wh'] = hsm
        pipeline_output_object['created_hsm']['wh']['attributes'].update(hsm.get('attributes'))
    else:
        hsm = debug.get('hsm_in')
        item_output_object['created_hsm']['wh'] = hsm
        pipeline_output_object['created_hsm']['wh'] = hsm

    return pipeline_output_object, item_output_object
