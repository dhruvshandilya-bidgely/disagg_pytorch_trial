
"""
Author - Nisha Agarwal
Date - 7th Sep 2022
Master file for updating appliance profile
"""

# import functions from within the project

from python3.config.pilot_constants import PilotConstants

from python3.config.pipeline_constants import GlobalConfigParams

from python3.itemization.prepare_results.update_appliance_profile_for_stat_app import update_ld_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_stat_app import update_ent_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_stat_app import update_cook_appliance_profile

from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_pp_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_ev_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_wh_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_stat_app import update_li_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_ao_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_ref_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_deleted_wh_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_cooling_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_heating_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_added_ev_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_added_pp_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_added_wh_appliance_profile
from python3.itemization.prepare_results.update_appliance_profile_for_disagg_app import update_removed_pp_appliance_profile


def update_appliance_profile_based_on_disagg_postprocessing(item_input_object, item_output_object, run_successful, logger_itemization):

    """
    updates appliance profile based on disagg postprocessing

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        run_successful            (bool)      : true if hybrid v2 run was successful
        logger_itemization        (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    itemization_successful = item_output_object.get("final_itemization") is not None and \
                             item_output_object.get('inference_engine_dict') is not None

    stat_app_list = item_input_object.get("item_input_params").get("backup_app")

    target_app = GlobalConfigParams.disagg_postprocess_enabled_app

    if item_input_object.get('item_input_params').get('run_hybrid_v2_flag') and run_successful:
        target_app = GlobalConfigParams.item_aer_seq

    non_mtd_mode = item_input_object.get('config').get('disagg_mode') != 'mtd'

    prepare_results = itemization_successful and non_mtd_mode

    if not prepare_results:
        return item_output_object

    if 'pp' in target_app and ('pp' not in stat_app_list):
        item_output_object = update_pp_appliance_profile(item_input_object, item_output_object, logger_itemization)
        item_output_object = update_removed_pp_appliance_profile(item_input_object, item_output_object, logger_itemization)
        item_output_object = update_added_pp_appliance_profile(item_input_object, item_output_object, logger_itemization)

    if 'ev' in target_app and ('ev' not in stat_app_list):
        item_output_object = update_ev_appliance_profile(item_input_object, item_output_object, logger_itemization)
        item_output_object = update_added_ev_appliance_profile(item_input_object, item_output_object, logger_itemization)

    if 'wh' in target_app and ('wh' not in stat_app_list):
        item_output_object = update_wh_appliance_profile(item_input_object, item_output_object, logger_itemization)
        item_output_object = update_added_wh_appliance_profile(item_input_object, item_output_object, logger_itemization)
        item_output_object = update_deleted_wh_appliance_profile(item_input_object, item_output_object, logger_itemization)

    if 'hvac' in target_app and ('cooling' not in stat_app_list):
        item_output_object = update_cooling_appliance_profile(item_input_object, item_output_object)

    if 'hvac' in target_app and ('heating' not in stat_app_list):
        item_output_object = update_heating_appliance_profile(item_input_object, item_output_object)

    if 'li' in target_app:
        item_output_object = update_li_appliance_profile(item_input_object, item_output_object, logger_itemization)

    if 'ao' in target_app:
        item_output_object = update_ao_appliance_profile(item_input_object, item_output_object, logger_itemization)

    item_output_object = update_appliance_profile_based_for_stat_app(item_input_object, item_output_object, run_successful, logger_itemization)

    return item_output_object


def update_appliance_profile_based_for_stat_app(item_input_object, item_output_object, run_successful, logger_itemization):

    """
    updates appliance profile based on disagg postprocessing

    Parameters:
        item_input_object         (dict)      : Dict containing all inputs
        item_output_object        (dict)      : Dict containing all outputs
        run_successful            (bool)      : true if hybrid v2 run was successful
        logger_itemization        (logger)    : logger object

    Returns:
        item_output_object        (dict)      : Dict containing all outputs
    """

    itemization_successful = item_output_object.get("final_itemization") is not None and \
                             item_output_object.get('inference_engine_dict') is not None

    target_app = GlobalConfigParams.disagg_postprocess_enabled_app

    if item_input_object.get('item_input_params').get('run_hybrid_v2_flag') and run_successful:
        target_app = GlobalConfigParams.item_aer_seq

    non_mtd_mode = item_input_object.get('config').get('disagg_mode') != 'mtd'

    prepare_results = itemization_successful and non_mtd_mode

    if not prepare_results:
        return item_output_object

    if 'ref' in target_app:
        item_output_object = update_ref_appliance_profile(item_input_object, item_output_object, logger_itemization)

    if 'cook' in target_app:
        item_output_object = update_cook_appliance_profile(item_input_object, item_output_object, logger_itemization)

    if 'ent' in target_app:
        item_output_object = update_ent_appliance_profile(item_input_object, item_output_object, logger_itemization)

    if 'ld' in target_app:
        item_output_object = update_ld_appliance_profile(item_input_object, item_output_object, logger_itemization)

    return item_output_object
