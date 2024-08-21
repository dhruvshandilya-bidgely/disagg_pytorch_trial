"""
Author - Nisha Agarwal
Date - 4th April 2021
This file has functions that is used to maintain a billing cycle level min/max consumption for appliances after adjustment
"""

# Import python packages

import copy
import numpy as np
from numpy.random import RandomState

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants
from python3.itemization.init_itemization_config import random_gen_config

from python3.itemization.aer.functions.itemization_utils import find_seq

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.functions.itemization_utils import get_index_array

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf


def add_prepare_box_type_additional_cons_points(required_cons, final_tou_consumption, additional_cons, vacation,
                                                target_days, app_idx, seed):

    """
    Calculate possible time of use where stat app can be added to maintain min cons for the given appliance as box type signature

    Parameters:
        required_cons
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        additional_cons             (np.ndarray)    : array containing ts level additonal cons to be added
        vacation                    (np.ndarray)    : vacation data
        target_days                 (np.ndarray)    : days of in the target billing cycle
        app_idx                     (int)           : appliance index
        seed                        (int)           : seed to be used to add slight randomness in ts level output added to stat app

    Returns:
        additional_cons             (np.ndarray)    : array containing ts level additonal cons to be added
    """

    samples_per_hour = int(final_tou_consumption.shape[2]/Cgbdisagg.HRS_IN_DAY)

    additional_cons[vacation[target_days]] = 0

    default_max_cons = 5000

    shape = copy.deepcopy(additional_cons.shape)
    additional_cons = additional_cons.flatten()

    # max consumption cap for the stat app

    if np.any(final_tou_consumption[app_idx] > 0):
        max_val = np.percentile(final_tou_consumption[app_idx][final_tou_consumption[app_idx] > 0], 97)
    else:
        max_val = default_max_cons / samples_per_hour

    # additional consumption required to maintain min stat app cons

    additional_cons = np.fmin(additional_cons, max_val)

    index_list1 = np.arange(len(additional_cons))
    index_list2 = copy.deepcopy(index_list1)

    # preparing additional consumption that is to be added into stat app to maintain min cons or consistency

    seed.shuffle(index_list2)
    original_idx_list = np.intersect1d(index_list2, index_list1, return_indices=True)[1]

    additional_cons = additional_cons[index_list2]

    # removing cons point if additional cons is more than required cons
    # by randomly removing points from additional cons array until additional cons array sum matches required consumption

    additiona_cons_more_than_req_cons = (np.sum(additional_cons) > required_cons) and np.any(np.cumsum(additional_cons) > required_cons)

    if additiona_cons_more_than_req_cons:
        index = np.where(np.cumsum(additional_cons) > required_cons)[0][0]
        additional_cons[(index + 1):] = 0

    additional_cons = additional_cons[original_idx_list]
    additional_cons = additional_cons.reshape(shape)

    return additional_cons


def add_step3_cons_into_stat_to_maintain_stability(data, app_tou, target_days, step3_app_id, max_cons, min_cons_required, app_idx, vacation,
                                                   app_cons, step3_app_thres):

    """
    add consumption to statistical PP/EV/WH appliance into cook/ld/ent inorder maintain consistent in stat app output

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        app_tou                     (np.ndarray)    : stat app tou
        target_days                 (np.ndarray)    : days of in the target billing cycle
        step3_app_id                (float)         : target app index
        processed_input_data        (np.ndarray)    : user input data
        max_cons                    (float)         : max cons cap on stat app
        min_cons_required           (float)         : min cons required for stat app in target billing cycle
        app_idx                     (int)           : appliance index
        vacation                    (np.ndarray)    : vacation data
        step3_app_thres             (float)         : max consumption that can be picked form pp/ev/wh app
        app_cons                    (float)         : appliance monthly consumption

    Returns:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
    """

    final_tou_consumption = data[0]
    processed_input_data = data[1]

    step_3_app_cons = final_tou_consumption[step3_app_id]

    stat_app_cons = final_tou_consumption[app_idx, target_days, :]

    # preparing additional consumption that is to be picked from step 3 app cons to maintain minimum cons of stat app

    additional_cons = np.zeros_like(processed_input_data[target_days])

    pick_cons = min(step3_app_thres * (step_3_app_cons[target_days].sum() / Cgbdisagg.WH_IN_1_KWH),
                    ((min_cons_required - app_cons) * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)) * 0.9) * Cgbdisagg.WH_IN_1_KWH

    additional_cons[:, app_tou] = np.fmin(max_cons, step_3_app_cons[target_days][:, app_tou]) /\
                                  np.max(np.fmin(max_cons, step_3_app_cons[target_days][:, app_tou]))

    if np.max(step_3_app_cons[target_days][:, app_tou]) == 0:
        return final_tou_consumption, 1

    # removing consumption on vacation days
    additional_cons = prepare_ts_level_additional_cons_points(additional_cons, vacation, pick_cons, target_days)

    # adding consumption into stat app
    stat_app_cons = stat_app_cons + np.minimum(processed_input_data[target_days], np.minimum(step_3_app_cons[target_days], additional_cons))

    stat_app_cons = np.minimum(stat_app_cons, processed_input_data[target_days])

    # removing cons from step 3 appliance and adding to stat app ts level consumption

    final_tou_consumption[step3_app_id, target_days, :] = np.fmax(0, final_tou_consumption[step3_app_id, target_days, :] -
                                                                  np.fmax(0, np.minimum(step_3_app_cons[target_days], additional_cons)))
    final_tou_consumption[app_idx, target_days, :] = stat_app_cons

    return final_tou_consumption, 0


def prepare_ts_level_additional_cons_points(additional_cons, vacation, pick_cons, target_days):

    """
    Prepare array that contain ts level extra consumption to be added to maintain stat app min cons

    Parameters:
        additional_cons             (np.ndarray)    : array containing ts level additonal cons to be added
        vacation                    (np.ndarray)    : vacation data
        pick_cons                   (float)         : amount of consumption to be added in the target billing cycle
        target_days                 (np.ndarray)    : days of in the target billing cycle

    Returns:
        additional_cons             (np.ndarray)    : updated array containing ts level additonal cons to be added
    """

    # adding slight randomness in additional consumption
    additional_cons = add_randomness(additional_cons)

    # consumption is removed on vacation days
    additional_cons[vacation[target_days]] = 0

    # limiting the additional cons to monthly consumption that is required be added to stat app(pick_cons)
    additional_cons = additional_cons / np.sum(additional_cons)
    additional_cons = np.fmax(0, additional_cons * pick_cons)
    additional_cons = np.nan_to_num(additional_cons)

    return additional_cons


def get_stat_app_min_cons_hard_limit(item_input_object, item_output_object, app_hsm, original_total,
                                     monthly_cons, idx, params):

    """
    Limit max bc level level consumption for all appliances based on hsm and pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        app_hsm                     (float)         : monthly cooking consumption based on HSM
        original_total              (float)         : day time total cons
        idx                         (float)         : billing cycle index

    Returns:
        hard_min_limit              (np.ndarray)    :
        app_cons                    (float)         : hard limit for the appliance in given billing cycle
    """

    initialized_min_cons = params.get('initialized_min_cons')
    use_hsm = params.get('use_hsm')
    app = params.get('app')
    total_cons = params.get('total_cons')
    target_days = params.get('target_days')
    app_cons = params.get('app_cons')
    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    ent_monthly_cons = monthly_cons[1]
    ld_monthly_cons = monthly_cons[0]
    cook_monthly_cons = monthly_cons[2]

    hsm_ent = app_hsm[1]
    hsm_ld = app_hsm[0]
    hsm_cook = app_hsm[2]

    scaling_factor = Cgbdisagg.DAYS_IN_MONTH/Cgbdisagg.WH_IN_1_KWH

    config = init_final_item_conf().get('min_max_limit_conf')

    cons_offset_for_eu = config.get('cons_offset_for_eu')
    cons_offset_for_ind = config.get('cons_offset_for_ind')
    cons_offset = config.get('cons_offset')
    cons_multiplier = config.get('cons_multiplier')

    # prepare threshold that will be used to scale min cons based on total consumption

    pilot = item_input_object.get("config").get("pilot_id")

    cons_threshold = 0

    ld_idx = hybrid_config.get("app_seq").index('ent')
    mid_cons = hybrid_config.get("mid_cons")[ld_idx]

    cons_threshold = cons_threshold + mid_cons

    ld_idx = hybrid_config.get("app_seq").index('ld')
    mid_cons = hybrid_config.get("mid_cons")[ld_idx]

    cons_threshold = cons_threshold + mid_cons

    ld_idx = hybrid_config.get("app_seq").index('cook')
    mid_cons = hybrid_config.get("mid_cons")[ld_idx]

    cons_threshold = cons_threshold + mid_cons

    if hybrid_config.get('geography') == 'eu':
        cons_threshold = cons_threshold + cons_offset_for_eu
    elif pilot in PilotConstants.SEASONAL_WH_ENABLED_PILOTS:
        cons_threshold = cons_threshold + cons_offset_for_ind
    else:
        cons_threshold = cons_threshold + cons_offset

    # preparing scaling factor (to be used to scale hard limit)
    # based on consumption level of current billing cycle compared to other billing cycles

    cons_threshold = cons_threshold * cons_multiplier

    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

    total = (processed_input_data[np.logical_not(vacation)][bc_list[np.logical_not(vacation)] == idx])
    total = ((np.sum(total) / len(processed_input_data[target_days])) * scaling_factor)

    vacation_factor = 1 - (np.sum(vacation[target_days]) / 30)

    scaling_factor_based_on_total_cons = (original_total / cons_threshold)

    scaling_factor_based_on_bc_cons = total / total_cons
    scaling_factor_based_on_bc_cons = np.fmin(1, (np.fmax(scaling_factor_based_on_bc_cons, 0.9)))

    # Calculating hard min limit for entertainment

    ent_params = {
        'initialized_min_cons': initialized_min_cons,
        'hsm_ent': hsm_ent,
        'use_hsm': use_hsm,
        'scaling_factor_based_on_bc_cons': scaling_factor_based_on_bc_cons,
        'ent_monthly_cons': ent_monthly_cons,
        'app_cons': app_cons,
        'vacation_factor': vacation_factor,
        'scaling_factor_based_on_total_cons': scaling_factor_based_on_total_cons,
    }

    ld_params = {
        'initialized_min_cons': initialized_min_cons,
        'hsm_ld': hsm_ld,
        'use_hsm': use_hsm,
        'scaling_factor_based_on_bc_cons': scaling_factor_based_on_bc_cons,
        'ld_monthly_cons': ld_monthly_cons,
        'app_cons': app_cons,
        'vacation_factor': vacation_factor,
        'scaling_factor_based_on_total_cons': scaling_factor_based_on_total_cons,
    }

    params = {
        'initialized_min_cons': initialized_min_cons,
        'hsm_cook': hsm_cook,
        'use_hsm': use_hsm,
        'scaling_factor_based_on_bc_cons': scaling_factor_based_on_bc_cons,
        'cook_monthly_cons': cook_monthly_cons,
        'app_cons': app_cons,
        'vacation_factor': vacation_factor,
        'scaling_factor_based_on_total_cons': scaling_factor_based_on_total_cons,
    }

    if app == "ent":
        hard_min_limit, app_cons = \
            get_hard_min_limit_for_entertainment_category(item_input_object, item_output_object, ent_params)

    # Calculating hard min limit for laundry
    elif app == "ld":
        hard_min_limit, app_cons = \
            get_hard_min_limit_for_laundry_category(item_input_object, item_output_object, ld_params)

    # Calculating hard min limit for cooking
    else:
        hard_min_limit, app_cons = \
            get_hard_min_limit_for_cooking_category(item_input_object, item_output_object, params)

    return hard_min_limit, app_cons


def get_hard_min_limit_for_cooking_category(item_input_object, item_output_object, cook_params):

    """
    Limit max bc level level consumption for cooking appliances based on hsm and pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        cook_params                 (dict)          : inputs required to calculate min cons

    Returns:
        hard_min_limit              (np.ndarray)    : calculated hard limit
        app_cons                    (float)         : hard limit for the appliance in given billing cycle
    """

    cook_monthly_cons = cook_params.get('cook_monthly_cons')
    initialized_min_cons = cook_params.get('initialized_min_cons')
    hsm_cook = cook_params.get('hsm_cook')
    use_hsm = cook_params.get('use_hsm')
    scaling_factor_based_on_bc_cons = cook_params.get('scaling_factor_based_on_bc_cons')
    scaling_factor_based_on_total_cons = cook_params.get('scaling_factor_based_on_total_cons')
    vacation_factor = cook_params.get('vacation_factor')
    app_cons = cook_params.get('app_cons')

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))
    config = init_final_item_conf().get('min_max_limit_conf')
    max_cap_on_hard_limit = config.get('max_cap_on_hard_limit')

    if use_hsm:
        initialized_min_cons = max(initialized_min_cons, hsm_cook)

    hard_min_limit = max(initialized_min_cons, cook_monthly_cons * scaling_factor_based_on_bc_cons * vacation_factor)
    ld_idx = hybrid_config.get("app_seq").index('cook')
    scale_cons_flag = hybrid_config.get("scale_app_cons")[ld_idx]
    mid_cons = hybrid_config.get("mid_cons")[ld_idx]

    scaling_factor_based_on_app_survey = 1


    # scale hard limit based on cooking app count in app profile

    if not item_input_object.get("appliance_profile").get("default_cooking_flag"):

        cooking_app_count = copy.deepcopy(item_input_object.get("appliance_profile").get("cooking"))
        cooking_app_type = item_input_object.get("appliance_profile").get("cooking_type")

        if str(np.nan_to_num(item_input_object.get('pilot_level_config').get('cook_config').get('type'))) == 'GAS':
            cooking_app_count[cooking_app_type == 2] = cooking_app_count[cooking_app_type == 2] * 2
            cooking_app_count[cooking_app_type == 0] = item_input_object.get("appliance_profile").get("default_cooking_count")[cooking_app_type == 0]
        else:
            cooking_app_count[cooking_app_type == 0] = cooking_app_count[cooking_app_type == 0] * 0

        app_profile = item_input_object.get("app_profile").get(31)

        if hybrid_config.get("dishwash_cat") == "cook":
            if app_profile is not None:
                app_profile = app_profile.get("number", 0)
            else:
                app_profile = 0
        else:
            app_profile = 0

        cooking_app_count = cooking_app_count.sum() + app_profile * 2
        scaling_factor_based_on_app_survey = (cooking_app_count / item_input_object.get("appliance_profile").get("default_cooking_count").sum())
        hard_min_limit = hard_min_limit * scaling_factor_based_on_app_survey

    max_cons_val = hybrid_config.get("max_cons")[ld_idx]

    # scale hard limit based on total consumption of the user
    if scale_cons_flag:
        new_limit = min(max_cap_on_hard_limit, adjust_cook_limit(item_input_object, item_output_object,
                                                                 (mid_cons * scaling_factor_based_on_total_cons),
                                                                 scaling_factor_based_on_app_survey, hard_min_limit))
        hard_min_limit = min(max_cons_val * 0.9, max(hard_min_limit, new_limit))

    # scale hard limit for cooking to resolve underestimation in high cooking pilots

    if mid_cons > 75:
        hard_min_limit = hard_min_limit * 1.2
    if mid_cons > 100:
        hard_min_limit = hard_min_limit * 1.1

    if app_cons == 0 and cook_monthly_cons > 0:
        app_cons = 1

    # hard limit should be atleast min app cons of the appliance
    ld_idx = hybrid_config.get("app_seq").index('cook')
    have_min_cons = hybrid_config.get("have_hard_min_lim")[ld_idx]
    min_cons = hybrid_config.get("hard_min_lim")[ld_idx]

    if have_min_cons:
        hard_min_limit = max(hard_min_limit, min_cons)

    return hard_min_limit, app_cons


def get_hard_min_limit_for_laundry_category(item_input_object, item_output_object, ld_params):
    """
    Limit max bc level level consumption for laundry appliances based on hsm and pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        ld_params                   (dict)          : inputs required to calculate min cons

    Returns:
        hard_min_limit              (np.ndarray)    : calculated hard limit
        app_cons                    (float)         : hard limit for the appliance in given billing cycle
    """

    ld_monthly_cons = ld_params.get('ld_monthly_cons')
    initialized_min_cons = ld_params.get('initialized_min_cons')
    hsm_ld = ld_params.get('hsm_ld')
    use_hsm = ld_params.get('use_hsm')
    scaling_factor_based_on_bc_cons = ld_params.get('scaling_factor_based_on_bc_cons')
    scaling_factor_based_on_total_cons = ld_params.get('scaling_factor_based_on_total_cons')
    vacation_factor = ld_params.get('vacation_factor')
    app_cons = ld_params.get('app_cons')

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))
    config = init_final_item_conf().get('min_max_limit_conf')
    max_cap_on_hard_limit = config.get('max_cap_on_hard_limit')

    if use_hsm:
        initialized_min_cons = max(initialized_min_cons, hsm_ld)

    hard_min_limit = max(initialized_min_cons, ld_monthly_cons * 1 * scaling_factor_based_on_bc_cons * vacation_factor)
    ld_idx = hybrid_config.get("app_seq").index('ld')
    scale_cons_flag = hybrid_config.get("scale_app_cons")[ld_idx]
    mid_cons = hybrid_config.get("mid_cons")[ld_idx]

    scaling_factor_based_on_app_survey = 1

    # scale hard limit based on laundry app count in app profile
    if not item_input_object.get("appliance_profile").get("default_laundry_flag"):
        ld_app_count = item_input_object.get("appliance_profile").get("laundry")
        scaling_factor_based_on_app_survey = np.sum(ld_app_count) / \
                                             (max(1, item_input_object.get("appliance_profile").get("default_laundry_count").sum() - 0.7))

        hard_min_limit = hard_min_limit * scaling_factor_based_on_app_survey

    max_cons_val = hybrid_config.get("max_cons")[ld_idx]

    # scale hard limit based on total consumption of the user
    if scale_cons_flag:
        new_limit = min(max_cap_on_hard_limit, adjust_ld_limit(item_input_object, item_output_object,
                                                               (mid_cons * scaling_factor_based_on_total_cons),
                                                               scaling_factor_based_on_app_survey, hard_min_limit))
        hard_min_limit = min(max_cons_val, max(hard_min_limit, new_limit))

    if app_cons == 0 and ld_monthly_cons > 0:
        app_cons = 1

    # hard limit should be atleast min app cons of the appliance
    ld_idx = hybrid_config.get("app_seq").index('ld')
    have_min_cons = hybrid_config.get("have_hard_min_lim")[ld_idx]
    min_cons = hybrid_config.get("hard_min_lim")[ld_idx]

    if have_min_cons:
        hard_min_limit = max(hard_min_limit, min_cons)

    return hard_min_limit, app_cons


def get_hard_min_limit_for_entertainment_category(item_input_object, item_output_object, ent_params):

    """
    Limit max bc level level consumption for entertainment appliances based on hsm and pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        ent_params                  (dict)          : inputs required to calculate min cons

    Returns:
        hard_min_limit              (np.ndarray)    : calculated hard limit
        app_cons                    (float)         : hard limit for the appliance in given billing cycle
    """

    ent_monthly_cons = ent_params.get('ent_monthly_cons')
    initialized_min_cons = ent_params.get('initialized_min_cons')
    hsm_ent = ent_params.get('hsm_ent')
    use_hsm = ent_params.get('use_hsm')
    scaling_factor_based_on_bc_cons = ent_params.get('scaling_factor_based_on_bc_cons')
    scaling_factor_based_on_total_cons = ent_params.get('scaling_factor_based_on_total_cons')
    vacation_factor = ent_params.get('vacation_factor')
    app_cons = ent_params.get('app_cons')

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))
    config = init_final_item_conf().get('min_max_limit_conf')
    max_cap_on_hard_limit = config.get('max_cap_on_hard_limit')

    if use_hsm:
        initialized_min_cons = max(initialized_min_cons, hsm_ent)

    hard_min_limit = max(initialized_min_cons, ent_monthly_cons * scaling_factor_based_on_bc_cons * vacation_factor)
    ld_idx = hybrid_config.get("app_seq").index('ent')
    scale_cons_flag = hybrid_config.get("scale_app_cons")[ld_idx]
    mid_cons = hybrid_config.get("mid_cons")[ld_idx]

    max_cons_val = hybrid_config.get("max_cons")[ld_idx]

    scaling_factor_based_on_app_survey = 1

    # scale hard limit based on entertainment app count in app profile
    if not item_input_object.get("appliance_profile").get("default_ent_flag"):
        ent = item_input_object.get("appliance_profile").get("ent")
        cons = [200, 200, 700]
        scaling_factor_based_on_app_survey = max(0.2, min(2, (np.dot(ent, cons)/ 700)))

        hard_min_limit = hard_min_limit * scaling_factor_based_on_app_survey

    # scale hard limit based on total consumption of the user

    if scale_cons_flag:
        new_limit = min(max_cap_on_hard_limit,
                        adjust_ent_limit_based_on_user_profile(item_input_object, item_output_object,
                                                               (mid_cons * scaling_factor_based_on_total_cons),
                                                               scaling_factor_based_on_app_survey, hard_min_limit))

        hard_min_limit = min(max_cons_val, max(hard_min_limit, new_limit))

    if app_cons == 0 and ent_monthly_cons > 0:
        app_cons = 1

    # hard limit should be atleast min app cons of the appliance
    ld_idx = hybrid_config.get("app_seq").index('ent')
    have_min_cons = hybrid_config.get("have_hard_min_lim")[ld_idx]
    min_cons = hybrid_config.get("hard_min_lim")[ld_idx]

    if have_min_cons:
        hard_min_limit = max(hard_min_limit, min_cons)

    return hard_min_limit, app_cons


def prepare_additional_cons_points_to_satisfy_min_cons(additional_cons, hard_min_limit, target_days, app_cons):

    """
    prepare the array that contain ts level extra consumption that is to be added to maintain stat app min cons

    Parameters:
        additional_cons             (np.ndarray)    : array containing ts level additonal cons to be added
        hard_min_limit              (float)         : min appliance consumption
        target_days                 (np.ndarray)    : days of in the target billing cycle
        app_cons                    (float)         : appliance monthly consumption

    Returns:
        additional_cons             (np.ndarray)    : updated array containing ts level additonal cons to be added
    """
    additional_cons = additional_cons / np.sum(additional_cons)
    additional_cons = np.nan_to_num(additional_cons)
    diff = (hard_min_limit - app_cons) * Cgbdisagg.WH_IN_1_KWH * (np.sum(target_days) / Cgbdisagg.DAYS_IN_MONTH)
    additional_cons = additional_cons * diff

    return additional_cons


def get_app_cons_in_target_bc(final_tou_consumption, vacation, target_days, app_idx):
    """
    This function calculates month level consumption of the user scaled based on  vacation days
    if all days are vacation, a high value is returned since this billing cycle wont be used for maintaining a minimum consumption

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        vacation                    (np.ndarray)    : vacation data
        target_days                 (np.ndarray)    : days of in the target billing cycle
        app_idx                     (int)           : appliance index

    Returns:
        app_cons                    (float)         : appliance montjly consumption
    """
    scaling_factor = Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH

    # calculates month level stat app consumption scaled based on number of days and vacation days

    app_cons = final_tou_consumption[app_idx, target_days, :][np.logical_not(vacation[target_days])]
    app_cons = ((np.sum(app_cons) / len(final_tou_consumption[app_idx, target_days, :])) * (scaling_factor))

    if (1 - (np.sum(vacation[target_days]) / np.sum(target_days))) == 0:
        app_cons = 1000000000
    else:
        app_cons = app_cons / (1 - (np.sum(vacation[target_days]) / np.sum(target_days)))

    app_cons = np.fmax(0, app_cons)

    return app_cons


def prepare_stat_tou(final_tou_consumption, app_idx, use_hsm, hsm_cons, app, hybrid_config, sleep_hours):
    """
    Calculate possible time of use where stat app can be added to maintain min cons for the given appliance

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        app_idx                     (int)           : appliance index
        use_hsm                     (bool)          : flag that denotes whether hsm can be used for the given appliannce
        hsm_cons                    (float)         : monthly consumption based on hsm
        samples_per_hour            (int)           : samples in an hour
        app                         (str)           : target appliance
        hybrid_config               (dict)          : pilot config
        sleep_hours                 (np.ndarray)    : inactive hours of the user

    Returns:
        stat_tou                    (np.ndarray)    : calculated stat time of use
    """
    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END

    samples_per_hour = int(final_tou_consumption.shape[2]/Cgbdisagg.HRS_IN_DAY)

    default_buffer_hours = samples_per_hour
    buffer_hours_for_cook = [1.5*samples_per_hour, 2.5*samples_per_hour][np.digitize(hybrid_config.get("mid_cons")[hybrid_config.get("app_seq").index('cook')], [75])]
    activity_hours = np.arange(12 * samples_per_hour, 20 * samples_per_hour + 1)

    stat_tou = get_stat_tou(final_tou_consumption, app_idx, use_hsm, hsm_cons)

    total_samples = Cgbdisagg.HRS_IN_DAY * samples_per_hour

    insufficient_stat_hours_count = np.sum(stat_tou) < 7 * samples_per_hour

    if insufficient_stat_hours_count:

        # extend app tou in nearby hours

        tou_seq = find_seq(stat_tou, np.zeros_like(stat_tou), np.zeros_like(stat_tou))
        for i in range(len(tou_seq)):

            if tou_seq[i, seq_label] and app == 'cook':
                stat_tou[get_index_array(tou_seq[i, seq_start] - buffer_hours_for_cook, tou_seq[i, seq_start], total_samples)] = 1
                stat_tou[get_index_array(tou_seq[i, seq_end], tou_seq[i, seq_end] + buffer_hours_for_cook, total_samples)] = 1
            if tou_seq[i, seq_label] and app in ['ent', 'ld']:
                stat_tou[get_index_array(tou_seq[i, seq_start] - default_buffer_hours, tou_seq[i, seq_start], total_samples)] = 1
                stat_tou[get_index_array(tou_seq[i, seq_end], tou_seq[i, seq_end] + default_buffer_hours, total_samples)] = 1

        sleep_hours[activity_hours] = 1
        stat_tou[np.logical_not(sleep_hours)] = 0

    return stat_tou


def get_stat_tou(final_tou_consumption, app_idx, use_hsm, hsm_cons):

    """
    Calculate initial time of use for target stat app

    Parameters:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        app_idx                     (int)           : index of target app
        use_hsm                     (int)           : flag that denotes whether HSM can be used for the given app
        hsm_cons                    (float)         : monthly cons in app hsm

    Returns:
        stat_tou                    (np.ndarray)    : stat app TOU
    """
    cons_frac_levels = [0.2, 0.15, 0]

    stat_tou = (np.sum(final_tou_consumption[app_idx] > 0, axis=0) / len(final_tou_consumption[app_idx])) > cons_frac_levels[0]

    if use_hsm and np.sum(stat_tou) == 0 and hsm_cons > 0:
        stat_tou[:] = 1

    if not np.any(stat_tou):
        stat_tou = (np.sum(final_tou_consumption[app_idx] > 0, axis=0) / len(final_tou_consumption[app_idx])) > cons_frac_levels[1]

        if not np.any(stat_tou):
            stat_tou = (np.sum(final_tou_consumption[app_idx] > 0, axis=0) / len(final_tou_consumption[app_idx])) > cons_frac_levels[2]

            if not np.any(stat_tou):
                return stat_tou

    return stat_tou


def apply_soft_limit_on_bc_level_min_cons(item_input_object, item_output_object, final_tou_consumption, length,
                                          app_month_cons, total_monthly_cons, logger):
    """
    Limit min bc level level consumption for all appliances based on pilot config

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        length                      (int)           : total non vacation days of the user
        app_month_cons              (list)          : monthly stat app consumption
        total_monthly_cons          (float)         : monthly total consumption
        logger                      (logger)        : logger object

    Returns:
        final_tou_consumption       (np.ndarray)    : final ts level estimation of appliances
        ld_monthly_cons             (float)         : monthly laundry consumption
        ent_monthly_cons            (float)         : monthly ent consumption
        cook_monthly_cons           (float)         : monthly cooking consumption
    """

    ld_monthly_cons = app_month_cons[0]
    ent_monthly_cons = app_month_cons[1]
    cook_monthly_cons = app_month_cons[2]
    processed_input_data = item_output_object.get('original_input_data')
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool)
    appliance_list = np.array(item_input_object.get("item_input_params").get('app_list'))

    ld_bc_cons = ld_monthly_cons
    ent_bc_cons = ent_monthly_cons
    cook_bc_cons = cook_monthly_cons

    # Min consumption limit based on total consumption of the user

    scaling_factor = Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH

    config = init_final_item_conf(total_monthly_cons).get('min_max_limit_conf')

    ld_limit = config.get('ld_min_limit')
    ent_limit = config.get('ent_min_limit')
    cook_limit = config.get('cook_min_limit')

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    stat_app_array = [ld_monthly_cons, ent_monthly_cons, cook_monthly_cons]
    dev_qa_idx_arr = [0, 1, 2]
    stat_app_list = ['ld', 'ent', 'cook']
    bc_level_cons = [ld_bc_cons, ent_bc_cons, cook_bc_cons]
    cons_limit_arr = [ld_limit, ent_limit, cook_limit]
    def_type_arr = [item_input_object.get("appliance_profile").get("default_laundry_flag"),
                    item_input_object.get("appliance_profile").get("default_ent_flag"),
                    item_input_object.get("appliance_profile").get("default_cooking_flag")]

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # Modify consumption limit based on pilot config

    ld_idx = hybrid_config.get("app_seq").index('ld')
    have_min_cons = hybrid_config.get("have_min_cons")[ld_idx]
    min_cons = hybrid_config.get("min_cons")[ld_idx]

    if have_min_cons:
        cons_limit_arr[0] = min(cons_limit_arr[0], min_cons)

    ld_idx = hybrid_config.get("app_seq").index('cook')
    have_min_cons = hybrid_config.get("have_min_cons")[ld_idx]
    min_cons = hybrid_config.get("min_cons")[ld_idx]

    if have_min_cons:
        cons_limit_arr[2] = min(cons_limit_arr[2], min_cons)

    ld_idx = hybrid_config.get("app_seq").index('ent')
    have_min_cons = hybrid_config.get("have_min_cons")[ld_idx]
    min_cons = hybrid_config.get("min_cons")[ld_idx]

    if have_min_cons:
        cons_limit_arr[1] = min(cons_limit_arr[1], min_cons)

    logger.info("Minimum cons required for ent | %s", int(cons_limit_arr[1]))
    logger.info("Minimum cons required for cook | %s", int(cons_limit_arr[2]))
    logger.info("Minimum cons required for ld | %s", int(cons_limit_arr[0]))

    # increase monthly consumption of each bc with stat app output less than the required cons

    for idx, app in enumerate(stat_app_list):
        if def_type_arr[idx] and bc_level_cons[idx] < cons_limit_arr[idx] and bc_level_cons[idx] > 0:
            factor = bc_level_cons[idx] / cons_limit_arr[idx]

            app_idx = get_app_idx(appliance_list, app)

            additional_cons = final_tou_consumption[app_idx] / factor - final_tou_consumption[app_idx]
            additional_cons = np.minimum(additional_cons, other_cons_arr)

            final_tou_consumption[app_idx] = final_tou_consumption[app_idx] + np.fmax(0, additional_cons)
            final_tou_consumption[app_idx] = np.minimum(final_tou_consumption[app_idx], processed_input_data)

            cons = final_tou_consumption[app_idx][np.logical_not(vacation)]

            stat_app_array[dev_qa_idx_arr[idx]] = ((np.sum(cons) / length) * scaling_factor)

    ld_monthly_cons = stat_app_array[0]
    ent_monthly_cons = stat_app_array[1]
    cook_monthly_cons = stat_app_array[2]

    return final_tou_consumption, ld_monthly_cons, ent_monthly_cons, cook_monthly_cons


def add_randomness(consumption):
    """
     add slight randomness in consumption to be added to maintain min consumption for stat app

     Parameters:
         consumption                 (np.ndarray)          : original additional cons

     Returns:
         new_consumption             (np.ndarray)          : updated additional cons
     """
    new_consumption = copy.deepcopy(consumption)
    seed = RandomState(random_gen_config.seed_value)

    randomness = seed.normal(1.05, 0.15, new_consumption.shape)
    randomness[randomness < 0.65] = 0

    new_consumption = np.multiply(new_consumption, np.fmin(randomness, 1.1))

    return new_consumption


def adjust_cook_limit(item_input_object, item_output_object, limit, scaling_factor_based_on_app_survey, min_cons):
    """
     adjust min appliance consumption based on meta data and occupancy profile

     Parameters:
         item_input_object           (dict)          : Dict containing all hybrid inputs
         item_output_object          (dict)          : Dict containing all hybrid outputs
         limit                       (float)         : appliance min cons

     Returns:
         limit                       (float)         : appliance min cons
     """
    config = init_final_item_conf().get('min_max_limit_conf')
    cook_occ_count_buc = config.get('cook_occ_count_buc')
    cook_occ_count_scaling_fac = config.get('cook_occ_count_scaling_fac')

    low_avg = (limit / min_cons) < 2

    # update cooking min cons based on user room count or occupants count , if available

    room_count = item_input_object.get('home_meta_data').get('totalRooms', 0)

    # update cooking min cons based on occupants type

    if room_count > 3 and low_avg:
        limit = limit*1.2
    elif room_count == 1:
        limit = limit*0.8

    room_count = item_input_object.get('home_meta_data').get('numOccupants', 0)

    if low_avg:
        limit = limit * cook_occ_count_scaling_fac[np.digitize(room_count, cook_occ_count_buc)]

    occupants_features = item_output_object.get('occupants_profile').get('occupants_features')

    if occupants_features[2] == 0:
        limit = limit * 0.8
    elif (occupants_features[0] == 0) or (occupants_features.sum() > 3):
        limit = limit * 1.1

    # update cooking min cons based on app profile information

    if not item_input_object.get("appliance_profile").get("default_cooking_flag"):
        limit = limit * scaling_factor_based_on_app_survey

    return limit


def adjust_ld_limit(item_input_object, item_output_object, limit, scaling_factor_based_on_app_survey, min_cons):
    """
    adjust min appliance consumption based on meta data and occupancy profile

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        limit                       (float)         : appliance min cons

    Returns:
        limit                       (float)         : appliance min cons
    """
    config = init_final_item_conf().get('min_max_limit_conf')
    ld_room_count_buc = config.get('ld_room_count_buc')
    ld_room_count_scaling_fac = config.get('ld_room_count_scaling_fac')
    ld_occ_count_buc = config.get('ld_occ_count_buc')
    ld_occ_count_scaling_fac = config.get('ld_occ_count_scaling_fac')

    low_avg = (limit / min_cons) < 2

    # update cooking min cons based on user room count or occupants count , if available

    if low_avg:
        room_count = item_input_object.get('home_meta_data').get('totalRooms', 0)

        limit = limit * ld_room_count_scaling_fac[np.digitize(room_count, ld_room_count_buc)]
        occ_count = item_input_object.get('home_meta_data').get('numOccupants', 0)

        # update laundry min cons based on occupants type

        limit = limit * ld_occ_count_scaling_fac[np.digitize(occ_count, ld_occ_count_buc)]

    occupants_features = item_output_object.get('occupants_profile').get('occupants_features', 0)

    if occupants_features[2] == 0:
        limit = limit * 0.8
    elif (occupants_features[0] == 0) or (occupants_features.sum() > 3):
        limit = limit * 1.1

    room_count = item_input_object.get('home_meta_data').get('dwelling', 0)

    if room_count == 1:
        limit = limit * 0.7

    # updating min limit based on app prof input

    if not item_input_object.get("appliance_profile").get("default_laundry_flag"):
        limit = limit * scaling_factor_based_on_app_survey

    limit = limit * 0.9

    return limit


def adjust_ent_limit_based_on_user_profile(item_input_object, item_output_object, limit, scaling_factor_based_on_app_survey, min_cons):
    """
    adjust min appliance consumption based on meta data and occupancy profile

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        item_output_object          (dict)          : Dict containing all hybrid outputs
        limit                       (float)         : appliance min cons

    Returns:
        limit                       (float)         : appliance min cons
    """
    config = init_final_item_conf().get('min_max_limit_conf')
    ent_room_count_buc = config.get('ent_room_count_buc')
    ent_room_count_scaling_fac = config.get('ent_room_count_scaling_fac')
    ent_occ_count_buc = config.get('ent_occ_count_buc')
    ent_occ_count_scaling_fac = config.get('ent_occ_count_scaling_fac')

    low_avg = (limit / min_cons) < 2

    if low_avg:
        # update entertainment min cons based on user room count or occupants count , if available

        room_count = item_input_object.get('home_meta_data').get('totalRooms', 0)

        limit = limit * ent_room_count_scaling_fac[np.digitize(room_count, ent_room_count_buc)]

    occupants_features = item_output_object.get('occupants_profile').get('occupants_features')

    if occupants_features[2] == 0:
        limit = limit * 0.8
    elif (occupants_features[0] == 0) or (occupants_features.sum() > 3):
        limit = limit * 1.1

    occ_count = item_input_object.get('home_meta_data').get('numOccupants', 0)

    limit = limit * ent_occ_count_scaling_fac[np.digitize(occ_count, ent_occ_count_buc)]

    limit = limit * scaling_factor_based_on_app_survey

    return limit


def get_app_idx(appliance_list, app):
    """
    fetch app index from appliance list

    Parameters:
        appliance_list           (list)         : appliance list of residential user
        app                      (str)          : target appliance

    Returns:
        app_idx                  (int)          : App index
    """
    app_idx = np.where(appliance_list == app)[0][0]
    return app_idx
