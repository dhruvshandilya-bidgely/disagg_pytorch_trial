
"""
Author - Nisha Agarwal
Date - 9th Feb 20
Laundry detection module
"""

# Import python packages

import logging
import numpy as np

# import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.itemization.aer.functions.get_config import get_hybrid_config

from python3.itemization.aer.laundry.config.get_detection_config import get_detection_config

from python3.itemization.aer.raw_energy_itemization.residual_analysis.detect_box_type_cons import box_detection


def detect_laundry(item_input_object, item_output_object, weekday_energy_delta, weekend_energy_delta, logger_pass):

    """
    Detect laundry for the user

       Parameters:
           item_input_object             (dict)               : Dict containing all hybrid inputs
           item_output_object            (dict)               : Dict containing all hybrid outputs
           weekday_energy_delta          (np.ndarray)         : weekday time stamp level energy delta
           weekend_energy_delta          (np.ndarray)         : weekend time stamp level energy delta
           logger_pass                   (np.ndarray)         : logger object

       Returns:
           laundry_detection             (int)                : True if laundry is present for the user
    """

    # initialization of required parameters

    pilot = item_input_object.get("config").get("pilot_id")
    activity_curve = item_input_object.get("activity_curve")
    dwelling = item_input_object.get("home_meta_data").get("dwelling")
    pp_user = item_input_object.get("item_input_params").get("pp_user")
    ev_user = item_input_object.get("item_input_params").get("ev_user")
    ao_cons = item_input_object.get("item_input_params").get("ao_cons")
    input_data = item_input_object.get("item_input_params").get("original_input_data")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")
    clean_days = item_input_object.get("clean_day_score_object").get("clean_day_masked_array")[:, 0]
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1) > 0

    logger_base = logger_pass.get('logger_base').getChild('detect_laundry')
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))

    config = get_detection_config(item_input_object.get('pilot_level_config'), pilot)

    # if appliance killer conditions are true for laundry, based on hybrid pilot config
    # laundry detection flag is made 0

    activity_curve_diff = np.nan_to_num(np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    app_killer = item_input_object['item_input_params']['app_killer'][10]

    if app_killer == 1:
        logger.info('Giving 0 laundry based on app profile | ')
        laundry_detection = 0
        return laundry_detection

    # Laundry detection for pilots with minimal wh and cooking usage,
    # thus major box type signatures in residual can be considered for laundry detection

    laundry_detected_by_box_detection = laundry_det_based_box_detection(item_input_object)

    days_in_month = Cgbdisagg.DAYS_IN_MONTH
    wh_multiplier = Cgbdisagg.WH_IN_1_KWH

    perc_cap_for_max_cons = config.get('perc_cap_for_max_cons')

    # checking consumption level of the user that will be used for laundry detection

    max_consumption = np.percentile(input_data, 98, axis=1)

    if np.sum(clean_days != -1) > days_in_month:
        max_consumption = np.percentile(max_consumption[clean_days != -1], perc_cap_for_max_cons) * samples_per_hour
    else:
        max_consumption = np.percentile(max_consumption, perc_cap_for_max_cons) * samples_per_hour

    dwelling_type = "not_known"

    sufficient_clean_days_present = ((clean_days != -1).sum() / len(clean_days)) > 0.05

    if sufficient_clean_days_present:
        monthly_cons = input_data[np.logical_and(np.logical_not(vacation), clean_days != -1)]
        length = np.sum(np.logical_and(np.logical_not(vacation), clean_days != -1))
        monthly_cons = ((np.sum(monthly_cons) / length) * (days_in_month / wh_multiplier))
    else:
        monthly_cons = input_data[(np.logical_not(vacation))]
        length = np.sum((np.logical_not(vacation)))
        monthly_cons = ((np.sum(monthly_cons) / length) * (days_in_month / wh_multiplier))

    # determine the dwelling type of the user

    if dwelling in config.get("house_ids"):
        dwelling_type = "house"
        logger.info('user has independent home based on user input | ')
    elif (dwelling in config.get("flat_ids")) or (ao_cons < config.get("flat_ao_lim")):
        dwelling_type = "flat"
    elif (ao_cons > config.get("house_ao_lim")) or ev_user or pp_user:
        dwelling_type = "house"

    laundry_prof_present = (not item_input_object.get("appliance_profile").get("default_laundry_flag")) and (not app_killer)

    if laundry_prof_present:
        laundry_app_count = item_input_object.get("appliance_profile").get("laundry")
        laundry_app_type = item_input_object.get("appliance_profile").get("laundry_type")
        count = np.dot(laundry_app_count, laundry_app_type)

        if count != 0:
            laundry_detection = 1
            logger.info("Laundry detection using app profile, %d", laundry_detection)
            return laundry_detection

    laundry_detection = 1

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # giving laundry detection if laundry coverage is 0 or 100
    coverage = hybrid_config.get("coverage")

    coverage_used_for_laundry_det = 0

    if coverage == 100 or coverage == 0:
        laundry_detection = coverage == 100
        coverage_used_for_laundry_det = 1

    # detection of laundry based on raw data of the user

    logger.info("Dwelling type | %s ", dwelling_type)
    logger.info("Required laundry coverage | %s ", coverage)
    logger.info("Range of activity curve, %d", activity_curve_diff)

    if not coverage_used_for_laundry_det:
        laundry_detection = \
            laundry_det_based_on_cons_level(item_input_object, item_output_object, weekend_energy_delta,
                                            weekday_energy_delta, max_consumption, logger,
                                            laundry_detected_by_box_detection, monthly_cons, dwelling_type)

    return laundry_detection


def laundry_det_based_box_detection(item_input_object):

    """
    Detect laundry for the user

       Parameters:
           item_input_object             (dict)               : Dict containing all hybrid inputs

       Returns:
           laundry_detection             (int)                : True if laundry is present for the user
    """

    pilot = item_input_object.get("config").get("pilot_id")
    input_data = item_input_object.get("item_input_params").get("original_input_data")
    samples_per_hour = item_input_object.get("item_input_params").get("samples_per_hour")

    config = get_detection_config(item_input_object.get('pilot_level_config'), pilot, samples_per_hour)

    max_ld_cons = config.get('max_ld_cons')
    min_ld_cons = config.get('min_ld_cons')
    min_ld_len = config.get('min_ld_len')
    max_ld_len = config.get('max_ld_len')
    active_usage_hours =  config.get('active_usage_hours')
    min_ld_box_in_a_wind = config.get('min_ld_box_in_a_wind')
    min_days_frac_for_ld_det = config.get('min_days_frac_for_ld_det')

    laundry_detected_by_box_detection = -1

    if pilot in PilotConstants.INDIAN_PILOTS:

        rolling_window = Cgbdisagg.DAYS_IN_MONTH * 0.5

        # determine features of box in leftover residual

        box_label, box_cons, box_seq = \
            box_detection(pilot, np.fmax(0, input_data - item_input_object['item_input_params']['output_data'][1]),
                          input_data, np.zeros_like(input_data),
                          min_amp=min_ld_cons / samples_per_hour,
                          max_amp=max_ld_cons / samples_per_hour,
                          min_len=min_ld_len,
                          max_len=int(max_ld_len * samples_per_hour), detect_wh=1)

        box_seq[box_seq[:, 3] > max(0.75 * samples_per_hour, 2), 0] = 0
        box_seq[box_seq[:, 4] < min_ld_cons / samples_per_hour, 0] = 0
        box_seq[box_seq[:, 4] > (max_ld_cons * 2) / samples_per_hour, 0] = 0

        box_data = box_cons.flatten()

        for i in range(len(box_seq)):
            if not box_seq[i, 0]:
                box_data[int(box_seq[i, 1]):int(box_seq[i, 2]) + 1] = 0

        box_data = np.reshape(box_data, input_data.shape)

        # boxes are only considered in active hours of the days

        active_hours_box_data = box_data[:, active_usage_hours]

        box_label = active_hours_box_data > 0
        box_label = np.sum(box_label, axis=1)

        ld_boxes_week_count = 0

        # Determines if detected boxes are seasonal or present throughout the year

        for i in range(0, len(box_label) - int(rolling_window) - 1, int(rolling_window)):

            sufficient_boxes_present_in_the_week = np.sum(box_label[i:i + int(rolling_window)] > 0) > min_ld_box_in_a_wind

            if sufficient_boxes_present_in_the_week:
                ld_boxes_week_count = ld_boxes_week_count + 1

        laundry_detected_by_box_detection = ld_boxes_week_count > (len(box_label) / rolling_window * min_days_frac_for_ld_det)

    return laundry_detected_by_box_detection


def laundry_det_based_on_cons_level(item_input_object, item_output_object, weekend_energy_delta,
                                    weekday_energy_delta, max_consumption, logger, laundry_detected_by_box_detection,
                                    monthly_cons, dwelling_type):

    """
    Detect laundry for the user

       Parameters:
           item_input_object                (dict)               : Dict containing all hybrid inputs
           item_output_object               (dict)               : Dict containing all hybrid outputs
           weekday_energy_delta             (np.ndarray)         : weekday time stamp level energy delta
           weekend_energy_delta             (np.ndarray)         : weekend time stamp level energy delta
           max_consumption                  (int)                : max consumption of the user
           logger                           (logger)             : logger object
           laundry_detected_by_box_detection(bool)               : flag that represents weather ld is already detected by box detection
           monthly_cons                     (float)              : month level cons of the user
           dwelling_type                    (str)                : dwelling type of the user

       Returns:
           laundry_detection                (int)                : True if laundry is present for the user
    """

    pilot = item_input_object.get("config").get("pilot_id")
    ao_cons = item_input_object.get("item_input_params").get("ao_cons")
    input_data = item_input_object.get("item_input_params").get("original_input_data")
    clean_days = item_input_object.get("clean_day_score_object").get("clean_day_masked_array")[:, 0]
    vacation = item_input_object.get("item_input_params").get("vacation_data").sum(axis=1) > 0
    vacation_frac = np.sum(vacation) / len(vacation)

    laundry_detection = 1
    config = get_detection_config(item_input_object.get('pilot_level_config'), pilot)

    max_monthly_cons = config.get('max_monthly_cons')
    min_monthly_cons = config.get('min_monthly_cons')
    pilots_to_check_ao_cons = config.get('pilots_to_check_ao_cons')

    zero_data_days_present = (np.sum(input_data == 0) / np.sum(input_data >= 0)) > 0.7
    high_vac_days_present = vacation_frac > 0.75
    high_ao_component_present = ((ao_cons * Cgbdisagg.DAYS_IN_MONTH / 1000) / monthly_cons > 0.75) and (pilot not in pilots_to_check_ao_cons)
    high_ao_component_present = high_ao_component_present or (((ao_cons * Cgbdisagg.DAYS_IN_MONTH / 1000) / monthly_cons > 0.9) and (pilot in pilots_to_check_ao_cons))
    less_days_present = len(input_data) < 15

    if high_vac_days_present:
        laundry_detection = 0
        logger.info("High vacation user, %d", laundry_detection)

    elif zero_data_days_present:
        laundry_detection = 0
        logger.info("High 0 consumption value user, %d", laundry_detection)

    elif monthly_cons > max_monthly_cons:
        laundry_detection = 1
        logger.info("High consumption user, %d", laundry_detection)

    elif monthly_cons < min_monthly_cons:
        laundry_detection = 0
        logger.info("Low consumption user, %d", laundry_detection)

    elif high_ao_component_present:
        laundry_detection = 0
        logger.info("High AO consumption user, %d", laundry_detection)

    elif less_days_present:
        laundry_detection = 0
        logger.info("Less number of days")

    elif laundry_detected_by_box_detection != -1:
        laundry_detection = laundry_detected_by_box_detection
        logger.info("Laundry detection for indian pilot user")

    # Laundry is absent if low consumption level

    elif np.sum(input_data[clean_days == 0, :]) / len(input_data[clean_days == 0, :]) < config.get("cons_limit"):
        laundry_detection = 0
        logger.info("Low consumption level")

    # Detected laundry if dwelling type is not known

    elif dwelling_type == "not_known":
        laundry_detection = \
            laundry_det_for_users_with_dwelling_type_unknown(item_input_object, item_output_object, weekend_energy_delta,
                                                             weekday_energy_delta, max_consumption, logger)

    # Detected laundry if dwelling type is flat type

    elif dwelling_type == "flat":
        laundry_detection = \
            laundry_det_for_flat_users(item_input_object, item_output_object, weekend_energy_delta,
                                       weekday_energy_delta, max_consumption, input_data, logger)

    # Detected laundry if dwelling type is independent house

    elif dwelling_type == "house":
        laundry_detection = \
            laundry_det_for_independent_home_users(item_input_object, item_output_object, weekend_energy_delta,
                                                   weekday_energy_delta, max_consumption, logger)


    return laundry_detection


def laundry_det_for_flat_users(item_input_object, item_output_object, weekend_energy_delta,  weekday_energy_delta, max_consumption, input_data, logger):

    """
    Detect laundry for the flat users

       Parameters:
           item_input_object                (dict)               : Dict containing all hybrid inputs
           item_output_object               (dict)               : Dict containing all hybrid outputs
           weekday_energy_delta             (np.ndarray)         : weekday time stamp level energy delta
           weekend_energy_delta             (np.ndarray)         : weekend time stamp level energy delta
           max_consumption                  (int)                : max consumption of the user
           input_data                       (np.ndarray)         : input data of the user
           logger                           (logger)             : logger object

       Returns:
           laundry_detection                (int)                : True if laundry is present for the user
    """

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # giving laundry detection if laundry coverage is 0 or 100
    coverage = hybrid_config.get("coverage")

    pilot = item_input_object.get("config").get("pilot_id")
    activity_curve = item_input_object.get("activity_curve")
    occupants_count = item_output_object.get("occupants_profile").get("occupants_count")
    samples_per_hour = int(input_data.shape[1] / Cgbdisagg.HRS_IN_DAY)

    config = get_detection_config(item_input_object.get('pilot_level_config'), pilot)

    activity_curve_diff = np.nan_to_num(np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    cov_thres_for_flat_users = config.get('cov_thres_for_flat_users')
    act_prof_thres_for_flat_users = config.get('act_prof_thres_for_flat_users')

    # removing laundry detection where coverage is less

    if coverage < cov_thres_for_flat_users:
        laundry_detection = 0

    # else checking laundry detection based on activity profile of the user

    elif occupants_count > config.get("occ_limit"):

        laundry_detection = (not (np.any(np.array(weekday_energy_delta) > config.get("flat_high_occ_delta")) or
                                  np.any(np.array(weekend_energy_delta) > config.get("flat_high_occ_delta")))) or \
                            (max_consumption < config.get("flat_high_occ_cons"))

        laundry_detection = int(not laundry_detection)

    else:
        laundry_detection = (np.percentile(input_data, 95) * samples_per_hour < config.get("flat_low_occ_cap")) and \
                            (max_consumption < config.get("flat_low_occ_cons"))

        laundry_detection = laundry_detection and (
            not (np.any(np.array(weekday_energy_delta) > config.get("flat_low_occ_delta")) or
                 np.any(np.array(weekend_energy_delta) > config.get("flat_low_occ_delta"))))

        laundry_detection = int(not laundry_detection)

    if activity_curve_diff < act_prof_thres_for_flat_users:
        laundry_detection = 0

    logger.info("Detected laundry if dwelling type is flat type, %d", laundry_detection)

    return laundry_detection


def laundry_det_for_independent_home_users(item_input_object, item_output_object, weekend_energy_delta,
                                           weekday_energy_delta, max_consumption, logger):

    """
    Detect laundry for the independent home user

       Parameters:
           item_input_object                (dict)               : Dict containing all hybrid inputs
           item_output_object               (dict)               : Dict containing all hybrid outputs
           weekday_energy_delta             (np.ndarray)         : weekday time stamp level energy delta
           weekend_energy_delta             (np.ndarray)         : weekend time stamp level energy delta
           max_consumption                  (int)                : max consumption of the user
           logger                           (logger)             : logger object

       Returns:
           laundry_detection                (int)                : True if laundry is present for the user
    """

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # giving laundry detection if laundry coverage is 0 or 100
    coverage = hybrid_config.get("coverage")

    pilot = item_input_object.get("config").get("pilot_id")
    activity_curve = item_input_object.get("activity_curve")
    occupants_count = item_output_object.get("occupants_profile").get("occupants_count")

    config = get_detection_config(item_input_object.get('pilot_level_config'), pilot)

    activity_curve_diff = np.nan_to_num(np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    cov_thres_for_ind_home_users = config.get('cov_thres_for_ind_home_users')
    act_prof_thres_for_ind_home_users = config.get('act_prof_thres_for_ind_home_users')
    cov_thres_for_lower_user_count = config.get('cov_thres_for_lower_user_count')
    cov_thres_for_higher_user_count = config.get('cov_thres_for_higher_user_count')

    # giving laundry detection where coverage is high

    if coverage > cov_thres_for_ind_home_users:
        laundry_detection = 1
        logger.info('pilot laundry coverage input is high | ')

    # removing laundry detection where coverage is less and occupants count is also less

    elif (coverage <= cov_thres_for_higher_user_count and occupants_count < 3) or \
            (coverage <= cov_thres_for_lower_user_count and occupants_count < 2):
        laundry_detection = 0

    elif coverage <= cov_thres_for_higher_user_count and occupants_count >= 3:
        limit = max(0, 1500 - 30 * coverage)
        laundry_detection = max(np.max(np.array(weekend_energy_delta)), np.max(np.array(weekday_energy_delta))) > limit

    else:
        # else checking laundry detection based on activity profile of the user

        laundry_detection = int(not (max_consumption < config.get("house_cap")))

    if activity_curve_diff < act_prof_thres_for_ind_home_users:
        laundry_detection = 0

    logger.info("Detected laundry if dwelling type is independent house, %d", laundry_detection)

    return laundry_detection


def laundry_det_for_users_with_dwelling_type_unknown(item_input_object, item_output_object, weekend_energy_delta,
                                                     weekday_energy_delta, max_consumption, logger):

    """
    Detect laundry for the user

       Parameters:
           item_input_object                (dict)               : Dict containing all hybrid inputs
           item_output_object               (dict)               : Dict containing all hybrid outputs
           weekday_energy_delta             (np.ndarray)         : weekday time stamp level energy delta
           weekend_energy_delta             (np.ndarray)         : weekend time stamp level energy delta
           max_consumption                  (int)                : max consumption of the user
           logger                           (logger)             : logger object

       Returns:
           laundry_detection                (int)                : True if laundry is present for the user
    """

    hybrid_config = get_hybrid_config(item_input_object.get('pilot_level_config'))

    # giving laundry detection if laundry coverage is 0 or 100
    coverage = hybrid_config.get("coverage")

    pilot = item_input_object.get("config").get("pilot_id")
    activity_curve = item_input_object.get("activity_curve")
    occupants_count = item_output_object.get("occupants_profile").get("occupants_count")
    ao_cons = item_input_object.get("item_input_params").get("ao_cons")

    config = get_detection_config(item_input_object.get('pilot_level_config'), pilot)

    activity_curve_diff = np.nan_to_num(np.percentile(activity_curve, 97) - np.percentile(activity_curve, 3))

    cov_thres = config.get('cov_thres')
    act_prof_thres = config.get('act_prof_thres')
    cov_thres_for_user_count = config.get('cov_thres_for_user_count')
    user_count_thres = config.get('user_count_thres')
    lower_cov_thres_for_user_count = config.get('lower_cov_thres_for_user_count')
    energy_thres_for_high_user_count = config.get('energy_thres_for_high_user_count')

    # giving laundry detection where coverage is very low

    if coverage <= cov_thres[0]:
        laundry_detection = 0
        logger.info('pilot laundry coverage input is less | ')

    # giving laundry detection where coverage is high

    elif coverage > cov_thres[1]:
        laundry_detection = 1

    elif coverage < lower_cov_thres_for_user_count and occupants_count >= user_count_thres:
        laundry_detection = \
            not (not (np.max(np.array(weekday_energy_delta)) < energy_thres_for_high_user_count) or
                 not (np.max(np.array(weekend_energy_delta)) < energy_thres_for_high_user_count))

    # removing laundry detection where coverage is less and occupants count is also less

    elif coverage < cov_thres_for_user_count and occupants_count < user_count_thres:
        laundry_detection = 0

    else:

        # else checking laundry detection based on activity profile of the user

        laundry_detection = (ao_cons < config.get("not_known_ao"))
        laundry_detection = laundry_detection or max_consumption < config.get("not_known_cons")
        laundry_detection = laundry_detection and \
                            not (np.any(np.array(weekday_energy_delta) > config.get("not_known_delta")) or
                                 np.any(np.array(weekend_energy_delta) > config.get("not_known_delta")))

        laundry_detection = int(not laundry_detection)

        if activity_curve_diff < act_prof_thres:
            laundry_detection = 0

    logger.info("Detected laundry if dwelling type is not known, %d", laundry_detection)

    return laundry_detection
