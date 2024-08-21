"""
Author - Nikhil Singh Chauhan / Paras Tehria
Date - 15-May-2020
Module to get EV detection confidence
"""

# Import python packages

import logging
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.disaggregation.aer.ev.functions.detection.get_monthly_box_feat import get_monthly_box_feat
from python3.disaggregation.aer.ev.functions.detection.get_monthly_box_feat import get_seasonal_box_feat
from python3.disaggregation.aer.ev.functions.detection.get_monthly_box_feat import get_seasonal_hod_diff
from python3.disaggregation.aer.ev.functions.detection.get_monthly_box_feat import remove_boxes_before_ev
from python3.disaggregation.aer.ev.functions.detection.get_monthly_box_feat import get_charging_frequency
from python3.disaggregation.aer.ev.functions.detection.seasonality_checks import multiseason_check, winter_seasonality_check


def home_level_detection(debug, ev_config, logger_base):
    """
    Function to get features for EV detection

        Parameters:
            debug                     (dict)              : Dict containing hsm_in, bypass_hsm, make_hsm
            ev_config                  (dict)              : Module config dict
            logger_base               (logger)            : logger base

        Returns:
            debug                     (object)            : Updated debug dict

    """
    # Taking new logger base for this module

    logger_local = logger_base.get('logger').getChild('home_level_detection')

    logger_pass = {'logger': logger_local,
                   'logging_dict': logger_base.get('logging_dict')}

    # Specific logger fo this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    region = ev_config.get('region')

    recent_ev = 0

    det_config = ev_config.get('detection')
    model_features = ev_config.get('detection_model_features')

    # Extract the EV model from debug object

    # If ev app profile is yes, use the high recall model
    if debug.get('ev_app_profile_yes'):
        hld_model = debug.get('models', {}).get('ev_hld', {}).get('xgb', {}).get(region)
        feature_type = 'xgb'
    else:
        hld_model = debug.get('models', {}).get('ev_hld', {}).get('rf', {}).get(region)
        feature_type = 'rf'

    if not hld_model:
        logger.warning("EV detection model not found, skipping detection | ")
        debug['ev_hld'] = 0
        debug['ev_probability'] = 0.0000
        debug['recent_ev'] = recent_ev

        return debug

    logger.info("Model used for EV detection | {}".format(feature_type))

    # Use the complete data features

    user_features = deepcopy(debug.get('user_features'))

    user_features_df = pd.DataFrame(user_features, index=[0])

    user_model_features = model_features[feature_type][region]

    user_features_df = user_features_df[user_model_features]

    # Replace other nan with zero

    user_features_df = user_features_df.replace([np.nan, np.inf, -np.inf], 0)

    # Logging features values

    for feature in user_model_features:
        feature_value = user_features[feature]

        logger.info('Feature {} | {}'.format(feature, feature_value))

    ev_probability = hld_model.predict_proba(user_features_df)[0, 1]

    ev_probability = conf_drop_case(ev_probability, user_features, user_features_df, hld_model, ev_config, debug,
                                    logger)

    logger.info('EV L2 detection probability | {}'.format(ev_probability))

    ev_hld = int(ev_probability >= det_config['detection_conf_thresh'])

    # If not found use the recent features

    if (ev_hld == 0) and (ev_probability >= det_config.get('recent_detection_conf_thresh')) and (
            debug.get('user_recent_features') is not None):
        logger.info('EV L2 probability significant enough for recent check | ')

        user_features = deepcopy(debug['user_recent_features'])

        user_features_df = pd.DataFrame(user_features, index=[0])

        user_features_df = user_features_df[user_model_features]

        # Replace other nan with zero

        user_features_df = user_features_df.replace([np.nan, np.inf, -np.inf], 0)
        # Logging features values

        for feature in user_model_features:
            feature_value = user_features[feature]

            logger.info('Feature {} | {}'.format(feature, feature_value))

        recent_ev_probability = hld_model.predict_proba(user_features_df)[0, 1]

        recent_ev_hld = int(recent_ev_probability > det_config.get('detection_conf_thresh'))

        logger.info('Recent EV L2 detection probability | {}'.format(recent_ev_probability))

    else:
        recent_ev_hld = 0
        recent_ev_probability = np.nan
        logger.info('EV L2 probability not significant enough for recent check | ')

    # Check whether EU model needs to be run or not for Duke & Avista users

    ev_probability, recent_ev_probability = check_eu_model(region, ev_hld, recent_ev_hld, ev_probability,
                                                           recent_ev_probability, ev_config, debug, logger)

    # Replace original EV probability only if recent is higher than threshold

    if recent_ev_probability > ev_probability and recent_ev_probability > det_config.get('detection_conf_thresh'):
        ev_probability = recent_ev_probability
        recent_ev = 1

    logger.info('Model output EV L2 detection probability | {}'.format(ev_probability))
    debug['model_probability'] = ev_probability

    ev_hld = int(ev_probability >= det_config['detection_conf_thresh'])

    if ev_hld == 1:
        ev_hld, ev_probability = hld_checks(ev_hld, ev_probability, debug, recent_ev, logger_pass, ev_config)

    # Handling the detection fluctuations
    hsm_in = debug.get('hsm_in', {})
    previous_hld = None
    if hsm_in is not None:
        previous_hld = debug.get('hsm_in', {}).get('ev_hld')

    logger.info("For mean ev probability using latest value from HSM- Prev HSM: {}".format(debug.get('hsm_in')))

    if ev_hld == 0 and debug.get('disagg_mode') != 'historical' and previous_hld is not None and previous_hld[-1] == 1:
        logger.info("EV hld is zero in the current run but one in the past run, taking mean probability as final")
        ev_probability = np.mean([ev_probability, debug.get('hsm_in').get('ev_probability')[-1]])

    ev_hld = int(ev_probability >= det_config['detection_conf_thresh'])

    logger.info('EV L2 home level detection | {}'.format(ev_hld))
    logger.info('EV L2 detection probability | {}'.format(ev_probability))

    # Save the probabilities and hld for EV L2 detection

    debug['ev_hld'] = ev_hld
    debug['l2']['ev_hld'] = ev_hld
    debug['ev_probability'] = round(float(ev_probability), 4)
    debug['l2']['ev_probability'] = round(float(ev_probability), 4)
    debug['recent_ev'] = recent_ev

    if ev_hld == 1:
        debug['charger_type'] = 'L2'
    else:
        debug['charger_type'] = 'None'

    return debug


def conf_drop_case(ev_probability, user_features, user_features_df, hld_model, ev_config, debug, logger):
    """
    Function used to update the EV probability in case the user falls in any of the seasonal feature skewing"
    Parameters:
        ev_probability                  (float)             : Current ev probability
        user_features                   (dict)              : User features dictionary
        user_features_df                (Dataframe)         : User features data frame
        hld_model                       (model)             : HLD model
        ev_config                       (dict)              : EV configurations dictionary
        debug                           (dict)              : Debug dictionary
        logger                          (Logger)            : Logger
    Returns:
        ev_probability                  (float)             : Updated ev probability
    """

    # Extract the required variables
    case_1 = debug['feature_case_1']
    case_2 = debug['feature_case_2']
    case_3 = debug['feature_case_3']
    case_user = case_1 | case_2 | case_3
    prev_prob_thr = ev_config.get('prev_prob_thr')
    feature_1_case_2_value = ev_config.get('feature_1_case_2_value')
    feature_2_case_2_value = ev_config.get('feature_2_case_2_value')
    feature_1_case_1_3_value = ev_config.get('feature_1_case_1_3_value')
    feature_2_case_1_3_value = ev_config.get('feature_2_case_1_3_value')

    ev_probability = np.round(ev_probability, 4)

    # Check for HSM availability
    hsm_in = debug.get('hsm_in', {})
    run_user = False
    previous_ev_probability = None
    if (hsm_in is not None) and (hsm_in.get('ev_probability') is not None):
        previous_ev_probability = debug.get('hsm_in').get('ev_probability')[-1]
        run_user = True

    # Check for confidence conditions
    if run_user and (previous_ev_probability < prev_prob_thr and ev_probability < prev_prob_thr):
        run_user = False

    # Run if the HSM is available and the probability conditions satisfy
    if run_user and case_user:

        # Update the feature values for Case 1 & Case 3
        if case_1 or case_3:
            logger.info('User falling under Case 1 or Case 3 |')
            seasonal_count_fraction_diff = feature_1_case_1_3_value
            seasonal_energy_fraction_diff = feature_2_case_1_3_value
            user_features['seasonal_count_fraction_diff'] = np.round(seasonal_count_fraction_diff, 3)
            user_features['seasonal_energy_fraction_diff'] = np.round(seasonal_energy_fraction_diff, 3)
            user_features['seasonal_count_fraction_diff'] = 1 / user_features['seasonal_count_fraction_diff']
            user_features['seasonal_energy_fraction_diff'] = 1 / user_features['seasonal_energy_fraction_diff']
            logger.info(
                'Updated Feature value - seasonal_count_fraction_diff | {}'.format(seasonal_count_fraction_diff))
            logger.info(
                'Updated Feature value - seasonal_energy_fraction_diff | {}'.format(seasonal_energy_fraction_diff))

        # Update the feature values for Case 2
        if case_2:
            logger.info('User falling under Case 2 |')
            seasonal_count_fraction_diff = feature_1_case_2_value
            seasonal_energy_fraction_diff = feature_2_case_2_value
            user_features['seasonal_count_fraction_diff'] = np.round(seasonal_count_fraction_diff, 3)
            user_features['seasonal_energy_fraction_diff'] = np.round(seasonal_energy_fraction_diff, 3)
            user_features['seasonal_count_fraction_diff'] = 1 / user_features['seasonal_count_fraction_diff']
            user_features['seasonal_energy_fraction_diff'] = 1 / user_features['seasonal_energy_fraction_diff']
            logger.info(
                'Updated Feature value - seasonal_count_fraction_diff | {}'.format(seasonal_count_fraction_diff))
            logger.info(
                'Updated Feature value - seasonal_energy_fraction_diff | {}'.format(seasonal_energy_fraction_diff))

        user_features_df['seasonal_count_fraction_diff'] = user_features['seasonal_count_fraction_diff']
        user_features_df['seasonal_energy_fraction_diff'] = user_features['seasonal_energy_fraction_diff']

        ev_probability = hld_model.predict_proba(user_features_df)[0, 1]

    return ev_probability


def hld_checks(hld, probability, debug, recent_ev, logger_base, ev_config):
    """
    Function to apply basic rules to minimize False Positives

        Parameters:
            hld                       (int)               : Home level detection
            probability               (float)              : Confidence of EV detection
            recent_ev                 (int)               : Recent EV detected or not
            debug                     (dict)              : Debug object containing useful info regarding EV run
            logger_base               (logger)            : logger base
            ev_config                  (dict)              : Module config dict

        Returns:
            debug                     (object)            : Updated debug dict

    """
    logger_local = logger_base.get('logger').getChild('hld_checks')

    # Specific logger for this function

    logger = logging.LoggerAdapter(logger_local, logger_base.get('logging_dict'))

    updated_box_keys = [key.split('_')[-1] for key in list(debug.keys()) if 'updated_box_data' in key]

    keys_int = list(map(int, updated_box_keys))
    max_key = str(np.max(keys_int))

    factor = debug.get('factor')

    box_data = deepcopy(debug.get('updated_box_data_' + max_key))
    box_features = deepcopy(debug.get('updated_box_features_' + max_key))

    if recent_ev:
        box_data = deepcopy(debug.get('recent_detection_box_data'))
        box_features = deepcopy(debug.get('recent_detection_box_features'))

    box_data, box_features, last_ev_index = remove_boxes_before_ev(box_data, box_features, debug, ev_config)
    # Getting month-on-month stability features

    # first ev month time; adding 86400 to tackle timezone alignment issues
    first_ev_month_ts = np.min(box_data[:, 0]) + Cgbdisagg.SEC_IN_DAY
    first_ev_month = datetime.datetime.utcfromtimestamp(first_ev_month_ts).strftime('%d-%b-%Y')

    last_ev_month_ts = np.max(
        box_data[box_data[:, ev_config['box_data_month_col']] <= last_ev_index, Cgbdisagg.INPUT_EPOCH_IDX])
    last_ev_month = datetime.datetime.utcfromtimestamp(last_ev_month_ts).strftime('%d-%b-%Y')
    logger.info(" First EV month: , Last EV month: | {}, {}".format(first_ev_month, last_ev_month))

    # Getting month-on-month stability features

    box_monthly_count_var, box_monthly_presence_var = get_monthly_box_feat(box_data, box_features, debug, ev_config)
    logger.info("Monthly count variation: , Monthly presence variation: | {}, {}".format(box_monthly_count_var,
                                                                                         box_monthly_presence_var))

    # Getting seasonal stability features

    box_seasonal_count_var, seasonal_boxes_frac = get_seasonal_box_feat(box_data, box_features, debug, ev_config)
    logger.info("Seasonal count variation: , Seasonal box fraction: | {}, {}".format(box_seasonal_count_var,
                                                                                     seasonal_boxes_frac))

    # Checking if boxes detected contain prominently HVAC boxes

    ac_day_bool, sh_night_bool, ac_sh_hod_dist, prom_smr_hrs, prom_wtr_hrs = get_seasonal_hod_diff(box_features,
                                                                                                   ev_config)
    logger.info("AC day bool: , SH night bool: | {}, {}".format(ac_day_bool, sh_night_bool))

    # Checking if there anomalous charging behavior

    charging_freq, charges_per_day, frac_multi_charge = get_charging_frequency(box_features, last_ev_index, factor,
                                                                               ev_config)
    logger.info("Charging Frequency: , Charges per day: | {}, {}".format(charging_freq, charges_per_day))

    # Adding post processing parameters to debug dict
    debug['first_ev_month'] = first_ev_month
    debug['last_ev_month'] = last_ev_month

    debug['box_monthly_count_var'] = box_monthly_count_var
    debug['box_monthly_presence_var'] = box_monthly_presence_var
    debug['box_seasonal_count_var'] = np.round(box_seasonal_count_var, 2)
    debug['seasonal_boxes_frac'] = np.round(seasonal_boxes_frac, 2)

    debug['ac_day_bool'] = ac_day_bool
    debug['sh_night_bool'] = sh_night_bool
    debug['prom_smr_hrs'] = prom_smr_hrs
    debug['prom_wtr_hrs'] = prom_wtr_hrs
    debug['ac_sh_hod_dist'] = ac_sh_hod_dist

    debug['charging_freq'] = charging_freq
    debug['charges_per_day'] = charges_per_day
    debug['frac_multi_charge'] = frac_multi_charge

    # Getting post-processing config
    det_post_process_config = ev_config['detection_post_processing']

    # Getting useful parameters from post-processing config
    monthly_count_var_thresh = det_post_process_config['monthly_count_var_thresh']
    max_probability_post_process = det_post_process_config['max_probability_post_process']
    max_transition_frac_hvac = det_post_process_config['max_transition_frac_hvac']
    charging_freq_thresh = det_post_process_config['charging_freq_thresh']
    charges_per_day_thresh = det_post_process_config['charges_per_day_thresh']
    min_charging_percent = det_post_process_config['min_charging_percent']
    new_seasonal_fp_probability = ev_config.get('new_seasonal_fp_probability')

    new_probability = det_post_process_config['new_probability']

    # Indices of summer, winter, and transition seasons
    wtr_idx = det_post_process_config['wtr_idx']
    transition_idx = det_post_process_config['transition_idx']
    smr_idx = det_post_process_config['smr_idx']

    # If any of the monthly stability, seasonal stability or charging behavior is anomalous setting conf to 0.4

    if box_monthly_count_var > monthly_count_var_thresh and probability <= max_probability_post_process:
        logger.info('Month Variation check failed, setting EV probability to 0.4 | ')
        probability = new_probability
        hld = 0

    elif seasonal_boxes_frac[transition_idx] < max_transition_frac_hvac and \
            (seasonal_boxes_frac[transition_idx] <= seasonal_boxes_frac[smr_idx]) and \
            (seasonal_boxes_frac[transition_idx] <= seasonal_boxes_frac[wtr_idx]) and sh_night_bool and ac_day_bool:
        logger.info('Captured HVAC boxes, setting EV probability to 0.4 | ')
        probability = new_probability
        hld = 0

    elif debug['charging_freq'] > charging_freq_thresh or charges_per_day > charges_per_day_thresh:
        logger.info('Charging frequency too high, setting EV probability to 0.4 | ')
        probability = new_probability
        hld = 0

    # Total EV charging check (300 KW)

    input_data = deepcopy(debug['input_data'])

    box_data = deepcopy(debug['features_box_data'])

    input_data_energy = np.sum(input_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])
    detection_boxes_energy = np.sum(box_data[:, Cgbdisagg.INPUT_CONSUMPTION_IDX])

    boxes_energy_percent = 100 * (detection_boxes_energy / input_data_energy)

    if (boxes_energy_percent < min_charging_percent) and (probability < max_probability_post_process):
        probability = new_probability
        hld = 0

        logger.info('Minimum user level charging check failed | ')
        logger.info('Input data energy | {}'.format(input_data_energy))
        logger.info('Detection boxes energy | {}'.format(detection_boxes_energy))
        logger.info('Detection boxes percentage | {}'.format(boxes_energy_percent))

    # Highly seasonal FPs check
    if hld != 0:
        hvac_dominance_true, seasonal_var, seasonal_proportion_var = multiseason_check(box_data, ev_config, debug,
                                                                                       probability)

        if hvac_dominance_true:
            probability = new_seasonal_fp_probability
            hld = 0

            logger.info('Highly seasonal user detected | ')
            logger.info('Multiple charging & Seasonal count variation | {}'.format(seasonal_var))
            logger.info('Seasonal EV proportion variation | {}'.format(seasonal_proportion_var))

    # NSP Winter False positive due to HVAC & WH check

    if hld != 0 and ev_config.get('pilot_id') in ev_config.get('nsp_winter_seasonality_configs').get('wtr_seasonal_pilots'):

        winter_device, debug = winter_seasonality_check(box_data, ev_config, debug)

        if winter_device:
            probability = new_seasonal_fp_probability
            hld = 0

            logger.info('Winter WH/HVAC device FP detected in EV, changing the detection to 0 | ')

    return hld, probability


def eu_model_continuity(debug):
    """
    Function to identify users to continue to run in EU model
    Parameters:
        debug                   (dict)                  : Debug dictionart
    Returns:
        run_eu_model_hsm        (int)                   : Value of 'run_eu_model_hsm' key in the EV hsm
    """

    # Identify whether to run the EU model or check the confidence score criteria

    run_eu_model_hsm = 0

    # Note :- Once a user starts being detected from the EU model, it's needed to have the EU model to be run for the
    # user in the subsequent runs. This information is stored in the HSM with the key name - 'run_eu_model'

    # Either identify if the user has been run with the eu model in the previous runs or check with the confidence
    # score condition

    ev_hsm = debug.get('hsm_in')
    if ev_hsm is not None and len(ev_hsm) > 0:
        use_eu_model = ev_hsm.get('run_eu_model')
        if (use_eu_model is not None) and (use_eu_model[0] == 1):
            run_eu_model_hsm = 1

    return run_eu_model_hsm


def check_eu_model(region, ev_hld, recent_ev_hld, ev_probability, recent_ev_probability, ev_config, debug, logger):
    """
    This function is used to check if the user needs to be run for EU model or not
    Parameters:
        region                  (str)           : User region
        ev_hld                  (int)           : EV detection status
        recent_ev_hld           (int)           : Recent EV detection status
        ev_probability          (float)         : EV probability
        recent_ev_probability   (float)         : Recent EV probability
        ev_config               (dict)          : Contains all the EV configurations
        debug                   (dict)          : Debug Dictionary
        logger                  (Logger)        : Logger
    Returns:
        ev_probability          (float)         : EV probability
        recent_ev_probability   (float)         : Recent EV probability
    """

    # Check whether this is a user with EU model continuity

    run_eu_model_hsm_new = 0
    final_worthy_bool = False
    run_eu_model_hsm = eu_model_continuity(debug)

    # Check if the EU model needs to be run or not

    worthy_bool_1 = (region == 'NA') and (ev_hld == 0) and (recent_ev_hld == 0) and \
                    (ev_config['pilot_id'] in ev_config['na_to_eu_model']['na_to_eu_model_pilots'])

    worthy_bool_2 = ev_probability > ev_config['na_to_eu_model']['na_confidence_thr']

    # Either if the hsm value is 1 or the confidence check + region conditions are True

    if run_eu_model_hsm == 1 or (worthy_bool_1 and worthy_bool_2):
        final_worthy_bool = True

    # log the hsm information
    if run_eu_model_hsm == 1:
        logger.info('EU model to be used indicated through the hsm | ')

    if final_worthy_bool:
        logger.info('Calling EU model for Year round EV check | ')
        user_features = deepcopy(debug.get('user_features'))
        ev_hld, ev_probability, run_eu_model_hsm_new = run_eu_model(debug, ev_hld, ev_probability, user_features,
                                                                    ev_config, region, run_eu_model_hsm, logger)

        # If detection is 0, run recent EV from EU model

        final_worthy_bool = (ev_probability >= ev_config.get('detection').get('recent_detection_conf_thresh')) and \
                            (recent_ev_probability >= ev_config['na_to_eu_model']['na_confidence_thr'])

        final_worthy_bool = ev_hld == 0 and (run_eu_model_hsm or final_worthy_bool)

        if final_worthy_bool:
            logger.info('Calling EU model for Recent EV check | ')
            user_features = deepcopy(debug['user_recent_features'])
            _, recent_ev_probability, run_eu_model_hsm_new = run_eu_model(debug, recent_ev_hld, recent_ev_probability,
                                                                          user_features, ev_config, region,
                                                                          run_eu_model_hsm, logger)

    # If this is a new detection from EU model then update the key, else continue with the original value

    if run_eu_model_hsm or run_eu_model_hsm_new:
        run_eu_model_hsm = 1
    debug['run_eu_model_hsm'] = run_eu_model_hsm
    return ev_probability, recent_ev_probability


def run_eu_model(debug, ev_hld, ev_probability, user_features, ev_config, model_region, run_eu_model_hsm, logger):
    """
    Update the EV probability with EU model for Avista & Duke users
    Args:
        debug                   (dict)              : Debug dictionary
        ev_hld                  (int)               : EV hld
        ev_probability
        user_features
        ev_config               (dict)              : Module config dict
        model_region            (string)            : model region
        run_eu_model_hsm        (int)               : HSM information whether to run EU model or not
        logger                  (Logger)            : Logger

    Returns:
        ev_probability          (float)             : EV probability
        recent_ev_probability   (float)             : Recent EV probability
    """

    # Extract the required variables

    if user_features is None:
        return ev_hld, ev_probability

    run_eu_model_hsm_new = 0
    weekly_count_pro = user_features['weekly_count_pro']
    energy_per_charge = user_features['energy_per_charge']
    na_probability_weight = ev_config['na_to_eu_model']['na_probability_weight']
    eu_probability_weight = ev_config['na_to_eu_model']['eu_probability_weight']
    energy_weekly_count_min = ev_config['na_to_eu_model']['energy_weekly_count_min']
    energy_weekly_count_max = ev_config['na_to_eu_model']['energy_weekly_count_max']

    # Check for a minimum criteria satisfaction

    use_eu_model_bool = ((energy_per_charge * weekly_count_pro) >= energy_weekly_count_min) and \
                        ((energy_per_charge * weekly_count_pro) <= energy_weekly_count_max)

    # If the criteria satisfies run the EU model

    if use_eu_model_bool or run_eu_model_hsm == 1:
        model_region = 'EU'
        logger.info('Running EU model to capture low amplitude and duration EV users in NA region | ')

    if model_region == 'EU':

        det_config = ev_config.get('detection')
        model_features = ev_config.get('detection_model_features')

        # Extract the EV model from debug object

        # If ev app profile is yes, use the high recall model
        if debug.get('ev_app_profile_yes'):
            hld_model = debug.get('models', {}).get('ev_hld', {}).get('xgb', {}).get(model_region)
            feature_type = 'xgb'
        else:
            hld_model = debug.get('models', {}).get('ev_hld', {}).get('rf', {}).get(model_region)
            feature_type = 'rf'

        # Use the complete data features

        user_features = deepcopy(debug.get('user_features'))

        user_features_df = pd.DataFrame(user_features, index=[0])

        user_model_features = model_features[feature_type][model_region]

        user_features_df = user_features_df[user_model_features]

        # Replace other nan with zero

        user_features_df = user_features_df.replace([np.nan, np.inf, -np.inf], 0)

        # EV probability from EU model

        ev_probability_eu = hld_model.predict_proba(user_features_df)[0, 1]

        # Weighted EV probability

        new_ev_probability = na_probability_weight * ev_probability + eu_probability_weight * ev_probability_eu

        logger.info('EU Model output EV detection probability | {}'.format(new_ev_probability))

        new_ev_hld = int(new_ev_probability >= det_config['detection_conf_thresh'])

        if new_ev_hld == 1:
            ev_hld = new_ev_hld
            ev_probability = new_ev_probability
            run_eu_model_hsm_new = 1
            logger.info('NA user detected through the EU model | {}'.format(new_ev_probability))

    return ev_hld, ev_probability, run_eu_model_hsm_new
