"""
Author - Nisha Agarwal
Date - 4th April 2021
Stage 3 of itemization framework - calculate final tou level estimation
"""
# Import python packages

import copy
import numpy as np
from sklearn.cluster import KMeans

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.config.pilot_constants import PilotConstants

from python3.itemization.init_itemization_config import seq_config

from python3.itemization.aer.functions.itemization_utils import find_seq
from python3.itemization.aer.functions.itemization_utils import fill_arr_based_seq_val_for_valid_boxes

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf

from python3.itemization.aer.raw_energy_itemization.inference_engine.config.get_inf_config import get_inf_config

from python3.itemization.aer.functions.get_hybrid_v2_generic_config import get_hybrid_v2_generic_config


def final_pp_post_process(pp_idx, pp_cons_mapping, variable_speed_bool, final_tou_consumption, item_input_object):
    """
    This function performs postprocessing on itemized pp output to maintain consistency throughout

    Parameters:
        pp_idx                    (int)           : mapping for pp output
        pp_cons_mapping           (np.ndarray)    : array containing mapping for high and low cons pp points
        variable_speed_bool       (bool)          : whether the poolpump is variable speed
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        item_input_object         (dict)          : Dict containing all hybrid inputs

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    disagg_confidence = 1

    config = init_final_item_conf().get('post_processing_config')

    pp_conf_buc_for_multi_speed_pp = config.get('pp_conf_buc_for_multi_speed_pp')
    perc_buc_for_multi_speed_pp = config.get('perc_buc_for_multi_speed_pp')

    # fetching PP detection confidence

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    if item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None:
        disagg_confidence = item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100

    # PP ts level estimate is made consistent (after adjustment) based on whether the PP is single or variable speed

    signature_based_pp_detected = np.any(final_tou_consumption[pp_idx] > 0) and (not item_input_object["item_input_params"]["backup_pp"])

    if signature_based_pp_detected:

        if not variable_speed_bool:

            median_val = \
                get_median_val_for_pp(samples_per_hour, disagg_confidence, final_tou_consumption, pp_idx,
                                      item_input_object)

            final_tou_consumption[pp_idx] = np.fmin(final_tou_consumption[pp_idx], median_val)
        else:

            perc_val = perc_buc_for_multi_speed_pp[np.digitize(disagg_confidence, pp_conf_buc_for_multi_speed_pp)]

            pp_cons = copy.deepcopy(final_tou_consumption[pp_idx])

            median_val = np.percentile(pp_cons[np.logical_and(pp_cons_mapping == 0, pp_cons > 0)], perc_val)

            pp_cons[np.logical_and(pp_cons_mapping == 0, pp_cons > 0)] = np.fmin(
                pp_cons[np.logical_and(pp_cons_mapping == 0, pp_cons > 0)], median_val)

            if np.any(np.logical_and(pp_cons_mapping == 1, pp_cons > 0)):
                median_val = np.percentile(pp_cons[np.logical_and(pp_cons_mapping == 1, pp_cons > 0)], perc_val)

                pp_cons[np.logical_and(pp_cons_mapping == 1, pp_cons > 0)] = np.fmin(
                    pp_cons[np.logical_and(pp_cons_mapping == 1, pp_cons > 0)], median_val)

            final_tou_consumption[pp_idx] = pp_cons

        final_tou_consumption[pp_idx] = np.fmin(final_tou_consumption[pp_idx], median_val)

    return final_tou_consumption


def pp_post_process(pp_idx, output_data, final_tou_consumption, item_input_object, processed_input_data,
                    disagg_confidence, logger):

    """
    This function performs postprocessing on itemized pp output to maintain consistency throughout

    Parameters:
        pp_idx                    (int)           : mapping for pp output
        output_data               (np.ndarray)    : array containing final ts level disagg output
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        processed_input_data      (np.ndarray)    : input data
        disagg_confidence         (float)         : pp detection confidence
        logger                    (logger)        : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        pp_cons_mapping           (np.ndarray)     : array containing mapping for high and low cons pp points
        variable_speed_bool       (bool)           : True if it is a variable speeed poolpump
    """

    samples_per_hour = item_input_object.get('item_input_params').get('samples_per_hour')
    variable_speed_bool = 0

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    pp_cons_mapping = np.zeros(processed_input_data.shape)

    config = init_final_item_conf().get('post_processing_config')

    pp_conf_buc_for_multi_speed_pp = config.get('pp_conf_buc_for_multi_speed_pp')
    perc_buc_for_multi_speed_pp = config.get('perc_buc_for_multi_speed_pp')
    perc_cap_for_multi_amp_pp_flag = config.get('perc_cap_for_multi_amp_pp_flag')
    min_diff_for_multi_amp_pp_flag = config.get('min_diff_for_multi_amp_pp_flag')
    min_days_to_check_min_pp_cons = config.get('min_days_to_check_min_pp_cons')

    values = output_data[pp_idx]
    multi_mode_amp_pp_flag = 0

    if np.sum(values):
        multi_mode_amp_pp_flag = (np.percentile(values[values > 0], perc_cap_for_multi_amp_pp_flag[1]) -
                                  np.percentile(values[values > 0], perc_cap_for_multi_amp_pp_flag[0])) > \
                                 min_diff_for_multi_amp_pp_flag / samples_per_hour

    signature_based_pp_detected = np.any(final_tou_consumption[pp_idx] > 0) and (not item_input_object["item_input_params"]["backup_pp"])

    if signature_based_pp_detected:

        # each timestamp of PP estimate is assigned to a high cons or low cons category based on its amplitude and type pf PP

        pp_cons_mapping, variable_speed_bool = \
            assign_pp_tags_based_on_multiple_pp_amp(pp_cons_mapping, item_input_object, output_data, pp_idx, multi_mode_amp_pp_flag)

        # Cap the pp consumption to maintain consistency through
        # this capping is done based on the disagg confidence value

        # If the pp has variable amplitudes, the capping is done separately for low and high consumption ponints

        if not variable_speed_bool:

            median_val = \
                get_median_val_for_pp(samples_per_hour, disagg_confidence, final_tou_consumption, pp_idx,
                                      item_input_object)

            logger.info('amplitude for single speed PP | %s', median_val)

            final_tou_consumption[pp_idx] = np.fmin(final_tou_consumption[pp_idx], median_val)
        else:

            perc_val = perc_buc_for_multi_speed_pp[np.digitize(disagg_confidence, pp_conf_buc_for_multi_speed_pp)]
            consumption = copy.deepcopy(final_tou_consumption[pp_idx])

            target_points = np.logical_and(pp_cons_mapping == 0, consumption > 0)

            median_val = np.percentile(consumption[target_points], perc_val)

            consumption[target_points] = np.fmin(consumption[target_points], median_val)

            target_points = np.logical_and(pp_cons_mapping == 1, consumption > 0)

            logger.info('amplitude for variable speed PP | %s', median_val)

            if np.any(target_points):

                median_val = np.percentile(consumption[target_points], perc_val)

                consumption[target_points] = np.fmin(consumption[target_points], median_val)

                logger.info('Second amplitude for variable speed PP | %s', median_val)

            final_tou_consumption[pp_idx] = consumption

            # maintaining a minimum ts level PP estimate

            pp_cons = final_tou_consumption[pp_idx]

            if np.sum(np.sum(pp_cons, axis=1) > 0) > min_days_to_check_min_pp_cons:
                value = np.percentile(pp_cons[pp_cons > 0], 10)
                additional_cons_req = np.minimum(copy.deepcopy(np.fmax(value, pp_cons) - pp_cons), other_cons_arr)
                pp_cons[pp_cons > 0] = additional_cons_req[pp_cons > 0] + pp_cons[pp_cons > 0]

                final_tou_consumption[pp_idx] = pp_cons

    return final_tou_consumption, pp_cons_mapping, variable_speed_bool


def assign_pp_tags_based_on_multiple_pp_amp(pp_cons_mapping, item_input_object, output_data, pp_idx, multi_mode_amp_pp_flag):

    """
    Each timestamp of PP estimate is assigned to a high cons or low cons category based on its amplitude and type pf PP

    Parameters:
        pp_cons_mapping           (np.ndarray)    : array containing mapping for high and low cons pp points
        item_input_object         (dict)          : Dict containing all hybrid inputs
        output_data               (np.ndarray)    : array containing final ts level disagg output
        pp_idx                    (int)           : mapping for pp output
        multi_mode_amp_pp_flag    (int)           : flag that represents whether multiple amplitude is present in pp estimates

    Returns:
        pp_cons_mapping           (np.ndarray)     : updated array containing mapping for high and low cons pp points
        variable_speed_bool       (bool)           : True if it is a variable speeed poolpump

    """

    variable_speed_bool = 0

    values = output_data[pp_idx]

    if ((item_input_object.get('config').get('disagg_mode')) == 'mtd') and np.any(values > 0):
        variable_speed_bool = multi_mode_amp_pp_flag

        if variable_speed_bool:
            kmeans_model = KMeans(n_clusters=2, random_state=0).fit(
                values[values > np.percentile(values[values > 0], 7)].flatten().reshape(-1, 1))
            consumption_level_boundry = np.mean(kmeans_model.cluster_centers_)
            pp_cons_mapping = np.zeros(output_data[1].shape)
            pp_cons_mapping[output_data[pp_idx] > consumption_level_boundry] = 1
            pp_cons_mapping[output_data[pp_idx] == 0] = -2

    if (item_input_object.get('created_hsm') is not None) and \
            (item_input_object.get('created_hsm').get('pp') is not None) and np.sum(output_data[pp_idx]):

        # determine if the pp is variable speed pp or 2 pp with different amplitudes

        variable_speed_bool = \
            item_input_object.get('created_hsm').get('pp').get('attributes').get('run_type_code')[0] == 3

        values = output_data[pp_idx]
        variable_speed_bool = variable_speed_bool or multi_mode_amp_pp_flag

        # if it is true , map the pp usage , and divide them into low ad high consumption points

        if variable_speed_bool:
            kmeans_model = KMeans(n_clusters=2, random_state=0).fit(
                values[values > np.percentile(values[values > 0], 7)].flatten().reshape(-1, 1))
            consumption_level_boundry = np.mean(kmeans_model.cluster_centers_)
            pp_cons_mapping = np.zeros(output_data[1].shape)
            pp_cons_mapping[output_data[pp_idx] > consumption_level_boundry] = 1
            pp_cons_mapping[output_data[pp_idx] == 0] = -2

    return pp_cons_mapping, variable_speed_bool


def get_median_val_for_pp(samples_per_hour, disagg_confidence, final_tou_consumption, pp_idx, item_input_object):

    """
    This function performs postprocessing on itemized pp output to maintain consistency throughout

    Parameters:
        samples_per_hour          (int)           : samples count in an hour
        disagg_confidence         (float)         : disagg confidence
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        pp_idx                    (int)           : mapping for pp output
        item_input_object         (dict)          : Dict containing all hybrid inputs

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    pp_values = final_tou_consumption[pp_idx][final_tou_consumption[pp_idx] > 0]

    config = init_final_item_conf().get('post_processing_config')

    pp_conf_buc_for_single_speed_pp = config.get('pp_conf_buc_for_single_speed_pp')
    perc_buc_for_single_speed_pp = config.get('perc_buc_for_single_speed_pp')
    min_pp_cons_for_single_speed_pp = config.get('min_pp_cons_for_single_speed_pp')

    # updating PP amplitude based on the PP detection confidence

    median_val =  np.percentile(pp_values, perc_buc_for_single_speed_pp[np.digitize(disagg_confidence, pp_conf_buc_for_single_speed_pp)])

    median_val = max(min_pp_cons_for_single_speed_pp / samples_per_hour, median_val)

    valid_pp_hsm = item_input_object.get("item_input_params").get('valid_pp_hsm')
    pp_amp_data = median_val

    valid_hsm_flag = check_validity_of_hsm(valid_pp_hsm, item_input_object.get("item_input_params").get('pp_hsm'), 'final_item_amp')

    # updating PP amplitude based on the PP HSM information to maintain ts level consistency in incremental/mtd mode

    valid_hsm = valid_hsm_flag and item_input_object.get("item_input_params").get('pp_hsm').get('final_item_amp')[0] > 0

    if valid_hsm:
        pp_hsm = item_input_object.get("item_input_params").get('pp_hsm').get('final_item_amp')[0]

        if pp_hsm is None:
            pp_amp_data = 0
        elif isinstance(pp_hsm, list):
            pp_amp_data = pp_hsm[0]
        else:
            pp_amp_data = pp_hsm

    if valid_pp_hsm and valid_hsm:
        median_val = np.minimum(median_val, pp_amp_data)
    elif (item_input_object.get("config").get('disagg_mode') == 'incremental') and valid_hsm:
        median_val = np.minimum(median_val, pp_amp_data * 1.5)

    return median_val


def post_process_based_on_pp_hsm(final_tou_consumption, item_input_object, output_data, pp_idx, length):
    """
    Postprocess final pp consumption based on hsm, to ensure consistent pp output wihtin the runs

    Parameters:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        output_data               (dict)          : disagg ts level output
        pp_idx                    (int)           : mapping for pp output
        length                    (int)           : count of non vacation days

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    pp_hsm = item_input_object.get("item_input_params").get('pp_hsm')

    valid_pp_hsm = item_input_object.get("item_input_params").get('valid_pp_hsm')

    scaling_factor_based_on_days_in_month = Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH

    valid_hsm_flag = check_validity_of_hsm(valid_pp_hsm, pp_hsm, 'pp_cons')

    # If the user is in mtd mode, and PP estimates are added from hybrid,
    # mtd pp output is scaled based on previous runs information to maintain consistency

    if valid_hsm_flag:

        hsm_pp_cons = pp_hsm.get('pp_cons')

        if hsm_pp_cons is not None and isinstance(hsm_pp_cons, list):
            hsm_pp_cons = hsm_pp_cons[0]

        if (item_input_object.get('config').get('disagg_mode') == 'mtd') and (np.sum(output_data[pp_idx]) == 0):

            overestimated_cons_in_mtd_mode = \
            ((np.sum(final_tou_consumption[pp_idx]) / length) * scaling_factor_based_on_days_in_month) > hsm_pp_cons * 1.1

            if hsm_pp_cons == 0:
                final_tou_consumption[pp_idx] = 0

            elif overestimated_cons_in_mtd_mode:
                factor = ((np.sum(final_tou_consumption[pp_idx]) / length) * scaling_factor_based_on_days_in_month) / hsm_pp_cons * 1.1
                final_tou_consumption[pp_idx] = final_tou_consumption[pp_idx] / np.fmax(1, factor)

    return final_tou_consumption


def ev_post_process(ev_idx, item_input_object, processed_input_data, output_data, final_tou_consumption, logger):

    """
    This function performs postprocessing on itemized ev output to maintain consistency throughout

    Parameters:
        ev_idx                    (int)           : mapping for ev output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        processed_input_data      (np.ndarray)    : input data
        output_data               (np.ndarray)    : array containing final ts level disagg output
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    samples = int(final_tou_consumption[ev_idx].shape[1]/Cgbdisagg.HRS_IN_DAY)

    # postprocess final ev output to maintain required maximum or minimum consumption

    if np.any(final_tou_consumption[ev_idx] > 0) and (item_input_object["item_input_params"]["backup_ev"] == 0):

        if not np.sum(output_data[ev_idx]) == 0:
            max_cons = np.percentile(final_tou_consumption[ev_idx][final_tou_consumption[ev_idx] > 0], 90)
            min_cons = np.percentile(final_tou_consumption[ev_idx][final_tou_consumption[ev_idx] > 0], 30)
            if (item_input_object.get('created_hsm') is not None) and (
                    item_input_object.get('created_hsm').get('ev') is not None) \
                    and (item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude") is not None):
                amp = item_input_object.get('created_hsm').get('ev').get('attributes').get("ev_amplitude")
                max_cons = amp * 1.1
        else:
            max_cons = np.percentile(final_tou_consumption[ev_idx][final_tou_consumption[ev_idx] > 0], 70)
            min_cons = np.percentile(final_tou_consumption[ev_idx][final_tou_consumption[ev_idx] > 0], 40)

        logger.info('Max ev ts level consumption | %s', max_cons)
        logger.info('Min ev ts level consumption | %s', min_cons)

        additional_ev_cons = copy.deepcopy(np.fmax(final_tou_consumption[ev_idx], min_cons) - final_tou_consumption[ev_idx])

        additional_ev_cons = np.minimum(additional_ev_cons, other_cons_arr)

        additional_ev_cons = additional_ev_cons - 500/samples

        additional_ev_cons = np.fmax(0, additional_ev_cons)

        final_tou_consumption[ev_idx][final_tou_consumption[ev_idx] > 0] = \
            additional_ev_cons[final_tou_consumption[ev_idx] > 0] + final_tou_consumption[ev_idx][final_tou_consumption[ev_idx] > 0]

        final_tou_consumption[ev_idx] = np.fmin(final_tou_consumption[ev_idx], max_cons)

        final_tou_consumption[ev_idx] = np.minimum(final_tou_consumption[ev_idx], processed_input_data)

    return final_tou_consumption


def twh_post_process(wh_idx, heat_idx, other_cons_arr, output_data, final_tou_consumption, item_input_object):

    """
    This function performs postprocessing on itemized twh output to maintain consistency throughout

    Parameters:
        wh_idx                    (int)           : mapping for wh output
        heat_idx                  (int)           : mapping for heating output
        other_cons_arr            (np.ndarray)    : leftover residual
        output_data               (np.ndarray)    : array containing final ts level disagg output
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        item_input_object         (dict)          : Dict containing all hybrid inputs

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    samples_per_hour = int(final_tou_consumption.shape[2] / Cgbdisagg.HRS_IN_DAY)

    heating_output = final_tou_consumption[heat_idx]
    original_wh = output_data[wh_idx]

    config = get_inf_config().get("wh")

    if np.any(final_tou_consumption[wh_idx] > 0) and item_input_object.get("item_input_params").get("timed_wh_user"):

        # maintaining a minimum consumption based on the distribution of twh output

        median_val = np.percentile(final_tou_consumption[wh_idx][final_tou_consumption[wh_idx] > 0], config.get('twh_min_cons_perc'))

        median_val = np.percentile(final_tou_consumption[wh_idx][final_tou_consumption[wh_idx] > 0], config.get('twh_min_cons_perc'))

        median_val = max(config.get('twh_min_cons') / samples_per_hour, median_val)

        ts = np.logical_and(output_data[wh_idx] > median_val, final_tou_consumption[wh_idx] > 0)

        diff  = copy.deepcopy(np.fmax(final_tou_consumption[wh_idx], median_val) - final_tou_consumption[wh_idx])

        diff = np.fmax(0, diff)

        diff = np.minimum(diff, other_cons_arr)

        diff = np.fmax(0, diff)

        final_tou_consumption[wh_idx][ts] = diff[ts] + final_tou_consumption[wh_idx][ts]

        median_val = np.percentile(final_tou_consumption[wh_idx][final_tou_consumption[wh_idx] > 0], config.get('twh_min_cons_perc'))

        median_val = max(config.get('twh_min_cons') / samples_per_hour, median_val)

        # maintaining a minimum consumption based on the distribution of disagg twh output
        # and picking up leftover consumption fromm heating, to maintain wh seasonality

        ts = np.logical_and(output_data[wh_idx] > median_val, final_tou_consumption[wh_idx] > 0)

        diff  = copy.deepcopy(np.fmax(final_tou_consumption[wh_idx], median_val) - final_tou_consumption[wh_idx])

        diff = np.fmax(0, diff)

        diff = np.minimum(diff, heating_output)
        diff = np.minimum(diff, original_wh-final_tou_consumption[wh_idx])

        diff = np.fmax(0, diff)

        final_tou_consumption[wh_idx][ts] = diff[ts] + final_tou_consumption[wh_idx][ts]
        final_tou_consumption[heat_idx][ts] = final_tou_consumption[heat_idx][ts] - diff[ts]

        # maintaining a maximum consumption based on the distribution of disagg twh output

        median_val = np.percentile(final_tou_consumption[wh_idx][final_tou_consumption[wh_idx] > 0], config.get('twh_max_cons_perc'))

        ts = np.logical_and(output_data[wh_idx] > 0, final_tou_consumption[wh_idx] > 0)

        final_tou_consumption[wh_idx][ts] = np.fmin(final_tou_consumption[wh_idx][ts], median_val)

    return final_tou_consumption


def allot_thin_pulse_boxes(item_input_object, item_output_object, final_tou_consumption, appliance_list, box_seq, logger):

    """
    This function performs postprocessing on itemized wh output to add leftover thin pulses

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        appliance_list            (list)          : list of appliances
        box_seq                   (np.darray)     : residual boxes info
        logger                    (logger)        : logger info

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    wh_idx = np.where(appliance_list == 'wh')[0][0]
    original_wh = copy.deepcopy(final_tou_consumption[wh_idx])

    box_seq = box_seq.astype(int)

    samples_per_hour = int(final_tou_consumption[0].shape[1] / Cgbdisagg.HRS_IN_DAY)

    processed_input_data = item_output_object.get('original_input_data')

    other_cons_arr = processed_input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

    # check whether to add thin pulses based on type of wh

    add_wh = check_wh_addition_bool(item_input_object, final_tou_consumption, samples_per_hour, appliance_list)

    if not add_wh:
        logger.info('Not adding leftover thin pulse | ')
        return final_tou_consumption

    config = get_inf_config().get('wh')

    thin_pulse_amp_max_thres = config.get('thin_pulse_amp_max_thres')
    thin_pulse_amp_min_thres = config.get('thin_pulse_amp_min_thres')
    thin_pulse_amp_max_ts_cons = config.get('thin_pulse_amp_max_ts_cons')
    thin_pulse_amp_buffer = config.get('thin_pulse_amp_buffer')
    thin_pulse_max_amp_factor = config.get('thin_pulse_max_amp_factor')

    seq_label = seq_config.SEQ_LABEL
    seq_len = seq_config.SEQ_LEN

    max_amp = thin_pulse_amp_max_thres / samples_per_hour
    min_amp = thin_pulse_amp_min_thres / samples_per_hour
    val = thin_pulse_amp_max_ts_cons

    # determining amplitude based in disagg thin pulse box consumption

    if (item_input_object.get("item_input_params").get("final_thin_pulse") is not None) \
            and (np.sum(final_tou_consumption[wh_idx]) > 0)\
            and (item_input_object.get("item_input_params").get("final_thin_pulse").sum() > 0):

        thin_pulse = item_input_object.get("item_input_params").get("final_thin_pulse")
        thin_pulse_tou = thin_pulse > 0

        if np.sum(thin_pulse_tou) > 0:
            max_amp = np.max(thin_pulse[thin_pulse_tou]) * thin_pulse_max_amp_factor
            min_amp = np.median(thin_pulse[thin_pulse_tou]) - thin_pulse_amp_buffer/samples_per_hour

            val = max_amp

            if np.isnan(val):
                val = thin_pulse_amp_max_ts_cons
                max_amp = thin_pulse_amp_max_thres / samples_per_hour
                min_amp = thin_pulse_amp_min_thres / samples_per_hour
        else:
            val = thin_pulse_amp_max_ts_cons
            max_amp = thin_pulse_amp_max_thres / samples_per_hour
            min_amp = thin_pulse_amp_min_thres / samples_per_hour

    # picking boxes that can be assigned to thin pulse category of WH
    # based on amplitude and length of boxes

    valid_boxes = np.logical_and(box_seq[:, seq_label] == 1,
                                 np.logical_and(box_seq[:, seq_len] <= max(1, 0.5 * samples_per_hour),
                                                np.logical_and(box_seq[:, 4] >= min_amp, box_seq[:, 4] <= max_amp)))

    valid_idx = np.zeros(np.size(final_tou_consumption[0]))

    valid_idx = fill_arr_based_seq_val_for_valid_boxes(box_seq, valid_boxes, valid_idx, 1, 1)

    valid_idx = np.reshape(valid_idx, final_tou_consumption[0].shape)
    valid_idx[valid_idx > 0] = valid_idx[valid_idx > 0] + other_cons_arr[valid_idx > 0]

    valid_idx = np.fmin(valid_idx, val)

    valid_idx[valid_idx < min_amp] = 0

    season = item_output_object.get("season")

    reduce_cons = np.mean(season) > 0 and item_input_object.get('config').get('disagg_mode') == 'mtd'

    valid_idx = valid_idx * (0.6 * reduce_cons + 1 * (not reduce_cons))

    valid_idx[final_tou_consumption[wh_idx] > 0] = 0

    valid_idx_tou = limit_count_for_thin_pulse(valid_idx)

    valid_idx[valid_idx_tou == 0] = 0

    if item_input_object.get('item_input_params').get('swh_hld') == 0:
        final_tou_consumption[wh_idx] = final_tou_consumption[wh_idx] + valid_idx

        item_input_object["item_input_params"]["hybrid_thin_pulse"] = valid_idx

    swh_days = np.sum(item_output_object.get("hybrid_input_data").get("output_data")[wh_idx + 1], axis=1)

    wh_type_is_seasonal = \
        (item_output_object.get("created_hsm") is not None and item_output_object.get("created_hsm").get('wh') is not None
         and item_output_object.get("created_hsm").get('wh').get("attributes").get('swh_hld'))

    if wh_type_is_seasonal:

        if np.sum(item_output_object.get("hybrid_input_data").get("output_data")[wh_idx + 1]) == 0:
            swh_days = np.sum(original_wh, axis=1)

        final_tou_consumption[wh_idx][swh_days == 0] = 0

    return final_tou_consumption


def limit_count_for_thin_pulse(valid_idx):

    """
    This function performs checks to limit the count of thin pulse

    Parameters:
        valid_idx                 (np.ndarray)          : ts where thin pulse is being added from hybrid

    Returns:
        valid_idx_tou             (np.ndarray)          : ts where thin pulse is being added from hybrid
    """

    config = get_inf_config().get('wh')

    max_thin_pulse_count = config.get('max_thin_pulse_count')

    valid_idx_tou = valid_idx > 0

    if np.sum(valid_idx) > 0:
        for i in range(len(valid_idx)):
            if np.sum(valid_idx_tou[i]) > max_thin_pulse_count:
                extra_thin_pulse_box = int(np.sum(valid_idx_tou[i])/(np.sum(valid_idx_tou[i]) - max_thin_pulse_count))
                idx_of_thin_pulse = (np.where(valid_idx_tou[i] > 0))[0]
                idx_of_thin_pulse = idx_of_thin_pulse[np.arange(0, len(idx_of_thin_pulse), extra_thin_pulse_box)]

                valid_idx_tou[i][idx_of_thin_pulse] = 0

    return valid_idx_tou


def check_wh_addition_bool(item_input_object, final_tou_consumption, samples_per_hour, appliance_list):

    """
    This function performs checks to determine whether to add leftover thin pulses in hybrid wh output

    Parameters:
        item_input_object         (dict)          : Dict containing all hybrid inputs
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        samples_per_hour          (int)           : samples in an hour
        appliance_list            (list)          : list of appliances

    Returns:
        add_wh_thin_pulse_flag    (bool)          : True if thin pulses shud be added to wh consumption
    """

    flow_wh = 0

    wh_idx = np.where(appliance_list == 'wh')[0][0]

    config = get_inf_config().get('wh')

    wh_cons_per_cap = config.get('wh_cons_per_cap')
    min_wh_amp_for_flow_wh = config.get('min_wh_amp_for_flow_wh')

    if np.sum(final_tou_consumption[wh_idx]):
        flow_wh = np.percentile(final_tou_consumption[wh_idx][final_tou_consumption[wh_idx] > 0], wh_cons_per_cap) > \
                  min_wh_amp_for_flow_wh / samples_per_hour

    pilot = item_input_object.get("config").get("pilot_id")

    add_wh_thin_pulse_flag = final_tou_consumption[wh_idx].sum() > 0

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'), 'item_hld')

    # check whether to add wh in the mtd mode, based on hsm info

    if (item_input_object.get('config').get('disagg_mode') == 'mtd') and valid_hsm_flag:
        wh_hld = item_input_object.get("item_input_params").get('wh_hsm').get('item_hld', 0)

        if wh_hld is not None and isinstance(wh_hld, list):
            wh_hld = wh_hld[0]

        wh_type = item_input_object.get("item_input_params").get('wh_hsm').get('item_type', 2)

        if wh_type is not None and isinstance(wh_type, list):
            wh_type = wh_type[0]

        add_wh_thin_pulse_flag = wh_hld

        add_wh_thin_pulse_flag = add_wh_thin_pulse_flag and (not (wh_type == 1))

    # thin pulses shouldnt be added for twh and swh

    add_wh_thin_pulse_flag = add_wh_thin_pulse_flag and item_input_object.get('item_input_params').get('swh_hld') == 0

    add_wh_thin_pulse_flag = add_wh_thin_pulse_flag and (not item_input_object.get("item_input_params").get("timed_wh_user"))

    add_wh_thin_pulse_flag = add_wh_thin_pulse_flag and not flow_wh and (pilot not in PilotConstants.SEASONAL_WH_ENABLED_PILOTS)

    if item_input_object.get('item_input_params').get('tankless_wh') > 0:
        add_wh_thin_pulse_flag = 0

    return add_wh_thin_pulse_flag


def prepare_wh_block_bool_arr(wh_idx, item_output_object, output_data, final_tou_consumption, app_list, bc_list, vacation):

    """
    This function performs postprocessing on itemized wh output to block low consumption addition

    Parameters:
        wh_idx                    (int)           : mapping for wh output
        item_output_object        (dict)          : Dict containing all hybrid outputs
        output_data               (np.ndarray)    : array containing final ts level disagg output
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        app_list                  (np.ndarray)    : list of all target appliances
        bc_list                   (np.ndarray)    : billing cycle data of all days
        vacation                  (np.ndarray)    : vacation data of all days

    Returns:
        block_wh                 (np.ndarray)    : this array contain billing cycle level flag that denote whether to block wh in that billing cycle
    """

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    config = get_inf_config().get('wh')

    item_cons_thres = config.get('item_cons_thres_for_low_cons_wh_blocking')
    disagg_cons_thres = config.get('disagg_cons_thres_for_low_cons_wh_blocking')

    block_wh = np.zeros_like(unique_bc)

    # identify billing cycles that need to be blocked

    season_bc = np.zeros_like(unique_bc)
    season = copy.deepcopy(item_output_object.get("season"))

    app = "wh"

    monthly_cons = np.zeros(len(unique_bc))

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]

        if np.sum(target_days) < 3:
            continue

        season_bc[i] = np.sign(season[target_days].sum())

        monthly_cons[i] = np.sum(final_tou_consumption[np.where(app_list == app)[0][0]][target_days])

        disagg_output = np.sum(output_data[wh_idx][target_days]) * (Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days))

        monthly_cons[i] = monthly_cons[i] * (Cgbdisagg.DAYS_IN_MONTH / np.sum(target_days))

        vac = 1 - vacation[target_days].sum() / np.sum(target_days)

        block_wh[i] = (monthly_cons[i] < item_cons_thres * vac) and \
                      ((np.sum(output_data[wh_idx][target_days]) == 0) and
                       (disagg_output < disagg_cons_thres * vac))

    return block_wh


def wh_seasonality_check(input_data, wh_idx, final_tou_consumption, item_input_object, item_output_object):

    """
    This function performs postprocessing to maintain seasonality of itemized wh output

    Parameters:
        input_data                (np.ndarray)    : input data
        wh_idx                    (int)           : mapping for wh output
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs

    Returns:
        final_tou_consumption     (np.ndarray)     : array containing final ts level itemized output
    """

    season = copy.deepcopy(item_output_object.get("season"))
    season_copy = copy.deepcopy(season.astype(int))

    season_copy[season_copy < 0] = -1
    season_copy[season_copy >= 0] = 1

    app_idx = wh_idx
    app_cons = copy.deepcopy(final_tou_consumption[app_idx])
    original = copy.deepcopy(final_tou_consumption[app_idx])

    season_consumption = np.zeros(len(np.unique(season_copy)))

    season_list = np.unique(season_copy)

    config = get_inf_config().get('wh')

    allowed_seasonality = config.get('seasonality_thres_for_adjustment')

    for c, i in enumerate(season_list):
        season_consumption[c] = np.sum([app_cons[season_copy == i]]) / np.sum(season_copy == i)

    opp_seasonality_detected_in_storage_wh = \
        (not item_input_object.get("item_input_params").get("timed_wh_user")) and \
        (season_consumption[season_list == 1] > allowed_seasonality * season_consumption[season_list == -1])

    if opp_seasonality_detected_in_storage_wh:

        # Modify consumption ranges if reverse seasonality is detected

        seasonal_diff = (season_consumption[season_list == 1] / allowed_seasonality - season_consumption[season_list == -1]) * np.sum(
            season_copy == 1)
        seasonal_diff = seasonal_diff / np.sum(np.logical_and(app_cons[season_copy == 1] > 0,
                                                              app_cons[season_copy == 1] >
                                                              np.percentile(app_cons[season_copy == 1][app_cons[season_copy == 1] > 0], 40)))

        seasonal_diff = np.fmax(0, seasonal_diff)

        app_cons[season_copy == 1] = app_cons[season_copy == 1] - seasonal_diff

        app_cons = np.fmax(0, app_cons)

        final_tou_consumption[app_idx] = np.minimum(original, app_cons)

    # maintain consumption of thin pulsee

    thin_pulse = item_input_object.get("item_input_params").get("final_thin_pulse")

    storage_wh_user_flag = (thin_pulse is not None) and np.sum(final_tou_consumption[app_idx]) > 0 and \
            (not item_input_object.get('item_input_params').get('swh_hld')) and (not item_input_object.get("item_input_params").get("timed_wh_user"))

    if storage_wh_user_flag:

        additional_wh_req = thin_pulse - final_tou_consumption[app_idx]
        additional_wh_req = np.fmax(0, additional_wh_req)
        additional_wh_req[thin_pulse == 0] = 0

        other_cons_arr = input_data - np.sum(np.nan_to_num(final_tou_consumption), axis=0)

        additional_wh_req = np.minimum(additional_wh_req, other_cons_arr)

        final_tou_consumption[app_idx][thin_pulse > 0] = final_tou_consumption[app_idx][thin_pulse > 0] + additional_wh_req[thin_pulse > 0]

    return final_tou_consumption


def block_low_cons_wh(wh_idx, pilot, item_input_object, item_output_object, output_data, final_tou_consumption, app_list, bc_list, vacation, logger):

    """
    This function performs postprocessing on itemized wh output to block low consumption addition

    Parameters:
        wh_idx                    (int)           : mapping for wh output
        pilot                     (int)           : pilot id
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        output_data               (np.ndarray)    : array containing final ts level disagg output
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
        app_list                  (list)          : list of appliances
        bc_list                   (list)          : list of bc start timestamp
        vacation                  (np.ndarray)    : vacation data
        logger                    (logger)        : logger object

    Returns:
        final_tou_consumption     (np.ndarray)    : array containing final ts level itemized output
    """

    unique_bc, bc_size = np.unique(bc_list, return_counts=True)

    # identify billing cycles that need to be blocked

    season_bc = np.zeros_like(unique_bc)

    seq_label = seq_config.SEQ_LABEL
    seq_start = seq_config.SEQ_START
    seq_end = seq_config.SEQ_END
    seq_len = seq_config.SEQ_LEN

    app = "wh"

    block_wh = prepare_wh_block_bool_arr(wh_idx, item_output_object, output_data, final_tou_consumption, app_list, bc_list, vacation)

    # handling cases to avoid inconsistency in wh output

    seq = find_seq(block_wh, np.zeros_like(block_wh), np.zeros_like(block_wh), overnight=0)

    seq = seq.astype(int)

    for i in range(len(seq)):
        if seq[i, seq_label] == 1 and seq[i, seq_len] <= 2:
            block_wh[seq[i, seq_start]:seq[i, seq_end] + 1] = 0

    seq = find_seq(block_wh, np.zeros_like(block_wh), np.zeros_like(block_wh), overnight=0)

    seq = seq.astype(int)

    for i in range(len(seq)):
        if seq[i, seq_label] == 0 and seq[i, seq_len] <= 1:
            block_wh[seq[i, seq_start]:seq[i, seq_end] + 1] = 1

    if np.any(season_bc < 0) and (np.sum(block_wh[season_bc < 0]) > min(np.sum(season_bc < 0), 3)):
        block_wh[:] = 1

    # no blocking for swh or twh

    swh_pilots = PilotConstants.SEASONAL_WH_ENABLED_PILOTS

    min_bc_required_for_consistency_check = get_hybrid_v2_generic_config().get('min_bc_required_for_consistency_check')

    if len(unique_bc) < min_bc_required_for_consistency_check or item_input_object.get("item_input_params").get("timed_wh_user") or \
            (item_input_object.get('config').get('disagg_mode') == 'mtd') or \
            (pilot in swh_pilots) or (np.sum(final_tou_consumption[wh_idx]) == 0):
        block_wh[:] = 0

    logger.info("BCs where wh is blocked due to low consumption level | %s", block_wh)

    for i in range(len(unique_bc)):

        target_days = bc_list == unique_bc[i]
        app_idx = np.where(app_list == app)[0][0]

        final_tou_consumption[app_idx][target_days] = (1 - block_wh[i]) * final_tou_consumption[app_idx][target_days]

    return final_tou_consumption
