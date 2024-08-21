
"""
Author - Mayank Sharan/Nisha Agarwal
Date - 4th April 2021
list of utils function to prepare or fetch hybrid v2 hsm attributes
"""

# Import python packages

import numpy as np

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.itemization.aer.functions.hsm_utils import check_validity_of_hsm

from python3.itemization.aer.raw_energy_itemization.get_final_consumption.init_final_item_config import init_final_item_conf


def update_wh_hsm(final_tou_consumption, wh_idx, length, item_input_object, item_output_object, logger):

    """
    Update wh hsm

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        wh_idx                    (int)           : wh appliance index
        length                    (int)           : count of non-vac days
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger                    (logger)        : logger object

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    # wh cons is monthly consumption of step1 or step 2 wh addition
    # wh backup cons is monthly consumption of step3 addition

    created_hsm = dict({
        'wh_cons': 0,
        'wh_backup_cons': 0
    })

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    # updating WH HSM

    vacation = np.logical_not(item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool))

    if np.any(vacation > 0):

        created_hsm['wh_cons'] = ((np.sum(final_tou_consumption[wh_idx][vacation]) / np.sum(vacation)) * scaling_factor)
        created_hsm['wh_backup_cons'] = ((np.sum(final_tou_consumption[wh_idx][vacation]) / np.sum(vacation)) * (scaling_factor))

        # updating wh cons attribute with consumption of the days for which results will be posted in incremental run

        if item_input_object.get('config').get('disagg_mode') == 'incremental':

            bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]

            app_output_for_target_days_in_inc_run = \
                final_tou_consumption[wh_idx][vacation][np.isin(bc_list[vacation], item_input_object.get('out_bill_cycles_by_module').get('disagg_bc')[:, 0])]
            created_hsm['wh_cons'] = ((np.sum(app_output_for_target_days_in_inc_run) / len(app_output_for_target_days_in_inc_run)) * scaling_factor)

    else:
        created_hsm['wh_cons'] = 0
        created_hsm['wh_backup_cons'] = 0

        # checking whether to update wh hsm

    logger.info('WH estimate being posted in hsm is | %s', created_hsm.get('wh_cons'))
    logger.info('WH step3 estimate being posted in hsm is | %s', created_hsm.get('wh_backup_cons'))

    config = init_final_item_conf().get('hsm_config')

    # updating wh hsm key with new hybrid v2 wh hsm attributes

    days_count = len(final_tou_consumption[0])
    disagg_mode = item_input_object.get('config').get('disagg_mode')

    update_hsm_condition = ((disagg_mode == 'historical' and days_count >= config.get('hist_hsm_min_days')) or
                            (disagg_mode == 'incremental' and days_count >= config.get('inc_hsm_min_days')))

    if update_hsm_condition and (item_output_object.get('created_hsm').get('wh') is None):
        item_output_object['created_hsm']['wh'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    if update_hsm_condition and (item_output_object.get('created_hsm').get('wh') is not None) and \
            (item_output_object.get('created_hsm').get('wh').get('attributes') is not None):
        item_output_object['created_hsm']['wh']['attributes'].update(created_hsm)

    return item_output_object


def update_ev_hsm(final_tou_consumption, ev_idx, length, item_input_object, item_output_object, logger):

    """
    Update ev hsm

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        ev_idx                    (int)           : ev appliance index
        length                    (int)           : count of non-vac days
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger                    (logger)        : logger object

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    # ev cons is monthly consumption of step1 or step 2 ev addition
    # ev backup cons is monthly consumption of step3 addition

    created_hsm = dict({
        'ev_cons': 0,
        'ev_backup_cons': 0
    })

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    vacation = np.logical_not(item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool))

    if np.any(vacation > 0):

        created_hsm['ev_cons'] = ((np.sum(final_tou_consumption[ev_idx][vacation]) / np.sum(vacation)) * scaling_factor)
        created_hsm['ev_backup_cons'] = ((np.sum(final_tou_consumption[ev_idx][vacation]) / np.sum(vacation)) * scaling_factor)

        # updating ev cons attribute with consumption of the days for which results will be posted in incremental run

        if item_input_object.get('config').get('disagg_mode') == 'incremental':

            bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
            app_output_for_target_days_in_inc_run = \
                final_tou_consumption[ev_idx][vacation][np.isin(bc_list[vacation], item_input_object.get('out_bill_cycles_by_module').get('disagg_bc')[:, 0])]

            created_hsm['ev_cons'] = ((np.sum(app_output_for_target_days_in_inc_run) / len(app_output_for_target_days_in_inc_run)) * scaling_factor)

    else:
        created_hsm['ev_cons'] = 0

    logger.info('EV estimate being posted in hsm is | %s', (created_hsm.get('ev_cons')))
    logger.info('EV step3 estimate being posted in hsm is | %s', (created_hsm.get('ev_backup_cons')))

    config = init_final_item_conf().get('hsm_config')

    days_count = len(final_tou_consumption[0])
    disagg_mode = item_input_object.get('config').get('disagg_mode')

    update_hsm_condition = ((disagg_mode == 'historical' and days_count >= config.get('hist_hsm_min_days')) or
                            (disagg_mode == 'incremental' and days_count >= config.get('inc_hsm_min_days')))

    if update_hsm_condition and (item_output_object.get('created_hsm').get('ev') is None):
        item_output_object['created_hsm']['ev'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    # update ev hsm with ev postprocessing attributes

    if update_hsm_condition and (item_output_object.get('created_hsm').get('ev') is not None) and \
            (item_output_object.get('created_hsm').get('ev').get('attributes') is not None):
        item_output_object['created_hsm']['ev']['attributes'].update(created_hsm)

    return item_output_object


def update_pp_hsm(output_data, final_tou_consumption, pp_idx, length, item_input_object, item_output_object, logger):
    """
    Update pp hsm

    Parameters:
        output_data               (np.ndarray)    : disagg output data for all users
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        pp_idx                    (int)           : pp appliance index
        length                    (int)           : count of non-vac days
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger                    (logger)        : logger object

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    # pp cons is monthly consumption of step1 or step 2 pp addition
    # pp backup cons is monthly consumption of step3 addition
    # item_extend_flag is true if PP consumption will be extended in the next disagg run, keeping tou same as current run
    # item_extend_tou is tou of PP of current disagg run

    samples_per_hour = int(final_tou_consumption.shape[2] / Cgbdisagg.HRS_IN_DAY)

    created_hsm = dict({
        'pp_cons': 0,
        'pp_backup_cons': 0,
        'item_extend_flag': 0,
        'item_extend_tou': np.zeros(int(samples_per_hour * Cgbdisagg.HRS_IN_DAY))
    })

    if np.sum(final_tou_consumption[pp_idx]) > 0:
        extend_pp, extend_pp_tou = \
            preapare_pp_extension_flag_for_hsm(final_tou_consumption, pp_idx, item_input_object, item_output_object, output_data, samples_per_hour)

    else:
        extend_pp = 0
        extend_pp_tou = np.zeros(int(samples_per_hour * Cgbdisagg.HRS_IN_DAY))

    config = init_final_item_conf().get('hsm_config')

    days_count = len(final_tou_consumption[0])
    disagg_mode = item_input_object.get('config').get('disagg_mode')

    update_hsm_condition = ((disagg_mode == 'historical' and days_count >= config.get('hist_hsm_min_days')) or
                            (disagg_mode == 'incremental' and days_count >= config.get('inc_hsm_min_days')))

    vacation = np.logical_not(item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool))

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    created_hsm['pp_cons'] = ((np.sum(final_tou_consumption[pp_idx]) / len(vacation)) * scaling_factor)

    created_hsm['pp_backup_cons'] = ((np.sum(final_tou_consumption[pp_idx]) / len(vacation)) * scaling_factor)

    # updating pp cons attribute with consumption of the days for which results will be posted in incremental run

    if item_input_object.get('config').get('disagg_mode') == 'incremental':

        bc_list = item_input_object.get("item_input_params").get("input_data")[Cgbdisagg.INPUT_BILL_CYCLE_IDX, :, 0]
        app_output_for_target_days_in_inc_run = \
            final_tou_consumption[pp_idx][np.isin(bc_list, item_input_object.get('out_bill_cycles_by_module').get('disagg_bc')[:, 0])]

        created_hsm['pp_cons'] = ((np.sum(app_output_for_target_days_in_inc_run) / len(app_output_for_target_days_in_inc_run)) * (scaling_factor))

    created_hsm['item_extend_flag'] = float(extend_pp)
    created_hsm['item_extend_tou'] = extend_pp_tou.astype(int)

    logger.info('PP estimate being posted in hsm is | %s', created_hsm.get('pp_cons'))
    logger.info('PP step3 estimate being posted in hsm is | %s', created_hsm.get('pp_backup_cons'))
    logger.info('PP extension flag | %s', created_hsm.get('item_extend_flag'))

    if update_hsm_condition and (item_output_object.get('created_hsm').get('pp') is None):
        item_output_object['created_hsm']['pp'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    # updating pp hsm key with new hybrid v2 pp hsm attributes

    if update_hsm_condition and (item_output_object.get('created_hsm') is not None) and \
            (item_output_object.get('created_hsm').get('pp') is not None) and \
            (item_output_object.get('created_hsm').get('pp').get('attributes') is not None):
        item_output_object['created_hsm']['pp']['attributes'].update(created_hsm)

    value = final_tou_consumption[pp_idx]

    amp = 0

    if np.sum(value) > 0:
        amp = np.percentile(value[value > 0], 90) * samples_per_hour

    if update_hsm_condition and (item_output_object.get('created_hsm') is not None) and \
            (item_output_object.get('created_hsm').get('pp') is not None) and \
            (item_output_object.get('created_hsm').get('pp').get('attributes') is not None):
        item_output_object['created_hsm']['pp']['attributes']['final_item_amp'] = int(amp)

    return item_output_object


def preapare_pp_extension_flag_for_hsm(final_tou_consumption, pp_idx, item_input_object, item_output_object, output_data, samples_per_hour):

    """
    Preparing pp extension flag that determines whether PP will be extension in next disagg run, based on current run TOU

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        pp_idx                    (int)           : pp appliance index
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        output_data               (np.ndarray)    : disagg output data for all users
        samples_per_hour          (int)           : samples in an hour

    Returns:
        extend_pp                 (int)           : PP extension flag
        extend_pp_tou             (np.ndarray)    : TOU of current PP run
    """

    pp_cons = final_tou_consumption[pp_idx]

    # checking whether the current pp output has multiple amplitudes,
    # it is done by either checking variable speed
    # or checking prepare of multiple amplitudes in pp schedules

    variable_speed_bool = 0

    config = init_final_item_conf().get('post_processing_config')

    perc_cap_for_multi_amp_pp_flag = config.get('perc_cap_for_multi_amp_pp_flag')
    min_diff_for_multi_amp_pp_flag = config.get('min_diff_for_multi_amp_pp_flag')
    min_detection_conf = config.get('min_detection_conf')
    max_pp_usage_hours = config.get('max_pp_usage_hours')
    min_pp_days_required = config.get('min_pp_days_required')

    multi_mode_amp_pp_flag = (np.percentile(pp_cons[pp_cons > 0], perc_cap_for_multi_amp_pp_flag[1]) -
                              np.percentile(pp_cons[pp_cons > 0], perc_cap_for_multi_amp_pp_flag[0])) > \
                             min_diff_for_multi_amp_pp_flag / samples_per_hour

    pp_hsm_info_present = \
        (item_input_object.get('created_hsm') is not None) and \
        (item_input_object.get('created_hsm').get('pp') is not None) and \
        (item_input_object.get('created_hsm').get('pp').get('attributes') is not None) and\
        (item_input_object.get('created_hsm').get('pp').get('attributes').get('run_type_code') is not None)

    if pp_hsm_info_present:
        # determine if the pp is variable speed pp or 2 pp with different amplitudes
        variable_speed_bool = \
            item_input_object.get('created_hsm').get('pp').get('attributes').get('run_type_code')[0] == 3
        variable_speed_bool = variable_speed_bool or multi_mode_amp_pp_flag

    elif np.sum(pp_cons) > 0:
        variable_speed_bool = multi_mode_amp_pp_flag

    #  This part determines whether PP consumption should be extended in the next disagg run, keeping tou same as current run
    # This extension is done inorder to handle PP underestimation cases in multiple runs
    # the conditions are :
    # pp usage hours are less than a certain limit
    # PP usage days is more than 90 days
    # detection confidence is atleast 0.75
    # and if current has pp usage only in summers, PP only be added in summer months in the next disagg run as well

    pp_detection_confidence = 1

    if item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') is not None:
        pp_detection_confidence = (item_input_object.get('disagg_special_outputs').get('pp_hybrid_confidence') / 100)

    pp_added_from_hybrid = \
        (item_output_object.get("timed_app_dict") is not None) and\
        (item_output_object.get("timed_app_dict").get('pp') is not None) and \
        (np.sum(item_output_object.get("timed_app_dict").get("pp")) > 0) and np.sum(output_data[pp_idx]) == 0

    if pp_added_from_hybrid:
        pp_detection_confidence = 1

    extend_pp_tou = np.sum(pp_cons > 0, axis=0) > (((np.sum(pp_cons > 0, axis=1) > 0).sum()) * 0.9)

    extend_pp = np.sum(extend_pp_tou) >= samples_per_hour
    extend_pp = extend_pp and (np.sum(extend_pp_tou) <= max_pp_usage_hours * samples_per_hour)
    extend_pp = extend_pp and (((np.sum(pp_cons > 0, axis=1) > 0).sum()) > min_pp_days_required)
    extend_pp = extend_pp and (pp_detection_confidence >= min_detection_conf and variable_speed_bool == 0)

    season = item_output_object.get("season")

    # given 0.5 score to pp extension flag if PP is being used only in summers
    # this is to communicate PP behavior to next disagg run

    pp_usage_only_in_summers = (extend_pp > 0) and np.any(season < -0.5) and ((pp_cons[season <= -1].sum(axis=1)) > 0).sum() < 40

    if pp_usage_only_in_summers:
        extend_pp = 0.5

    return extend_pp, extend_pp_tou


def ev_hsm_post_process(final_tou_consumption, item_input_object, output_data, ev_idx, length):

    """
    postprocess mtd output based on ev hsm info

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        output_data               (np.ndarray)    : ts level disagg output
        ev_idx                    (int)           : ev appliance index
        length                    (int)           : count of non-vac days

    Returns:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
    """

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    valid_ev_hsm = item_input_object.get("item_input_params").get('valid_ev_hsm')

    # if disagg mode is mtd, update ev output based on hsm output

    valid_hsm_flag = check_validity_of_hsm(valid_ev_hsm, item_input_object.get("item_input_params").get('ev_hsm'),
                                           'ev_cons')

    if valid_hsm_flag:

        hsm_ev = item_input_object.get("item_input_params").get('ev_hsm').get('ev_cons')

        if hsm_ev is not None and isinstance(hsm_ev, list):
            hsm_ev = hsm_ev[0]

        if (item_input_object.get('config').get('disagg_mode') == 'mtd') and (np.sum(output_data[ev_idx]) == 0):

            if hsm_ev == 0:
                final_tou_consumption[ev_idx] = 0

            # checking if mtd output is within the range of previous hist/inc run,
            # if not, it is scaled accordingly

            elif ((np.sum(final_tou_consumption[ev_idx]) / length) * scaling_factor) > hsm_ev * 4:
                factor = ((np.sum(final_tou_consumption[ev_idx]) / length) * scaling_factor) / hsm_ev * 4
                final_tou_consumption[ev_idx] = final_tou_consumption[ev_idx]/factor

    return final_tou_consumption


def post_process_based_on_wh_hsm(final_tou_consumption, item_input_object, wh_idx):

    """
    postprocess mtd output based on wh hsm

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        item_input_object         (dict)          : Dict containing all hybrid inputs
        wh_idx                    (int)           : wh idx

    Returns:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        hsm_wh                    (int)           : wh hsm info
    """

    valid_wh_hsm = item_input_object.get("item_input_params").get('valid_wh_hsm')

    hsm_wh = 0

    valid_hsm_flag = check_validity_of_hsm(valid_wh_hsm, item_input_object.get("item_input_params").get('wh_hsm'),
                                           'wh_backup_cons')

    # killing itemization wh output in mtd, if HSM wh output is 0

    if valid_hsm_flag:

        hsm_wh = item_input_object.get("item_input_params").get('wh_hsm').get('wh_backup_cons')

        if hsm_wh is not None and isinstance(hsm_wh, list):
            hsm_wh = hsm_wh[0]

        if (item_input_object.get('config').get('disagg_mode') == 'mtd') and hsm_wh == 0:
            final_tou_consumption[wh_idx] = 0

    return final_tou_consumption, hsm_wh


def update_stat_app_hsm(final_tou_consumption, item_output_object, item_input_object, ld_idx, ent_idx, cook_idx, length, logger):

    """
    Update stat app consumption in hsm

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        item_output_object        (dict)          : Dict containing all hybrid outputs
        item_input_object         (dict)          : Dict containing all hybrid inputs
        ld_idx                    (int)           : laundry index in app seq
        ent_idx                   (int)           : ent index in app seq
        cook_idx                  (int)           : cook index in app seq
        length                    (int)           : count of non-vac days
        logger                    (logger)        : logger object
    Returns:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    config = init_final_item_conf().get('hsm_config')

    # adding cooking/laundry/entertainment consumption in hsm, for stability in multiple incremental runs of the user

    created_hsm = dict({
        'cook_cons': 0,
        'ent_cons': 0,
        'ld_cons': 0,
        'wh_cons': 0
    })

    days_in_month = Cgbdisagg.DAYS_IN_MONTH

    # preparing new HSM attributes for stat appliances

    ld_cons = ((np.sum(final_tou_consumption[ld_idx]) / length) * (days_in_month / Cgbdisagg.WH_IN_1_KWH))
    ent_cons = ((np.sum(final_tou_consumption[ent_idx]) / length) * (days_in_month / Cgbdisagg.WH_IN_1_KWH))
    cook_cons = ((np.sum(final_tou_consumption[cook_idx]) / length) * (days_in_month / Cgbdisagg.WH_IN_1_KWH))

    if length == 0:
        cook_cons = 0
        ent_cons = 0
        ld_cons = 0

    logger.info('laundry estimate being posted in hsm is | %s', int(ld_cons))
    logger.info('entertainment estimate being posted in hsm is | %s', int(ent_cons))
    logger.info('Cooking estimate being posted in hsm is | %s', int(cook_cons))

    created_hsm['cook_cons'] = int(cook_cons)
    created_hsm['ent_cons'] = int(ent_cons)
    created_hsm['ld_cons'] = int(ld_cons)

    days_count = len(final_tou_consumption[0])
    disagg_mode = item_input_object.get('config').get('disagg_mode')

    # updating attributes into existing HSM key

    update_hsm_condition = ((disagg_mode == 'historical' and days_count >= config.get('hist_hsm_min_days')) or
                            (disagg_mode == 'incremental' and days_count >= config.get('inc_hsm_min_days')))

    if update_hsm_condition and (item_output_object.get('created_hsm').get('li') is None):
        item_output_object['created_hsm']['li'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    if update_hsm_condition and (item_output_object.get('created_hsm').get('li') is not None) and \
            (item_output_object.get('created_hsm').get('li').get('attributes') is not None):
        item_output_object['created_hsm']['li']['attributes'].update(created_hsm)

    return final_tou_consumption, item_output_object


def get_backup_app_hsm(item_input_object, backup_app, logger):

    """
    Fetched required hsm attributes for appliances (pp/ev/wh) is added through step 3 addition

    Parameters:
        item_input_object           (dict)          : Dict containing all hybrid inputs
        backup_app                  (list)          : list of all appliances for which output is given through step 3 addition
        logger                      (logger)        : logger object

    Returns:
        valid_hsm_flag             (np.ndarray)    : list of flag for each appliance, where the flag denotes
                                                    whether the hsm can be used for the given appliance
        hsm_output                 (np.ndarray)     : list of required HSM output for each appliance
    """

    hsm_output = np.zeros(len(backup_app))
    valid_hsm_flag = np.zeros(len(backup_app))

    app_idx = 0

    for app in backup_app:

        valid_wh_hsm = item_input_object.get("item_input_params").get('valid_' + app + '_hsm')

        if valid_wh_hsm and item_input_object.get("item_input_params").get(app + '_hsm') is not None and \
                item_input_object.get("item_input_params").get(app + '_hsm').get(app + '_backup_cons') is not None:

            app_hsm_input = item_input_object.get("item_input_params").get(app + '_hsm').get(app + '_backup_cons')

            if app_hsm_input is not None and isinstance(app_hsm_input, list):
                app_hsm_input = app_hsm_input[0]

            if item_input_object.get('config').get('disagg_mode') in ['incremental', 'mtd']:
                valid_hsm_flag[app_idx] = 1
                hsm_output[app_idx] = app_hsm_input

            app_idx = app_idx + 1

    logger.info('step 3 appliances HSM output | %s', hsm_output)
    logger.info('step 3 appliances HSM output flag | %s', valid_hsm_flag)

    return valid_hsm_flag, hsm_output


def update_ref_hsm(final_tou_consumption, ref_idx, length, item_input_object, item_output_object, logger):

    """
    Update wh hsm

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        wh_idx                    (int)           : wh appliance index
        length                    (int)           : count of non-vac days
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger                    (logger)        : logger object

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    created_hsm = dict({
        'ref_backup_cons': 0
    })

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    # updating ref HSM

    vacation = np.logical_not(item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool))

    created_hsm['ref_backup_cons'] = ((np.sum(final_tou_consumption[ref_idx]) / len(vacation)) * (scaling_factor))

    logger.info('Ref step3 estimate being posted in hsm is | %s', created_hsm.get('ref_backup_cons'))

    config = init_final_item_conf().get('hsm_config')

    # updating wh hsm key with new hybrid v2 ref hsm attributes

    days_count = len(final_tou_consumption[0])
    disagg_mode = item_input_object.get('config').get('disagg_mode')

    update_hsm_condition = ((disagg_mode == 'historical' and days_count >= config.get('hist_hsm_min_days')) or
                            (disagg_mode == 'incremental' and days_count >= config.get('inc_hsm_min_days')))

    if update_hsm_condition and (item_output_object.get('created_hsm').get('ref') is None):
        item_output_object['created_hsm']['ref'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    if update_hsm_condition and (item_output_object.get('created_hsm').get('ref') is not None) and \
            (item_output_object.get('created_hsm').get('ref').get('attributes') is not None):
        item_output_object['created_hsm']['ref']['attributes'].update(created_hsm)

    return item_output_object


def update_li_hsm(final_tou_consumption, li_idx, length, item_input_object, item_output_object, logger):

    """
    Update wh hsm

    Parameters:
        final_tou_consumption     (np.ndarray)    : ts level itemization output
        wh_idx                    (int)           : wh appliance index
        length                    (int)           : count of non-vac days
        item_input_object         (dict)          : Dict containing all hybrid inputs
        item_output_object        (dict)          : Dict containing all hybrid outputs
        logger                    (logger)        : logger object

    Returns:
        item_output_object        (dict)          : Dict containing all hybrid outputs
    """

    created_hsm = dict({
        'li_backup_cons': 0
    })

    scaling_factor = (Cgbdisagg.DAYS_IN_MONTH / Cgbdisagg.WH_IN_1_KWH)

    # updating li HSM

    vacation = np.logical_not(item_input_object.get("item_input_params").get("vacation_data").sum(axis=1).astype(bool))

    if np.any(vacation > 0):
        created_hsm['li_backup_cons'] = ((np.sum(final_tou_consumption[li_idx][vacation]) / np.sum(vacation)) * (scaling_factor))

    else:
        created_hsm['li_backup_cons'] = 0

    logger.info('li step3 estimate being posted in hsm is | %s', created_hsm.get('li_backup_cons'))

    config = init_final_item_conf().get('hsm_config')

    # updating wh hsm key with new hybrid v2 li hsm attributes

    days_count = len(final_tou_consumption[0])
    disagg_mode = item_input_object.get('config').get('disagg_mode')

    update_hsm_condition = ((disagg_mode == 'historical' and days_count >= config.get('hist_hsm_min_days')) or
                            (disagg_mode == 'incremental' and days_count >= config.get('inc_hsm_min_days')))

    if update_hsm_condition and (item_output_object.get('created_hsm').get('li') is None):
        item_output_object['created_hsm']['li'] = {
            'timestamp': item_input_object.get('input_data')[-1, Cgbdisagg.INPUT_EPOCH_IDX],
            'attributes': dict()
        }

    if update_hsm_condition and (item_output_object.get('created_hsm').get('li') is not None) and \
            (item_output_object.get('created_hsm').get('li').get('attributes') is not None):
        item_output_object['created_hsm']['li']['attributes'].update(created_hsm)

    return item_output_object
