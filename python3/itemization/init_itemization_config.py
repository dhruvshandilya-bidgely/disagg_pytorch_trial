"""
Author - Nisha Agarwal
Date - 10/9/20
Initialize generic itemization config
"""

# Import python packages
import copy
import numpy as np
from copy import deepcopy

# Import packages from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pipeline_constants import GlobalConfigParams
from python3.itemization.initialisations.init_empty_appliance_profile import init_empty_appliance_profile


class seq_config:
    """
      Config for index description of seq output
    """
    SEQ_LABEL = 0
    SEQ_START = 1
    SEQ_END = 2
    SEQ_LEN = 3


class random_gen_config:
    """
      Config for random number generation seed
    """

    seed_value = 1234567890


def init_itemization_params():

    """
     This function initialises the necessary configurations for the itemization run
     Parameters:
         pipeline_input_object           (dict)          : Contains pipeline input object
         pipeline_output_object          (dict)          : Contains all the output object
     Returns:
         item_input_object               (dict)          : Itemization input object
     """

    # Initialize the config dictionary

    itemization_config = {}

    itemization_config.update()

    # hybrid_mod_seq: hybrid module sequences

    # score weight age values for cleanliness scoring computation

    seq_config = {
        'label': 0,
        'start': 1,
        'end': 2,
        'length': 3,
        'max_deri': 4,
        'low_perc': 5,
        'high_perc': 6,
        'net_deri': 7,
        'deri_strength': 8,
        'mid_perc': 9,
        'mid_perc_val': 50,
        'low_perc_val': 5,
        'high_perc_val': 95
    }

    # column indexes of labels sequence array

    # 'label': label of seq ,
    # 'start': start index,
    # 'end': end index ,
    # 'length': length of seq,
    # 'max_deri': maximum derivative in seq,
    # 'low_perc': low percentile value,
    # 'high_perc': high percentile value,
    # 'net_deri': net derivative,
    # 'deri_strength': strength of derivative (no od increasing points in increasing sequences),
    # 'mid_perc': mid percentile value,

    itemization_config.update({
        'seq_config': seq_config
    })

    data_preparation = {
        "outlier_perc": 95,
        "pp_amp_limit": 100
    }

    itemization_config.update({
        'data_preparation': data_preparation
    })

    enable = 0

    itemization_config.update({
        'dump_results': False,
        'dump_plots': 0,
        'dump_runtime': 0,
        'results_folder': '/var/log/bidgely/hybrid_v2/',
        'dump_disagg_plot': 0 and enable,
        'dump_item_plot': 0 and enable,
        'dump_diff_plot': 0 and enable,
        'dump_potential_plot': 0 and enable,
        'dump_reg_plot': 0 and enable,
        'dump_range_plot': 0 and enable,
        'dump_inf_plot': 0 and enable,
        'dump_new_inf_plot': 0 and enable,
        'dump_timed_app_plot': 0 and enable,
        'dump_hvac_plot': 0 and enable,
        'dump_final_plot': 1 and enable,
        'app_list':['ev'],
        "dump_sig_plot": 0 and enable,
        "dump_box_plot": 0 and enable
    })

    return itemization_config


def init_itemization_config(pipeline_input_object, pipeline_output_object, item_input_object):

    """
    This function initialises the necessary configurations for the itemization run
    Parameters:
        pipeline_input_object           (dict)          : Contains pipeline input object
        pipeline_output_object          (dict)          : Contains all the output object
    Returns:
        item_input_object               (dict)          : Itemization input object
    """

    # Initialize the config dictionary

    itemization_config = {}

    itemization_config.update(
        {'itemization_module_seq' : copy.deepcopy(GlobalConfigParams.itemization_seq)})

    # hybrid_mod_seq: hybrid module sequences

    # score weight age values for cleanliness scoring computation

    seq_config = {
        'label': 0,
        'start': 1,
        'end': 2,
        'length': 3,
        'max_deri': 4,
        'low_perc': 5,
        'high_perc': 6,
        'net_deri': 7,
        'deri_strength': 8,
        'mid_perc': 9,
        'mid_perc_val': 50,
        'low_perc_val': 5,
        'high_perc_val': 95
    }

    # column indexes of labels sequence array

    # 'label': label of seq ,
    # 'start': start index,
    # 'end': end index ,
    # 'length': length of seq,
    # 'max_deri': maximum derivative in seq,
    # 'low_perc': low percentile value,
    # 'high_perc': high percentile value,
    # 'net_deri': net derivative,
    # 'deri_strength': strength of derivative (no od increasing points in increasing sequences),
    # 'mid_perc': mid percentile value,

    itemization_config.update({
        'seq_config': seq_config
    })

    data_preparation = {
        "outlier_perc": 95,
        "pp_amp_limit": 100
    }

    # "outlier_perc": percentile used to cap high consumption outlier points

    itemization_config.update({
        'data_preparation': data_preparation
    })

    # 'dump_results': True if dumping of hybrid results is required,
    # 'results_folder': path to results
    itemization_config.update({
        'api_env': pipeline_input_object.get('global_config').get('api_env'),
        'cache_mode': pipeline_input_object.get('global_config').get('cache_mode'),
        'disagg_mode': pipeline_input_object.get('global_config').get('disagg_mode'),
        'downsample_rate': pipeline_input_object.get('global_config').get('downsample_rate'),
        'dump_csv': pipeline_input_object.get('global_config').get('dump_csv'),
        'dump_debug': pipeline_input_object.get('global_config').get('dump_debug'),
        'generate_plots': pipeline_input_object.get('global_config').get('generate_plots'),
        'pilot_id': pipeline_input_object.get('global_config').get('pilot_id'),
        'priority': pipeline_input_object.get('global_config').get('priority'),
        'run_mode': pipeline_input_object.get('global_config').get('run_mode'),
        'sampling_rate': pipeline_input_object.get('global_config').get('sampling_rate'),
        'uuid': pipeline_input_object.get('global_config').get('uuid'),
        'user_type': pipeline_input_object.get('global_config').get('user_type'),
        'write_results': pipeline_input_object.get('global_config').get('write_results'),
        'write_vip_user': pipeline_input_object.get('global_config').get('write_vip_user'),
        'disagg_aer_module_seq': pipeline_input_object.get('global_config').get('disagg_aer_seq'),
        'index': pipeline_input_object.get('global_config').get('index')
    })

    # Insert all these configurations into the itemization input object

    item_input_object['config'] = itemization_config
    item_input_object['config']['itemization_to_disagg'] = ["li", "wh", "pp", "ev", "hvac"]

    # Get disagg inputs

    item_input_object['created_hsm'] = deepcopy(pipeline_output_object.get('created_hsm'))

    # If disagg output is present, get the required data

    if pipeline_output_object.get('disagg_output_object') is not None:
        item_input_object['disagg_epoch_estimate'] = deepcopy(pipeline_output_object.get('disagg_output_object').get('epoch_estimate'))
        item_input_object['disagg_bill_cycle_estimate'] =  deepcopy(pipeline_output_object.get('disagg_output_object').get('bill_cycle_estimate'))
        item_input_object['disagg_special_outputs'] = deepcopy(pipeline_output_object.get('disagg_output_object').get('special_outputs'))
        item_input_object['disagg_appliance_profile'] = deepcopy(pipeline_output_object.get('disagg_output_object').get('appliance_profile'))
        item_input_object['disagg_output_write_idx_map'] = deepcopy(pipeline_output_object.get('disagg_output_object').get('output_write_idx_map'))
        item_input_object['disagg_data_present'] = True

    else:
        item_input_object = fill_empty_values(item_input_object)
        item_input_object['disagg_data_present'] = False

    return item_input_object


def fill_empty_values(item_input_object):
    """
    This function is used to create empty values inplace of disagg outputs
    Parameters:
        item_input_object               (dict)          : Itemization input object
    Returns:
        item_input_object               (dict)          : Itemization input object
    """

    item_input_object['disagg_output_write_idx_map'] = {}

    module_seq = item_input_object.get('config').get('disagg_aer_module_seq')
    num_modules = len(module_seq)

    # Allow extra column for HVAC. Allow 6 extra columns for SMB

    num_columns = num_modules + 1 + int('hvac' in module_seq)

    # Initialize billing cycle estimate

    out_bill_cycles = item_input_object.get('out_bill_cycles')
    bill_cycle_estimate = np.full(shape=(out_bill_cycles.shape[0], num_columns), fill_value=np.nan)
    bill_cycle_estimate[:, 0] = out_bill_cycles[:, 0]
    item_input_object['disagg_bill_cycle_estimate'] = bill_cycle_estimate

    # Initialize epoch estimate

    input_ts = item_input_object['input_data'][:, Cgbdisagg.INPUT_EPOCH_IDX]
    epoch_estimate = np.full(shape=(len(input_ts), num_columns), fill_value=np.nan)
    epoch_estimate[:, 0] = input_ts
    item_input_object['disagg_epoch_estimate'] = epoch_estimate

    # Initialize output write idx map, disagg metrics and created hsm

    write_idx = 1

    for module_name in module_seq:
        if module_name == 'hvac' or module_name == 'va':
            item_input_object['disagg_output_write_idx_map'][module_name] = [write_idx, write_idx + 1]
            write_idx += 2
        else:
            item_input_object['disagg_output_write_idx_map'][module_name] = write_idx
            write_idx += 1

    # initiate appliance profile object for all out bill cycles

    item_input_object['disagg_appliance_profile'] = dict()

    # Get default dataRange for this gbdisagg event

    appliance_data_range_start = item_input_object.get('out_bill_cycles_by_module').get('app_profile')[0][0]
    appliance_data_range_end = item_input_object.get('out_bill_cycles_by_module').get('app_profile')[-1][-1]

    for billcycle_start, billcycle_end in out_bill_cycles:
        item_input_object['disagg_appliance_profile'][int(billcycle_start)] = init_empty_appliance_profile()

        # set start and end dates for each out bill cycles

        item_input_object['disagg_appliance_profile'][int(billcycle_start)]['profileList'][0]['start'] = int(
            billcycle_start)
        item_input_object['disagg_appliance_profile'][int(billcycle_start)]['profileList'][0]['end'] = int(billcycle_end)

        # set start and end for data range for each out bill cycles

        item_input_object['disagg_appliance_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['start'] = \
            int(appliance_data_range_start)
        item_input_object['disagg_appliance_profile'][int(billcycle_start)]['profileList'][0]['dataRange']['end'] = \
            int(appliance_data_range_end)

    return item_input_object
