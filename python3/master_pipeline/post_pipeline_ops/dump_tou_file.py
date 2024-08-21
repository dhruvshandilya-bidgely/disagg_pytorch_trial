"""
Author - Paras Tehria
Date - 18th Jan 2021
Contains function to dump tou output csv to the log directory
"""

# Import python packages

import os
import numpy as np
import pandas as pd
from copy import deepcopy

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants


def dump_tou_file(pipeline_input_object, pipeline_output_object, logger_pipeline):
    """
    Function to dump tou output csv to the log directory

    Parameters:
        pipeline_input_object (dict)              : Contains all inputs required to run the pipeline
        pipeline_output_object(dict)              : Dictionary containing all outputs
        logger_pipeline     (logger)            : The logger to use here

    Returns:

    """

    dump_tou_flag = 'tou_all' in pipeline_input_object.get('global_config').get('dump_csv')

    logger_pipeline.info("Tou dump flag: | {}".format(dump_tou_flag))

    if not dump_tou_flag:
        logger_pipeline.info("Not dump TOU file | ")
        return

    root_dir = PathConstants.LOG_DIR + 'tou_disagg/'

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    tou_array = deepcopy(pipeline_output_object.get('disagg_output_object').get('epoch_estimate'))

    column_names_dict = ['epoch']

    write_idx_map = pipeline_output_object.get('disagg_output_object').get('output_write_idx_map')
    for app in write_idx_map:
        if type(write_idx_map[app]) is list:
            for i in range(len(write_idx_map[app])):
                column_names_dict.append(app + '_' + str(i + 1))
        else:
            column_names_dict.append(app)

    if 'smb' in pipeline_input_object.get('global_config').get('user_type'):
        column_names_dict.append('others')

    if pipeline_output_object.get('disagg_output_object').get('special_outputs', {}).get('timed_water_heater') is not None:
        timed_wh_estimate = pipeline_output_object.get('disagg_output_object').get('special_outputs').get('timed_water_heater')
        _, idx_mem_1, idx_mem_2 = np.intersect1d(tou_array[:, 0], timed_wh_estimate[:, Cgbdisagg.INPUT_EPOCH_IDX],
                                                 return_indices=True)
        tou_array = np.c_[tou_array, np.zeros((tou_array.shape[0],))]
        tou_array[idx_mem_1, -1] = timed_wh_estimate[idx_mem_2, Cgbdisagg.INPUT_CONSUMPTION_IDX]
    else:
        logger_pipeline.warning("Timed water heater output not available, writing zeros in the tou file | ")
        tou_array = np.c_[tou_array, np.zeros((tou_array.shape[0],))]

    column_names_dict.append('timed_wh')

    if 'hvac_output' in pipeline_output_object.get('disagg_output_object'):
        hvac_output = pipeline_output_object.get('disagg_output_object').get('hvac_output')
        tou_array = np.c_[tou_array, hvac_output]
        column_names_dict += ['temperature', 'total_consumption', 'ac_always', 'ac_demand', 'sh_always', 'sh_demand']

    if (pipeline_output_object.get('item_output_object') is not None) and \
            (pipeline_output_object.get('item_output_object').get('special_outputs', {}).get('seasonal_wh') is not None):
        seasonal_wh_estimate = pipeline_output_object.get('item_output_object').get('special_outputs').get(
            'seasonal_wh').get('ts_estimate')
        _, idx_mem_1, idx_mem_2 = np.intersect1d(tou_array[:, 0], seasonal_wh_estimate[:, 1], return_indices=True)
        tou_array = np.c_[tou_array, np.zeros((tou_array.shape[0],))]
        tou_array[idx_mem_1, -1] = seasonal_wh_estimate[idx_mem_2, 1]
    else:
        logger_pipeline.warning("Timed water heater output not available, writing zeros in the tou file | ")
        tou_array = np.c_[tou_array, np.zeros((tou_array.shape[0],))]

    column_names_dict.append('seasonal_wh')

    column_names_dict, tou_array = \
        add_postprocessing_output_and_raw_data_in_tou_dump(pipeline_input_object, pipeline_output_object, tou_array,
                                                           column_names_dict)

    tou_df = pd.DataFrame(data=tou_array, columns=column_names_dict)
    tou_df.rename(columns={'hvac_1': 'ac', 'hvac_2': 'sh'}, inplace=True)

    uuid = pipeline_input_object.get('global_config').get('uuid')
    tou_df.to_csv(root_dir + uuid + '_tou.csv', index=None)

    logger_pipeline.info("TOU csv dumping successful | ")
    return


def add_postprocessing_output_and_raw_data_in_tou_dump(pipeline_input_object, pipeline_output_object, tou_array, column_names_dict):

    """
    update appliance tou array with raw data and postprocessing module output

    Parameters:
        pipeline_input_object (dict)              : Contains all inputs required to run the pipeline
        pipeline_output_object(dict)              : Dictionary containing all outputs
        tou_array             (np.ndarray)        : appliance tou array
        column_names_dict     (list)              : lis of column names in appliance tou array

    Returns:
        tou_array             (np.ndarray)        : appliance tou array
        column_names_dict     (list)              : lis of column names in appliance tou array
    """

    if pipeline_input_object.get("input_data_without_outlier_removal") is None:
        original_input_data = deepcopy(pipeline_input_object.get("input_data"))
    else:
        original_input_data = deepcopy(pipeline_input_object.get("input_data_without_outlier_removal"))

    _, idx_mem_1, idx_mem_2 = np.intersect1d(tou_array[:, 0], original_input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], return_indices=True)
    tou_array = np.c_[tou_array, np.zeros((tou_array.shape[0],))]
    tou_array[idx_mem_1, -1] = original_input_data[idx_mem_2, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    column_names_dict.append('original_input_data')

    if pipeline_input_object.get("input_data_with_neg_and_nan") is None:
        input_data_with_neg_and_nan = deepcopy(pipeline_input_object.get("input_data"))
    else:
        input_data_with_neg_and_nan = deepcopy(pipeline_input_object.get("input_data_with_neg_and_nan"))

    _, idx_mem_1, idx_mem_2 = np.intersect1d(tou_array[:, 0], input_data_with_neg_and_nan[:, Cgbdisagg.INPUT_EPOCH_IDX], return_indices=True)
    tou_array = np.c_[tou_array, np.zeros((tou_array.shape[0],))]
    tou_array[idx_mem_1, -1] = input_data_with_neg_and_nan[idx_mem_2, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    column_names_dict.append('input_data_with_neg_nan')

    input_data = deepcopy(pipeline_input_object.get("input_data"))

    _, idx_mem_1, idx_mem_2 = np.intersect1d(tou_array[:, 0], input_data[:, Cgbdisagg.INPUT_EPOCH_IDX], return_indices=True)
    tou_array = np.c_[tou_array, np.zeros((tou_array.shape[0],))]
    tou_array[idx_mem_1, -1] = input_data[idx_mem_2, Cgbdisagg.INPUT_CONSUMPTION_IDX]

    column_names_dict.append('input_data')

    if pipeline_output_object.get('item_output_object') is not None:
        itemization_output = pipeline_output_object.get('item_output_object').get('epoch_estimate')

        tou_array = np.c_[tou_array, itemization_output]

        column_names_dict.append('epoch_post_processing')

        write_idx_map = pipeline_output_object.get('disagg_output_object').get('output_write_idx_map')
        write_idx_map['cook'] = 12
        write_idx_map['ent'] = 13
        write_idx_map['ld'] = 14

        for app in write_idx_map:
            if type(write_idx_map[app]) is list:
                for i in range(len(write_idx_map[app])):
                    column_names_dict.append(app + '_' + str(i + 1)+ '_post_processing')
            else:
                column_names_dict.append(app + '_post_processing')

        column_names_dict.append('others')

    return column_names_dict, tou_array
