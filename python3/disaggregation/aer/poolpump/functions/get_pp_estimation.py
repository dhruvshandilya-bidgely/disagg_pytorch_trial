"""
Author: Arpan Agrawal
Date - 01/04/2019
Gives detection and estimation for the type of PP detected
"""

import numpy as np

from python3.config.Cgbdisagg import Cgbdisagg

from python3.disaggregation.aer.poolpump.functions.estimation_utils_2 import reject_poolpump

from python3.disaggregation.aer.poolpump.functions.estimation_utils import build_confidence_score

from python3.disaggregation.aer.poolpump.functions.get_single_run_estimation import get_single_run_estimation
from python3.disaggregation.aer.poolpump.functions.get_variable_pp_estimation import get_variable_pp_estimation
from python3.disaggregation.aer.poolpump.functions.get_multiple_run_estimation import get_multiple_run_estimation


def get_pp_estimation(input_dict, steps, num_of_runs_each_day, days_label, duration_each_day, filled_days,
                      pp_config):
    """Gives detection and estimation for the type of PP detected"""

    probability_threshold = 0.45

    uncontested_matrix = input_dict['uncontested_matrix']
    data_clean_edges = input_dict['data_clean_edges']
    data_nms = input_dict['data_nms']
    data_bl_removed = input_dict['data_bl_removed']
    time_div_dict = input_dict['time_div_dict']
    runs_tuple = input_dict['runs_tuple']
    all_pos_edges = input_dict['all_pos_edges']
    all_neg_edges = input_dict['all_neg_edges']
    global_pos_edges = input_dict['global_pos_edges']
    global_neg_edges = input_dict['global_neg_edges']
    global_pairs = input_dict['global_pairs']
    day_seasons = input_dict['day_seasons']

    run_type = runs_tuple[0]

    num_samples_per_hr = int(Cgbdisagg.SEC_IN_HOUR / pp_config.get('sampling_rate'))

    structures = np.array([])
    consumption_matrix = np.zeros_like(data_clean_edges)

    if run_type == 'Single' or run_type == 'Variable':
        final_edges_matrix, runs_tuple, consumption_matrix, pp_run_days_arr, structures = \
            get_single_run_estimation(input_dict, days_label, filled_days, duration_each_day, num_of_runs_each_day,
                                      pp_config)

        steps[3] = final_edges_matrix

        single_run_consumption_matrix = consumption_matrix
        single_run_days_arr = pp_run_days_arr

        if run_type == 'Variable' and runs_tuple[0] != 'NoRun':

            var_pair_matrix, consumption_matrix, high_amp_pairs, structures = \
                get_variable_pp_estimation(data_clean_edges, data_nms, data_bl_removed, num_samples_per_hr, pp_config)

            steps[3] = var_pair_matrix

            confidence_score, vpp_run_days_arr = build_confidence_score(data_nms, high_amp_pairs, pp_config,
                                                                        variable=True)
            reject_winter_pp_bool = reject_poolpump(day_seasons, vpp_run_days_arr)

            if reject_winter_pp_bool:
                confidence_score = str(confidence_score) + "W"
                runs_tuple = ('NoRun', 0, confidence_score)
            elif confidence_score < probability_threshold:
                confidence_score = str(confidence_score)
                runs_tuple = ('NoRun', 0, confidence_score)
            else:
                confidence_score = str(confidence_score)
                runs_tuple = runs_tuple[:-1] + (confidence_score,)

            whitespaces_to_be_filled = np.where((single_run_days_arr != 0) & (vpp_run_days_arr == 0))[0]
            consumption_matrix[whitespaces_to_be_filled, :] = single_run_consumption_matrix[whitespaces_to_be_filled, :]

    elif run_type == 'Multiple':

        input_dict['uncontested_matrix'] = uncontested_matrix
        input_dict['data_clean_edges'] = data_clean_edges
        input_dict['data_nms'] = data_nms
        input_dict['data_bl_removed'] = data_bl_removed
        input_dict['runs_tuple'] = runs_tuple
        input_dict['day_seasons'] = day_seasons
        input_dict['global_pos_edges'] = global_pos_edges
        input_dict['global_neg_edges'] = global_neg_edges
        input_dict['global_pairs'] = global_pairs
        input_dict['time_div_dict'] = time_div_dict
        final_edges_matrix, runs_tuple, consumption_matrix, structures = get_multiple_run_estimation(input_dict,
                                                                                                     days_label,
                                                                                                     filled_days,
                                                                                                     duration_each_day,
                                                                                                     num_of_runs_each_day,
                                                                                                     all_pos_edges,
                                                                                                     all_neg_edges,
                                                                                                     pp_config)
        # get_multiple_run_estimation(uncontested_matrix, days_label, filled_days, duration_each_day,
        #                             num_of_runs_each_day, data_nms, data_clean_edges, data_bl_removed,
        #                             all_pos_edges, all_neg_edges, global_pos_edges, global_neg_edges, global_pairs,
        #                             time_div_dict, runs_tuple, day_seasons, pp_config)

        steps[3] = final_edges_matrix

    return consumption_matrix, steps, runs_tuple, structures
