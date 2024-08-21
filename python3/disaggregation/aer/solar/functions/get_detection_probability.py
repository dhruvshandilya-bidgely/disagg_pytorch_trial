"""
Author - Paras Tehria
Date - 12/11/19
This module converts the data to suitable format for solar detection
"""

# import python packages

import torch
import torch.nn as nn
from copy import deepcopy
from python3.disaggregation.aer.solar.functions.cnn_model import Net


def get_detection_probability(detection_arr_original, model, solar_config):
    """
    This function gives the probability of solar detection using cnn model

    Parameters:
        detection_arr_original     (np.ndarray)    :       input array containing instances for detection
        model                      (Net)           :       cnn model
        solar_config               (dict)          :       config file

    Return:
        output                     (list)          :       output probability
    """

    cnn_detection_array = deepcopy(detection_arr_original)

    # cnn model expects input instances of shape (n_instances, 2, 24, 90)

    # converting shape from (n_instances, 90, 24, 2)

    permute_sequece = solar_config.get('get_detection_probability').get('permute_sequence')

    # number of instances

    n_inst = permute_sequece[0]

    # number of channels

    n_channels = permute_sequece[1]

    # number of columns

    n_col = permute_sequece[2]

    # number of rows

    n_row = permute_sequece[3]

    test_data_x = torch.tensor(cnn_detection_array).float().permute(n_inst, n_channels, n_col, n_row)
    output_val = model(test_data_x)

    # softmax applied to find probability

    softmax = nn.Softmax(dim=1)
    output = softmax(output_val)

    return output
