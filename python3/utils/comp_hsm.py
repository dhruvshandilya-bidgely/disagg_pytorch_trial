#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Comp HSM compares 2 hsms"""

import sys
import copy
import numpy as np

from python3.utils.get_error import get_error


def comp_hsm(hsm_1, hsm_2):

    """Compares 2 hsms and returns a dictionary with attributes and errors"""

    return_message = ""

    # Extract attribute dictionaries from the hsms

    try:
        attr_dic_1 = hsm_1['attributes']
    except KeyError:
        return_message = "HSM1 has an issue\n"

    try:
        attr_dic_2 = hsm_2['attributes']
    except KeyError:
        return_message = "HSM2 has an issue\n"

    if not(return_message == ""):
        return return_message

    # Extract keys from the attribute dictionaries

    keys_dic_1 = list(attr_dic_1.keys())
    keys_dic_2 = list(attr_dic_2.keys())

    if not(len(keys_dic_1) == len(keys_dic_2)):
        return "HSM attribute lengths do not match\n"

    for idx in range(len(keys_dic_1)):

        dic_key = keys_dic_1[idx]

        value_1 = attr_dic_1[dic_key]
        value_2 = attr_dic_2[dic_key]

        aae, ape, mae, mpe, _ = get_error(value_1, value_2)

        return_message += dic_key + ': avg abs error = ' + str(aae) + '; max abs error = ' + str(mae) + '\n'

    return return_message
