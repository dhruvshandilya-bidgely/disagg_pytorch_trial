#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""GET_ERROR computes and returns the difference between 2 variables"""

import copy
import numpy as np


def get_error(m_var, p_var):

    """
    get error computes difference between 2 variables given. The input should be numpy arrays
    :param m_var: matlab version of the variable
    :param p_var: python version of the variable
    :return: average absolute error, absolute error percentage value, maximum error percentage value,
             indices with error greater than 0.01%
    """

    mat_var = copy.deepcopy(m_var)
    py_var = copy.deepcopy(p_var)

    if not type(mat_var) is np.ndarray:
        print('Please make sure the matlab variable is a numpy array')
        return

    if not type(py_var) is np.ndarray:
        py_var = np.array(py_var)

    if not(mat_var.shape == py_var.shape):
        py_var = np.reshape(py_var, newshape=mat_var.shape)

    m_shape = mat_var.shape
    p_shape = py_var.shape

    if not m_shape == p_shape:
        print('The variable sizes do not match. Please check', m_shape, p_shape)
        return

    base_var_m = mat_var
    base_var_p = py_var

    for idx in range(len(m_shape)):
        base_var_m = base_var_m[0]
        base_var_p = base_var_p[0]

    if type(base_var_p) is np.bool_:
        mat_var_flt = np.zeros(mat_var.shape)
        py_var_flt = np.zeros(py_var.shape)

        mat_var_flt[mat_var == True] = 1
        py_var_flt[py_var == True] = 1

        mat_var = mat_var_flt
        py_var = py_var_flt

    mat_var[np.isnan(mat_var)] = 0
    py_var[np.isnan(py_var)] = 0

    diff_mat = np.subtract(mat_var, py_var)
    abs_mat = np.absolute(diff_mat)

    avg_abs_error = np.sum(abs_mat[abs_mat > 0]) / abs_mat.size

    ratio_mat = np.divide(abs_mat, mat_var)
    ratio_mat[np.isnan(ratio_mat)] = 0
    ratio_mat[~np.isfinite(ratio_mat)] = 2
    perc_error = np.nanmean(ratio_mat[ratio_mat > 0]) * 100

    problem_idx = np.argwhere(ratio_mat > 0.0001)

    return avg_abs_error, perc_error, np.max(abs_mat), np.max(ratio_mat * 100), problem_idx


if __name__ == "__main__":

    a = np.array([1.1, 1.2, 1.3])
    b = np.array([1.1, 1.1, np.nan])

    print(get_error(a, b))

    c = np.array([[True, False, False], [True, False, True]])
    d = np.array([[True, False, True], [True, False, True]])

    get_error(c, d)
