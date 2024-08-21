"""MATLAB utils_global contains utility functions that emulate functions from MATLAB"""

import copy
import numpy as np


def percentile_1d(arr, ptile, method = 'matlab'):
    """1d percentile based on MATLAB implemented for python"""

    arr_clean = np.sort(arr[~np.isnan(arr)])
    num_el = len(arr_clean)

    if num_el > 0:
        p_rank = 100.0 * (np.arange(num_el) + 0.5) / num_el
        return np.interp(ptile, p_rank, arr_clean)
    else:
        return np.nan


def conv_matlab_same(u, v):
    """ MA implementation"""
    npad = len(v) - 1
    full = np.convolve(u, v, 'full')
    first = npad - npad // 2

    return full[first:first + len(u)]


def superfast_matlab_percentile(data, ptile, axis=0, method='matlab'):
    """Superfast vectorized implementation in python of MATLAB's percentile function"""

    if data.ndim == 1:

        return percentile_1d(data, ptile)

    elif data.ndim == 2:

        if data.shape[axis] == 0:
            num_out = data.shape[1 - axis]
            res = np.zeros(shape=(num_out,))
            res[:] = np.nan
            return res

        quantile = float(ptile) / 100

        valid_obs = np.sum(np.isfinite(data), axis=axis)
        data_s = np.sort(data, axis=axis)

        k_arr = quantile * valid_obs - 0.5
        k_arr[k_arr < 0] = 0
        k_arr[k_arr > valid_obs - 1] = valid_obs[k_arr > valid_obs - 1] - 1

        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        num_out = data.shape[1 - axis]
        res = np.zeros(shape=(num_out, ))

        if axis == 1:
            floor_val = data_s[np.arange(num_out), f_arr] * (c_arr - k_arr)
            ceil_val = data_s[np.arange(num_out), c_arr] * (k_arr - f_arr)
            res = floor_val + ceil_val
            res[fc_equal_k_mask] = (data_s[np.arange(num_out), f_arr])[fc_equal_k_mask]

        if axis == 0:
            floor_val = data_s[f_arr, np.arange(num_out)] * (c_arr - k_arr)
            ceil_val = data_s[c_arr, np.arange(num_out)] * (k_arr - f_arr)
            res = floor_val + ceil_val
            res[fc_equal_k_mask] = (data_s[f_arr, np.arange(num_out)])[fc_equal_k_mask]

    elif data.ndim == 3 and axis == 2:

        quantile = float(ptile) / 100

        valid_obs = np.sum(np.isfinite(data), axis=axis)
        data_s = np.sort(data, axis=axis)

        k_arr = quantile * valid_obs - 0.5
        k_arr[k_arr < 0] = 0
        k_arr[k_arr > valid_obs - 1] = valid_obs[k_arr > valid_obs - 1] - 1

        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        num_out_x = data.shape[0]
        num_out_y = data.shape[1]

        res = np.zeros(shape=(num_out_x, num_out_y))

        floor_val = data_s[np.tile(np.arange(num_out_x), (num_out_y, 1)).transpose(),
                           np.tile(np.arange(num_out_y), (num_out_x, 1)), f_arr] * (c_arr - k_arr)
        ceil_val = data_s[np.tile(np.arange(num_out_x), (num_out_y, 1)).transpose(),
                          np.tile(np.arange(num_out_y), (num_out_x, 1)), c_arr] * (k_arr - f_arr)
        res = floor_val + ceil_val
        res[fc_equal_k_mask] = (data_s[np.tile(np.arange(num_out_x), (num_out_y, 1)).transpose(),
                                       np.tile(np.arange(num_out_y), (num_out_x, 1)), f_arr])[fc_equal_k_mask]
    else:
        res = np.array([])

    return res


def is_member(a_vec, b_vec):
    """MATLAB equivalent ismember function"""

    bool_out = np.isin(a_vec, b_vec)

    b_dict = {b_vec[i]: i for i in range(0, len(b_vec))}
    index = [b_dict.get(x) if b_dict.get(x) is not None else -1 for x in a_vec]

    return bool_out, np.array(index, dtype=int)


def rolling_sum(data, roll_window, axis=0):

    """Utility to get rolling sum"""

    ret = np.cumsum(data, axis=axis)

    if axis == 0:
        ret[roll_window:, :] = ret[roll_window:, :] - ret[:-roll_window, :]
        return ret[roll_window - 1:, :]
    elif axis == 1:
        ret[:, roll_window:] = ret[:, roll_window:] - ret[:, :-roll_window]
        return ret[:, roll_window - 1:]
