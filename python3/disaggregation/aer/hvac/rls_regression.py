#!/usr/bin/env python
# -*- coding:utf-8 -*-
""" RLS Regression """
import sys

import numpy as np
from scipy.optimize import nnls


def rls_regression(a0, y0, lamb0, error_weights=None):
    """  Run scipy.nnls """
    if error_weights is None:
        error_weights = np.ones(y0.shape[0]) / y0.shape[0]
    n_variables = a0.shape[1]
    a2 = np.concatenate([a0, np.sqrt(lamb0) * np.eye(n_variables)])
    y2 = np.concatenate([y0, np.zeros(n_variables)])

    weights0 = np.concatenate([error_weights, np.ones(n_variables)])

    aw = a2 * np.sqrt(weights0)[:, None]
    yw = y2 * np.sqrt(weights0)

    aw = np.nan_to_num(aw)
    yw = np.nan_to_num(yw)

    x, _ = nnls(aw, yw)
    return x
