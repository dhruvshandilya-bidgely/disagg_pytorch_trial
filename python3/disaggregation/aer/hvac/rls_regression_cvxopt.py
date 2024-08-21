"""
Author - Abhinav
Date - 10/10/2018
Calculate adjusted ao
"""

# Import python packages

import numpy as np

# Import functions from within the project

from cvxopt.solvers import qp, options
from cvxopt import matrix as cvxmat


def rls_regression_cvxopt(a0, y0, lamb0, error_weights=None, extra_constraint=0):

    """
    Parameters:
        a0 (numpy array)                             : Input Vector
        y0 (numpy array)                             : Output vector
        lamb0 (numpy array)                          : Regularizer
        error_weights (numpy array)                  : Regularization
        extra_constraint (numpy array)               : Variable

    Returns:
        coefficients (numpy array)                   : The coefficients of learned regression
    """

    if error_weights is None:
        error_weights = np.ones(y0.shape[0]) / y0.shape[0]

    n_variables = a0.shape[1]
    i = np.eye(n_variables)
    l = -i
    b = np.zeros(n_variables)
    i[-1, :] = 0
    h = np.around(np.matmul(np.matmul(np.transpose(a0), np.diagflat(error_weights)), a0) + lamb0 * i, 9)
    f = np.around(np.matmul(np.matmul(np.transpose(-y0), np.diagflat(error_weights)), a0), 7)

    if extra_constraint:
        l = np.vstack((l, a0))
        b = np.vstack((b, y0))

    p = cvxmat(h)
    q = cvxmat(f)
    g = cvxmat(l)
    h = cvxmat(b)

    options['show_progress'] = False
    results = qp(p, q, g, h)
    xstar = results['x']
    coefficients = np.squeeze(np.asarray(xstar))

    return coefficients
