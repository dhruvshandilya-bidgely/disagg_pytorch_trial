"""
Author - Abhinav
Date - 10/10/2018
Calculate adjusted ao
"""

# Import python packages

import numpy as np

# Import functions from within the project

from scipy import stats


def linear_fit(independent_var, dependent_var):

    """
        Function to detect setpoint for sh or ac for a user

        Parameters:
            independent_var     (np.ndarray)           : Array of input variable
            dependent_var       (np.ndarray)           : Array of input variable
        Returns:
            model_parameters    (dict)                 : model fit related model parameters
        """

    try:
        # Calculate the regression coefficient and standard error of its estimate
        n_samples = len(independent_var)
        temp = np.c_[np.ones((n_samples, 1)), independent_var]
        coeff, _, _, model_parameters = np.linalg.lstsq(temp, dependent_var)
        y_hat = np.matmul(np.c_[np.ones((n_samples, 1)), independent_var], coeff)
        ss_err = np.sum((y_hat - dependent_var) ** 2)
        ss_tot = np.nanvar(dependent_var, ddof=1) * (n_samples - 1)

        # Check the fit of the least squares model
        r_squared = 1 - ss_err / ss_tot
        r_squared = np.around(r_squared, 5)
        se_coeff = np.sqrt(1 / np.nanvar(independent_var, ddof=1) / (n_samples - 1) * ss_err / (n_samples - 2))

        # Store p-value of the coefficient to check validity later
        p_value = np.around(stats.t.cdf(coeff[1] / se_coeff, n_samples - 2), 5)
        p_value = np.around(2 * min(p_value, 1 - p_value), 5)
        model_parameters = {
            'Coefficients': {
                'Estimate': [coeff[0], coeff[1]],
                'pValue': [0, p_value],
                'tValue': [0, coeff[1]/se_coeff],
            },
            'Rsquared': {'Ordinary': np.round(r_squared, 5)}
        }
    except Exception:
        model_parameters = {
            'Coefficients': {
                'Estimate': [0, 0],
                'pValue': [0, np.nan],
                'tValue': [0, np.nan],
            },
            'Rsquared': {'Ordinary': np.nan}
        }

    return model_parameters
