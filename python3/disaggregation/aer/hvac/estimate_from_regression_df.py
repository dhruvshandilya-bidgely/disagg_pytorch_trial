"""
Author - Abhinav
Date - 10/10/2018
Estimating HVAC
"""

# Import python packages
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression


def perform_regression(regression_carrier_dict, regression_df, cluster, logger_hvac):

    """
    Function performs regression to get coefficient and estimates

    Parameters:
        regression_carrier_dict     (dict)           : Dictionary containing key regression object
        regression_df               (object)         : Main Dataframe for regression
        cluster                     (int)            : Cluster identifier
        logger_hvac                 (logging object) : Writes logs during code flow

    Returns:
        hvac_estimate_x_hours       (np.ndarray)     : HVAC estimates at x-hour aggregate
    """
    # Read all input objects
    cluster_info_master = regression_carrier_dict.get('cluster_info_master')
    appliance = regression_carrier_dict.get('appliance')
    cluster_info = regression_carrier_dict.get('cluster_info')
    filtered_data_df = regression_carrier_dict.get('filtered_data_df')
    identifier = regression_carrier_dict.get('identifier')
    hvac_estimate_x_hours = regression_carrier_dict.get('hvac_estimate_day')

    # Check whether it is linear or square root model
    if cluster_info[cluster]['regression_kind'] == 'linear':
        # Fit linear regression model
        regression = LinearRegression().fit(
            np.array(regression_df[identifier['degree_day']]).reshape(-1, 1),
            np.array(regression_df[identifier['day_consumption']]).reshape(-1, 1))

        # Store calculated regression coefficient and r_square values and the fit model
        hvac_coefficient = regression.coef_
        intercept = regression.intercept_
        r_square = pearsonr(np.array(regression_df[identifier['degree_day']]), np.array(regression_df[identifier['day_consumption']]))[0]

        cluster_info_master[appliance][cluster]['coefficient'] = hvac_coefficient
        cluster_info_master[appliance][cluster]['intercept'] = intercept
        cluster_info_master[appliance][cluster]['r_square'] = r_square
        cluster_info_master[appliance][cluster]['model'] = regression

        logger_hvac.info(' Regression Info : Kind {} , Coefficient {} , R-square {} |'.format('Linear', hvac_coefficient, r_square))

        # Get the hvac estimate from the regression coefficient calculated above
        hvac_estimate = list(np.array(regression_df[identifier['degree_day']]) * hvac_coefficient[0])
        hvac_estimate = [0 if cluster_flag != cluster else hvac_estimate.pop(0)
                         for cluster_flag in filtered_data_df[identifier['cluster_id']]]
        hvac_estimate_x_hours = np.sum([hvac_estimate_x_hours, hvac_estimate], axis=0)

    if cluster_info[cluster]['regression_kind'] == 'root':
        # Fit square root regression model
        regression = LinearRegression().fit(
            np.sqrt(np.array(regression_df[identifier['degree_day']])).reshape(-1, 1),
            np.array(regression_df[identifier['day_consumption']]).reshape(-1, 1))

        # Store calculated regression coefficient and r_square values and the fit model
        hvac_coefficient = regression.coef_
        intercept = regression.intercept_
        r_square = pearsonr(np.sqrt(np.array(regression_df[identifier['degree_day']])), np.array(regression_df[identifier['day_consumption']]))[0]

        cluster_info_master[appliance][cluster]['coefficient'] = hvac_coefficient
        cluster_info_master[appliance][cluster]['intercept'] = intercept
        cluster_info_master[appliance][cluster]['r_square'] = r_square
        cluster_info_master[appliance][cluster]['model'] = regression

        logger_hvac.info(' Regression Info : Kind {} , Coefficient {} , R-square {} |'.format('Root', hvac_coefficient, r_square))

        # Get the hvac estimate from the regression coefficient calculated above
        hvac_estimate = list(np.sqrt(np.array(regression_df[identifier['degree_day']])) * hvac_coefficient[0])
        hvac_estimate = [0 if cluster_flag != cluster else hvac_estimate.pop(0)
                         for cluster_flag in filtered_data_df[identifier['cluster_id']]]
        hvac_estimate_x_hours = np.sum([hvac_estimate_x_hours, hvac_estimate], axis=0)

    return hvac_estimate_x_hours


def estimate_from_regression_df(regression_carrier_dict, logger_hvac):

    """
    Function to prepare dataframe for performing regression

    Parameters:
        regression_carrier_dict     (dict)          : Dictionary containing key regression object
        logger_hvac                 (logger)        : Writes logs during code flow

    Returns:
        hvac_estimate_x_hour        (np.ndarray)    : Array containing hvac estimates at X-Hour level
    """

    # Read all input objects
    cluster_info = regression_carrier_dict.get('cluster_info')
    filtered_data_df = regression_carrier_dict.get('filtered_data_df')
    filter_day = regression_carrier_dict.get('filter_day')
    identifier = regression_carrier_dict.get('identifier')
    hvac_estimate_x_hour = regression_carrier_dict.get('hvac_estimate_day')

    # Perform regression for each cluster
    for cluster in cluster_info.keys():

        regression_df = filtered_data_df[(filtered_data_df[filter_day] == 1) &
                                         (filtered_data_df[identifier['cluster_id']] == cluster)][list(identifier.values())]

        # Check if the number of points and r2 within the cluster is above a threshold i.e. valid
        if cluster_info[cluster]['validity']:

            logger_hvac.info(' >> Regression is done on {} points for cluster {} |'.format(regression_df.shape[0], cluster))
            hvac_estimate_x_hour = perform_regression(regression_carrier_dict, regression_df, cluster, logger_hvac)

    return hvac_estimate_x_hour
