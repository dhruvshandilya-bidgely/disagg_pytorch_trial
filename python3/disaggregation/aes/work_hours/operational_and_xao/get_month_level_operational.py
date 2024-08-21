"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to get SMB Operational load
"""

# Import python packages
import copy
import logging
import numpy as np


def get_month_level_operational_load(month_level_dict, logger_base):

    """
    Function to get month level operational load

    Parameters:
        month_level_dict    (dict)               : Dictionary containing month level consumption information
        logger_base         (logging object)     : Keeps log of code flow

    Returns:
        operational_values  (np.array)           : Array containing operational consumption data
        ac_global           (np.array)           : Array containing ac consumption data
        sh_global           (np.array)           : Array containing sh consumption data
    """

    # initializing logger object
    logger_local = logger_base.get("logger").getChild("op_month")
    logger_op_month = logging.LoggerAdapter(logger_local, logger_base.get("logging_dict"))

    # getting key attributes for getting month level operational load
    unique_months = month_level_dict.get('unique_months')
    day_validity = month_level_dict.get('day_validity')
    month_identifier = month_level_dict.get('month_identifier')
    residue_work_hours = month_level_dict.get('residue_work_hours')
    operational_values = month_level_dict.get('operational_values')
    open_close_map = month_level_dict.get('open_close_map')
    ac_global = month_level_dict.get('ac_global')
    sh_global = month_level_dict.get('sh_global')
    operational_map = month_level_dict.get('operational_map')

    # getting operational load for each month at epoch level
    for month in unique_months:

        valid_days_in_month = day_validity[month_identifier == month]

        # operational load will be estimated only for valid days in a month. No consumption makes a day invalid.
        if np.sum(valid_days_in_month) > 0:

            logger_op_month.debug("Valid number of days exist |")

            # getting valid epoch from current month
            month_bool = month_identifier == month
            valid_day_bool = day_validity == 1
            days_to_consider = month_bool & valid_day_bool

            # getting residue in work hours for current month
            residue_work_hours_month = residue_work_hours[days_to_consider]

            # getting base hour medians, initial proxy for operational load
            hour_medians = residue_work_hours_month[residue_work_hours_month > 0].median(axis=0).values
            hour_medians = np.nan_to_num(hour_medians)

            # checking consistency of operational load across hours in a month
            for idx in range(len(hour_medians)):

                curr_hour_med = hour_medians[idx]

                # initializing operational load
                operational_values[days_to_consider & (operational_values[:, idx] > curr_hour_med), idx] = curr_hour_med
                operational = copy.deepcopy(operational_values[month_bool, idx])

                # checking operational load deficiency, with reference to hour medians base
                operational_deficiency_bool = ((operational >= 0) & (operational < hour_medians[idx])).astype(int)
                operational_deficiency_bool = operational_deficiency_bool * open_close_map.values[month_bool, idx]

                operational_deficiency = operational - hour_medians[idx]
                operational_deficiency = operational_deficiency * operational_deficiency_bool

                # accessing hvac consumption to enable consistency of operational load
                ac_values = ac_global[month_bool, idx]
                sh_values = sh_global[month_bool, idx]

                # getting boolean of hvac points, where consumption exists
                ac_bool = (ac_values > 0).astype(int)
                sh_bool = (sh_values > 0).astype(int)

                # checking possibility of homogenizing operational load using hvac
                operational_from_ac_bool = operational_deficiency_bool * ac_bool
                operational_from_sh_bool = operational_deficiency_bool * sh_bool

                # checking existing operational deficiency
                operational_required_from_ac = operational_deficiency * operational_from_ac_bool * -1
                operational_required_from_sh = operational_deficiency * operational_from_sh_bool * -1

                # finding epochs where hvac is sufficient to cater to operational deficiency
                ac_potential_sufficient = ((ac_values * operational_from_ac_bool) >= (operational_required_from_ac)).astype(int)
                ac_potential_sufficient = ac_potential_sufficient * operational_from_ac_bool
                sh_potential_sufficient = ((sh_values * operational_from_sh_bool) >= (operational_required_from_sh)).astype(int)
                sh_potential_sufficient = sh_potential_sufficient * operational_from_sh_bool

                # finding epochs where hvac is in-sufficient to cater to operational deficiency
                ac_potential_insufficient = ((ac_values * operational_from_ac_bool) < (operational_required_from_ac)).astype(int)
                ac_potential_insufficient = ac_potential_insufficient * operational_from_ac_bool
                sh_potential_insufficient = ((sh_values * operational_from_sh_bool) < (operational_required_from_sh)).astype(int)
                sh_potential_insufficient = sh_potential_insufficient * operational_from_sh_bool

                # getting operational out of hvac where sufficiency condition is met
                operational_from_ac_suf = ac_potential_sufficient * operational_deficiency * -1
                operational_from_ac_suf = operational_from_ac_suf * ac_potential_sufficient
                operational_from_sh_suf = sh_potential_sufficient * operational_deficiency * -1
                operational_from_sh_suf = operational_from_sh_suf * sh_potential_sufficient

                # getting partial-operational out of hvac where in-sufficiency condition is met
                operational_from_ac_insuf = ac_potential_insufficient * ac_values
                operational_from_sh_insuf = sh_potential_insufficient * sh_values

                operational_from_ac = operational_from_ac_suf + operational_from_ac_insuf
                operational_from_sh = operational_from_sh_suf + operational_from_sh_insuf

                # updating net operational load
                operational = operational + operational_from_ac + operational_from_sh
                ac_values = ac_values - operational_from_ac
                sh_values = sh_values - operational_from_sh

                # updating net hvac load
                operational_values[month_bool, idx] = operational
                ac_global[month_bool, idx] = ac_values
                sh_global[month_bool, idx] = sh_values

        else:

            logger_op_month.debug("Valid number of days do not exist |")
            hour_medians = np.zeros((operational_map.shape[1]))

    return operational_values, ac_global, sh_global
