"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to regulate HVAC estimations
"""

# Import python packages
import copy


def homogenize_hvac_maps(hour_medians, operational_values, open_close_map, ac_global, sh_global):

    """
    Function to homogenize hvac consumption keeping operational load in mind

    Parameters:

        hour_medians        (np.ndarray)               : Array containing median residue consumption left at hour level
        operational_values  (np.ndarray)               : Array containing operational consumption
        open_close_map      (pd.DataFrame)             : Contains open-close hour information
        ac_global           (np.ndarray)               : Array containing ac consumption
        sh_global           (np.ndarray)               : Array containing sh consumption

    Returns:

        operational_values  (np.ndarray)         : Array containing operational consumption data
        ac_global           (np.ndarray)         : Array containing ac consumption data
        sh_global           (np.ndarray)         : Array containing sh consumption data
    """

    # homogenizing operational load and updating hvac consumption
    for idx in range(len(hour_medians)):

        # getting representative initial comparator for operational load for any hour
        hour_median = hour_medians[idx]
        operational = copy.deepcopy(operational_values[:, idx])

        # identifying deficiency epochs in current operational loa based on comparator
        operational_deficiency_bool = ((operational >= 0) & (operational < hour_median)).astype(int)
        operational_deficiency_bool = operational_deficiency_bool * open_close_map.values[:, idx]

        # quantifying operational deficiency
        operational_deficiency = operational - hour_median
        operational_deficiency = operational_deficiency * operational_deficiency_bool

        # accessing raw ac and sh estimations
        ac_values = ac_global[:, idx]
        sh_values = sh_global[:, idx]

        # identifying ac and sh estimation epochs
        ac_bool = (ac_values > 0).astype(int)
        sh_bool = (sh_values > 0).astype(int)

        # seeing possibility of extracting operational load out of ac based on qualified epochs through comparator
        operational_from_ac_bool = operational_deficiency_bool * ac_bool
        operational_from_sh_bool = operational_deficiency_bool * sh_bool

        # requirement to meet an ideal operational hour consistency
        operational_required_from_ac = operational_deficiency * operational_from_ac_bool * -1
        operational_required_from_sh = operational_deficiency * operational_from_sh_bool * -1

        # practical sufficiency level based on hvac values and requirement
        ac_potential_sufficient = ((ac_values * operational_from_ac_bool) >= operational_required_from_ac).astype(int)
        ac_potential_sufficient = ac_potential_sufficient * operational_from_ac_bool
        sh_potential_sufficient = ((sh_values * operational_from_sh_bool) >= operational_required_from_sh).astype(int)
        sh_potential_sufficient = sh_potential_sufficient * operational_from_sh_bool

        # practical insufficiency based on hvac values and requirement
        ac_potential_insufficient = ((ac_values * operational_from_ac_bool) < operational_required_from_ac).astype(int)
        ac_potential_insufficient = ac_potential_insufficient * operational_from_ac_bool
        sh_potential_insufficient = ((sh_values * operational_from_sh_bool) < operational_required_from_sh).astype(int)
        sh_potential_insufficient = sh_potential_insufficient * operational_from_sh_bool

        # getting operational load out of hvac for requirement sufficient epochs
        operational_from_ac_suf = ac_potential_sufficient * operational_deficiency * -1
        operational_from_ac_suf = operational_from_ac_suf * ac_potential_sufficient
        operational_from_sh_suf = sh_potential_sufficient * operational_deficiency * -1
        operational_from_sh_suf = operational_from_sh_suf * sh_potential_sufficient

        # getting best operational load out of hvac for requirement insufficient epochs
        operational_from_ac_insuf = ac_potential_insufficient * ac_values
        operational_from_sh_insuf = sh_potential_insufficient * sh_values

        # net operational extracted
        operational_from_ac = operational_from_ac_suf + operational_from_ac_insuf
        operational_from_sh = operational_from_sh_suf + operational_from_sh_insuf

        # updating operational load based on extracted values out of hvac
        operational = operational + operational_from_ac + operational_from_sh
        ac_values = ac_values - operational_from_ac
        sh_values = sh_values - operational_from_sh

        # assigning and updating operational, hvac loads
        operational_values[:, idx] = operational
        ac_global[:, idx] = ac_values
        sh_global[:, idx] = sh_values

    return operational_values, ac_global, sh_global
