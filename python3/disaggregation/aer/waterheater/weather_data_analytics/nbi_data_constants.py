"""
Author: Mayank Sharan
Created: 27-Jan-2020
Contains classes defining constants used to define variables throughout the engine
"""

# Import python packages

import numpy as np


class SHCData:

    """Class defining constants related to shc data in nbi input dict"""

    # Define the number of columns in the array

    num_cols = 15

    # Define the column indices that are in the array

    bc_start_col = 0
    bc_end_col = 1

    app_id_col = 2
    app_cons_col = 3

    ptile_0_col = 4
    ptile_10_col = 5
    ptile_20_col = 6
    ptile_30_col = 7
    ptile_40_col = 8
    ptile_50_col = 9
    ptile_60_col = 10
    ptile_70_col = 11
    ptile_80_col = 12
    ptile_90_col = 13
    ptile_100_col = 14

    # List of app ids for which we populate shc data
    # Current appliances - to, pp, sh, ac, wh, ao, ref, ev, li

    app_id_list = np.array([17, 2, 3, 4, 7, 8, 9, 18, 71])


class BCData:

    """Class defining constants related to bc data in nbi input dict"""

    # Define the number of columns in the array

    num_cols = 3

    # Define the column indices that are in the array

    bc_start_col = 0
    bc_end_col = 1
    bc_dur_col = 2


class InvoiceData:

    """Class defining constants related to invoice data in nbi input dict"""

    # Define the number of columns in the array

    num_cols = 4

    # Define the column indices that are in the array

    bc_start_col = 0
    bc_end_col = 1
    bc_cost_col = 2
    bc_cons_col = 3


class DisaggData:

    """Class defining constants related to disagg data in nbi input dict"""

    # Define the column indices that are in the array

    app_id_col = 0

    # Define the row indices that are in the array

    bc_start_row = 0
    bc_end_row = 1

    bc_num_offset = 2
    app_id_num_offset = 1

    # List of app ids for which we populate disagg data
    # Current appliances - to, pp, sh, ac, wh, ao, ref, ev, li
    # Total is a special case copied from itemization since true disagg does not get it and it is identical

    app_id_list = np.array([17, 2, 3, 4, 7, 8, 9, 18, 71])


class ItemizationData:

    """Class defining constants related to itemization data in nbi input dict"""

    # Define the column indices that are in the array

    app_id_col = 0

    # Define the row indices that are in the array

    bc_start_row = 0
    bc_end_row = 1

    bc_num_offset = 2
    app_id_num_offset = 1

    # List of app ids for which we populate itemization data
    # Current appliances - to, pp, sh, ac, wh, ao, ref, ev, li

    app_id_list = np.array([17, 2, 3, 4, 7, 8, 9, 18, 71])


class WeatherData:

    """Class defining constants related to weather data in nbi input dict"""

    # Define the number of columns in the array

    num_cols = 6

    # Define the column indices that are in the array

    epoch_ts_col = 0
    day_ts_col = 1
    temp_col = 2
    feels_like_col = 3
    prec_col = 4
    snow_col = 5


class VacationData:

    """Class defining constants related to vacation data in nbi input dict"""

    # Define the number of columns in the array

    num_cols = 3

    # Define the column indices that are in the array

    epoch_ts_col = 0
    granularity_col = 1
    type_col = 2


class SeqData:

    """Class defining constants related to sequence data generated from find_seq function"""

    # Define the number of columns in the array

    num_cols = 4

    # Define the column indices that are in the array

    seq_val_col = 0
    seq_start_col = 1
    seq_end_col = 2
    seq_len_col = 3


class InsightValues:

    """Class defining constants related to insight values data in nbi process dict"""

    # Define the number of columns in the array

    num_cols = 6

    # Define the column indices that are in the array

    savings_ref_col = 0
    kwh_delta_col = 1
    pct_delta_col = 2
    ptile_delta_col = 3
    cost_col = 4
    strike_cnt_col = 5


class RankedNBIs:

    """Class defining constants related to Ranked NBIs"""

    # Define list of column headings

    column_headings = ['insight_id', 'action_id', 'nbi_id', 'insight', 'action', 'insight_relevance', 'action_value',
                       'raw_interaction_score', 'interaction_score', 'variable_value', 'insight_tags', 'action_tags',
                       'generic_tags', 'unit_of_measurement', 'nbi_family', 'nbi_type', 'nbi_fuel_type',
                       'nbi_appliance']

    # Define the number of columns in the array

    num_cols = 18

    # Define the column indices that are in the array

    insight_id_col = 0
    action_id_col = 1
    nbi_id_col = 2

    insight_text_col = 3
    action_text_col = 4

    insight_relevance_col = 5
    action_value_col = 6
    raw_interaction_score_col = 7
    interaction_score_col = 8

    var_value_col = 9

    insight_tags_col = 10
    action_tags_col = 11
    generic_tags_col = 12
    unit_of_measurement_col = 13

    nbi_family_col = 14
    nbi_type_col = 15
    nbi_fuel_type_col = 16
    nbi_appliance_col = 17


class InputDataframe:

    """Class defining constants related to input dataframe"""

    uuid_col = 'uuid'
    pilot_id_col = 'pilot_id'
    shc_data_col = 'shc_data'
    nbi_config_col = 'nbi_config'
    disagg_data_col = 'disagg_data'
    billing_data_col = 'billing_data'
    weather_data_col = 'weather_data'
    program_data_col = 'program_data'
    vacation_data_col = 'vacation_data'
    home_meta_data_col = 'home_meta_data'
    peak_usage_data_col = 'strike_agg_output'
    itemization_data_col = 'itemization_data'
    notification_data_col = 'notification_data'


class DataColumns:

    """Class defining constants related to DP columns """

    billing_data = 'billing_data'
    configurable_rules = 'configurable_rules'
    context = 'context'
    disagg_data = 'disagg_data'
    home_meta_data = 'home_meta_data'
    itemization_data = 'itemization_data'
    program_data = 'program_data'
    notification_data = 'notification_data'
    pilot_id = 'pilot_id'
    nbi_sheet = 'nbi_sheet'
    weather_data = 'weather_data'
    vacation_data = 'vacation_data'
    shc_data = 'shc_data'
