"""
Author - Abhinav Srivastava
Date - 10-Oct-2020
Called to populate smb lifestyle
"""

# Import python packages
import numpy as np


def populate_smb_lifestyle(disagg_input_object, disagg_output_object):

    """
    Function to populate smb lifestyle related key attributes

    Parameters:
        disagg_input_object     (dict)             : Dictionary containing all inputs
        disagg_output_object    (dict)             : Dictionary containing all outputs

    Returns:

        disagg_output_object    (dict)             : Dictionary containing all outputs
    """

    # initializing main carrier dictionary
    lifestyle_11_dict = {}

    # getting billing cycles and corresponding work hours of smb
    out_bill_cycles = disagg_input_object.get('out_bill_cycles')
    smb_time = disagg_output_object.get('special_outputs').get('smb')

    # populating smb lifestyle for every month and adding in main dictionary
    for bill_cycle_start, bill_cycle_end in out_bill_cycles:

        if smb_time.get(bill_cycle_start) == None:
            open_hour = None
            open_min = None
            close_hour = None
            close_min = None
        else:
            open_hour = smb_time.get(bill_cycle_start).get('open').hour
            open_min = smb_time.get(bill_cycle_start).get('open').minute
            close_hour = smb_time.get(bill_cycle_start).get('close').hour
            close_min = smb_time.get(bill_cycle_start).get('close').minute

        # initializing lifestyle object for current billing cycle
        lifestyleid_11 = {
            "name": "OperationalHoursSMB",
            "attributes": {
                "samplingRate": int(disagg_input_object['config']['sampling_rate']),
                "operationalHours": {
                    "openhour": open_hour,
                    "openminute": open_min,
                    "closehour": close_hour,
                    "closeminute": close_min
                },
                "dayOperationalHours": [
                ]
            },
            "debugAttributes": {

            }
        }

        # getting key attributes for current billing cycle
        day_xao = disagg_output_object.get('special_outputs').get('smb_outputs').get('day_extra_ao')
        valid_bill_cycles = day_xao.keys()

        # initializing daily operational hours
        day_operational_hours = []

        if bill_cycle_start in valid_bill_cycles:

            bill_cycle_days = day_xao.get(bill_cycle_start).get('day_epoch')
            bill_cycle_open_close = disagg_output_object.get('special_outputs').get('smb').get(bill_cycle_start).get('open_close_table')
            time_array = disagg_output_object.get('special_outputs').get('smb').get(bill_cycle_start).get('time_array')

            # populating daily work hours
            for idx in range(len(bill_cycle_days)):

                # work hours populated only for valid smb days
                try:
                    is_valid_day = np.any(bill_cycle_open_close[idx, :] == 1)
                except IndexError:
                    is_valid_day = False

                # populating work hours
                if is_valid_day:

                    # filling identified work hours
                    current_day = {
                        "dayepoch": int(bill_cycle_days[idx]),
                        "openhour": time_array[np.argwhere(bill_cycle_open_close[idx, :] == 1)[0][0]].hour,
                        "openminute": time_array[np.argwhere(bill_cycle_open_close[idx, :] == 1)[0][0]].minute,
                        "closehour": time_array[np.argwhere(bill_cycle_open_close[idx, :] == 1)[-1][0]].hour,
                        "closeminute": time_array[np.argwhere(bill_cycle_open_close[idx, :] == 1)[-1][0]].minute
                    }
                else:

                    # fail-safe work-hours
                    current_day = {
                        "dayepoch": int(bill_cycle_days[idx]),
                        "openhour": None,
                        "openminute": None,
                        "closehour": None,
                        "closeminute": None
                    }

                day_operational_hours.append(current_day)

        lifestyleid_11['attributes']['dayOperationalHours'] = day_operational_hours
        lifestyle_11_dict[bill_cycle_start] = lifestyleid_11

    # populating lifestyle in disagg output object
    disagg_output_object['special_outputs']['smb_outputs']['lifestyle_11'] = lifestyle_11_dict
