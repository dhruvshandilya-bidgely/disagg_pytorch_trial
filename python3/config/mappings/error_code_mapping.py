"""
Author - Mayank Sharan
Date - 17/03/20
Maintain a mapping of rejection reason to error code
"""


class ErrorCodeMapping:

    """One on one mapping of each rejection reason to an error code"""

    # Mapping of data quality rejection error error message to error codes

    ERROR_CODE_DICT = {
        'Timezone is empty string': 1,
        'Timezone is not a string': 2,
        'Timezone field missing from the data': 3,
        'Country is empty string': 4,
        'Country is not a string': 5,
        'Country field missing from the data': 6,
        'Pilot id is invalid': 7,
        'Pilot id is not an int': 8,
        'Pilot id field missing from the data': 9,
        'Less data points than minimum required': 10,
        'Greater consecutive missing days than maximum allowed': 11,
        'Greater percentage of data points missing than maximum allowed': 12,
        'Billing cycle timestamps have more NaN than maximum': 13,
        'Billing cycle timestamps difference is negative': 14,
        'Week timestamps have more NaN than maximum': 15,
        'Week timestamps difference is negative': 16,
        'Day timestamps have more NaN than maximum': 17,
        'Day timestamps difference is negative': 18,
        'Day of week has more NaN than maximum': 19,
        'Hour of day has more NaN than maximum': 20,
        'Epoch timestamps have more NaN than maximum': 21,
        'Epoch timestamps difference is negative': 22,
        'Temperature present is less than required percentage': 23,
        'Sunrise timestamps present are less than required percentage': 24,
        'Sunset timestamps present are less than required percentage': 25,
        'Percentage of valid billing cycles less than required': 26,
        'Percentage of points needed in last billing cycle is less than required': 27,
        'Valid consumption percentage is less than required': 28,
        'Invalid sampling rate': 29,
    }
