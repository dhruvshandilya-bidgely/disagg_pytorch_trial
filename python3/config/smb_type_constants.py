"""
pilot_constants file contains all the pilot related configurations
"""


class SMBTypeConstants:
    """Class containing all SMB Type related constants"""
    AC_DISABLED_SMB_TYPES = []
    SH_DISABLED_SMB_TYPES = ['COLD_STORAGE', 'DATA_CENTER']
    LIGHTING_DISABLED_SMB_TYPES = []

    WH_ENABLED_SMB_TYPES = ['HOTEL', 'HOSPITAL']
    SERVICE_ENABLED_SMB_TYPES = ['CHURCH']
    COOKING_ENABLED_SMB_TYPES = ['RESTAURANT', 'GROCERY_STORE', ]
    EQUIPMENTS_ENABLED_SMB_TYPES = ['OFFICE', 'HOSPITALS', 'HOTELS', 'STORE', 'DATA_CENTER']
    REF_ENABLED_SMB_TYPES = ['CHURCH', 'OFFICE', 'RESTAURANT', 'HOTEL', 'BAR', 'GROCERY_STORE']

    APPLIANCE_DISTRIBUTION = {
        # 'BAR': {},

        'CHURCH': {'lighting': 0.12,
                   'ref': 0.05,
                   'miscellaneous': 0.48},

        'GROCERY_STORE': {'lighting': 0.07,
                          'ref': 0.73,
                          'cooking': 0.05,
                          'miscellaneous': 0.09},

        # Couldn't get the pie chart on https://esource.bizenergyadvisor.com/article/hospitals
        # Took data from: https://www.researchgate.net/publication/336274024_Energy_Consumption_Analysis_and_
        # Characterization_of_Healthcare_Facilities_in_the_United_States
        'HOSPITAL': {'lighting': 0.17,
                     'ref': 0.05,
                     'equipments': 0.14,
                     'cooking': 0.02,
                     'miscellaneous': 0.19},

        'HOTEL': {'lighting': 0.12,
                  'ref': 0.12,
                  'equipments': 0.16,
                  'miscellaneous': 0.35},

        'OFFICE': {'lighting': 0.16,
                   'equipments': 0.24,
                   'miscellaneous': 0.21},

        # Couldn't get the pie chart on https://esource.bizenergyadvisor.com/article/restaurants
        # Took data from: https://solutions.rdtonline.com/blog/restaurant-energy-consumption-statistics
        'RESTAURANT': {'lighting': 0.13,
                       'ref': 0.06,
                       'cooking': 0.35},

        'STORE': {'lighting': 0.26,
                  'ref': 0.19,
                  'equipments': 0.04,
                  'miscellaneous': 0.21},

        'DATA_CENTER': {'lighting': 0.15,
                        'ref': 0.06,
                        'equipments': 0.19,
                        'miscellaneous': 0.23}
    }
