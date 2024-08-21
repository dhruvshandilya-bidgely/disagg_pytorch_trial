"""
Author - Mayank Sharan
Date - 11/10/18
Returns appliance id given the appliance and vice versa
"""

# No imports here


def get_app_id(app_name):

    """
    Parameters:
        app_name            (string)            : The name of the appliance we want an id for

    Returns:
        app_id              (int)               : The appliance id corresponding to the given appliance name
    """

    # 2 : Pool Pump
    # 3 : Space Heater
    # 4 : Air Conditioner
    # 7 : Water Heater
    # 8 : Always On
    # 9 : Refrigerator
    # 15 : Heating, Ventilation and Cooling
    # 16 : Solar
    # 18 : Electric Vehicle
    # 71 : Lighting
    # 81 : SMB Operational
    # 82 : SMB Extra-AO
    # 101 : Vacation type 1
    # 102 : Vacation type 2 disconnection

    app_name_id_map = {
        'pp': 2,
        'sh': 3,
        'ac': 4,
        'wh': 7,
        'ao': 8,
        'ref': 9,
        'hvac': 15,
        'solar': 16,
        'cook': 5,
        'ent': 66,
        'ld': 59,
        'ev': 18,
        'li': 71,
        'op': 81,
        'x-ao': 82,
        'others': 99,
        'va': 1001,
        'vad': 1002,
        'hvac_ineff': 1005
    }

    app_id = app_name_id_map.get(str.lower(app_name))

    if app_id is None:
        app_id = -1

    return app_id


def get_app_name(app_id):
    """
    Parameters:
        app_id              (int)               : The appliance id corresponding to the given appliance name

    Returns:
        app_name            (string)            : The name of the appliance we want an id for
    """

    # 2 : Pool Pump
    # 3 : Space Heater
    # 4 : Air Conditioner
    # 7 : Water Heater
    # 8 : Always On
    # 9 : Refrigerator
    # 15 : Heating, Ventilation and Cooling
    # 16 : Solar
    # 18 : Electric Vehicle
    # 19 : Room Heater
    # 71 : Lighting
    # 81 : SMB Operational
    # 82 : SMB Extra-AO
    # 101 : Vacation type 1
    # 102 : Vacation type 2 disconnection

    app_id_name_map = {
        2: 'pp',
        3: 'sh',
        4: 'ac',
        7: 'wh',
        8: 'ao',
        9: 'ref',
        13: 'rac',
        15: 'hvac',
        16: 'solar',
        18: 'ev',
        19: 'rh',
        20: 'ch',
        48: 'hp',
        52: 'cac',
        71: 'li',
        81: 'op',
        82: 'x-ao',
        1001: 'va',
        1002: 'vad',
        1005: 'hvac_ineff',
        5: 'cook',
        66: 'ent',
        59: 'ld',
        99: 'others'
    }

    app_name = app_id_name_map.get(int(app_id))

    if app_name is None:
        app_name = app_id

    return app_name
