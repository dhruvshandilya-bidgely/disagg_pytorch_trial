"""
Author - Paras Tehria
Date - 27 May 2021
Module to retrieve EV app profile from analytics_input_object
"""


def get_ev_app_profile(analytics_input_object, logger):
    """
    Parameters:
        analytics_input_object     (dict)      : Dictionary containing all inputs
        logger                  (logger)    : Logger object to logs values

    Returns:
        ev_present              (bool)      : Boolean to mark presence/absence of EV
        ev_app_profile_yes       (bool)      : Boolean to mark if user has said yes to ev in app profile
    """

    # Retrieve the water heater profile from disagg input object

    ev_app_profile = analytics_input_object.get('app_profile').get('ev')

    # By default set EV present to True in order to run the algorithm

    ev_present = True
    ev_app_profile_yes = False

    # If no status info present in meta data then run module assuming present

    if ev_app_profile is not None:
        logger.info('EV app profile is present | ')

        # ev_present is False if profile says zero EV present

        ev_number = ev_app_profile.get('number')
        ev_charging = ev_app_profile.get('size')

        # Cast the values to int

        ev_number = int(ev_number) if ev_number is not None else ev_number
        ev_charging = int(ev_charging) if ev_charging is not None else ev_charging

        # Check if the person said "no" for EV ownership or home charging

        if (ev_number == 0) or (ev_charging == 0):
            ev_present = False

        # Check if the person said "yes" for EV ownership

        if ev_number is not None and ev_number > 0:
            ev_app_profile_yes = True

        logger.info('EV ownership status | {}'.format(ev_number))
        logger.info('EV charging status | {}'.format(ev_charging))
    else:

        logger.info('EV app profile is not present | ')

    return ev_present, ev_app_profile_yes
