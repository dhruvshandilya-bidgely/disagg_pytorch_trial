"""
Author - Nikhil Singh Chauhan
Date - 10/10/2018
Class containing info of boxes features in timed water heater
"""

class Boxes:
    """
    Column numbers to use in boxes feature matrix
    """

    # Number of initial columns in boxes info
    NUM_COLS = 8

    # Start index of the box
    START_IDX = 0

    # End idx
    END_IDX = 1

    # Energy at the start data point of box
    START_ENERGY = 2

    # Energy at the end data point of box
    END_ENERGY = 3

    # Fraction of boxes start / end at the current box time division
    BOX_FRACTION = 4

    # Time division of the current box start / end
    TIME_DIVISION = 5

    # Boolean to represent if the box is valid timed water heater box
    IS_VALID = 6

    # Season of the box
    SEASON = 7

    # Seasonal fraction of boxes start / end at the current box time division
    SEASONAL_FRACTION = 8

    # Day number of the particular box
    DAY_NUMBER = 9
