"""
Date Created - 13 Nove 2018
Author name - Pratap
Unixtimestamp 2 month or year function
"""
from datetime import datetime


def unix2daydateint(unix_time):
    """

    :param unix_time:
    :return:
    """
    #unixtimestamp to date conversion
    unix_epoch_date = int(datetime.utcfromtimestamp(unix_time).strftime('%d'))
    return unix_epoch_date


def unix2mthdateint(unix_time):
    """

    :param unix_time:
    :return:
    """
    #unixtimestamp to month conversion
    unix_epoch_mth = int(datetime.utcfromtimestamp(unix_time).strftime('%m'))
    return unix_epoch_mth


def unix2yeardateint(unix_time):
    """

    :param unix_time:
    :return:
    """
    #unixtimestamp to year conversion
    unix_epoch_year = int(datetime.utcfromtimestamp(unix_time).strftime('%Y'))
    return unix_epoch_year
