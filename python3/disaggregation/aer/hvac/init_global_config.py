#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Init config"""
import sys
import numpy as np

# noinspection PyUnresolvedReferences
from python3.disaggregation.aer.hvac.init_hourly_hvac_params import Struct


def init_global_config():
    """
    Imported from MATHLAB  globalConfig fields
    """
    gconfig = Struct(**{
        'APIFACTORY': [],
        'UUID': 'bf9f01f7-0534-4a16-afb4-5039752d68d8',
        'HOME': 1,
        'PILOTID': 10009,
        'MODEL_LOCATION': '+GB/trainedmodel.mat',
        'IS_OLD_MODEL': 1,
        'MINS_PER_SAMPLE': 60,
        'HDD_POOLPUMP_THRESH': 2,
        'CDD_POOLPUMP_THRESH': -1,
        'TEMP_VALID_MIN': 0.5000,
        'N_HRS_PP': 3,
        'MIN_DAYS_PER_MONTH': 29,
        'MIN_NUM_HOURS': 24,
        'PP_PRESENT': -1000,
        'AC_PRESENT': -1000,
        'HEATER_PRESENT': -1000,
        'IS_SOLAR': 0,
        'DO_TOD': 1,
        'DO_GAS': 0,
        'ALGO': 'unsupervised_v2_hourly',
        'UNSUP_MONTHMODE': 'day',
        'UNSUP_MIN_MONTHS': 4,
        'UNSUP_CREATE_HSM_MIN_MONTHS': 8,
        'UNSUP_CREATE_HSM_PREFERRED_MONTHS': 12,
        'UNSUP_SAMPLING_RATE_THRESH': 48,
        'MONTHS_TO_EXPIRE_HVAC_HSM': 1,
        'DAYS_TO_EXPIRE_HVAC_HSM': 120,
        'IGNORE_HSM': 0,
        'IGNORE_CREATE_HSM': 0,
        'SOLAR_ALGO': 'SVM_new50',
        'CAPACITY_PRE_SOLAR': -2000,
        'CAPACITY_SOLAR_MIN': -100,
        'SAMPLING_RATE_EFF': 0.2500,
        'DEBUG': 0,
        'WRITE_LOCAL': 0,
        'WRITE_LOCAL_DIR': 'output/',
        'timesofday': np.array([[0, 5], [6, 11], [12, 18], [19, 23]]),
        'weekbands': np.array([[1, 5], [6, 7]]),
        'seasons': np.array([1, 12]),
        'COLUMN_FAMILY': [],
        'DATA_STREAM': 'GreenButton',
        'VERBOSE': 0,
        'bypassPoolPumpInquiry': 0,
        'isTimeBandAlgo': 0,
        'API_ENV': 'NA',
        'MONTHLY_FLAG': 1,
        'DAILY_FLAG': 0,
        'END_TIME_STAMP': 1.6000e+09,
        'CONFIG_JSON': '+GB/conf-inp.json',
        'MONTH_MODE': 'month',
        'debugHVAC': 1,
        'debugHVACPlots': 1,
        'extraString': '_Smoothed',
        'loggerOn': 0,
        'logger': [],
        'inputSignal': 0,
        'HRS_PADDING': 2,
        'WINDOW_LENGTH': 5,
        'EOMT': 0,
        'previoushsm': [],
        'CREATE_HSM': 1,
        'not_enough_hrs': 0,
        'tempMissing': 0,
        'nummonths': 14,
        'numweeks': 61,
        'HAS_SOLAR_FROM_PROFILE': [],
        'CAPACITY_FROM_PROFILE': [],
        'hsm': [],
        'SHType': 'Unknown',
        'EV_PRESENT': -1000,
        'vacationPeriods': [],
        'numDays': 422,
    })
    return gconfig
