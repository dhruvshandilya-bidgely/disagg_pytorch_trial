"""
Author - Pratap
Date - 13/11/18
Intial params initiated here
"""

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg


def initialize_gbref_params(sampling_rate):
    """
    Ref Config parameters are initiated here

    Parameters:
        sampling_rate (int)     : Sampling rate in seconds
    """
    ref_config = {
        'UUID': '',
        'samplingRate': sampling_rate,
        'LAPDetection': {
            'lowPercentileLimit': 6,
            'lapHalfGap': 3,
            'sampling_rate': sampling_rate,
            'shortenLAPnumHours': 0,
            'shortenLAPdataPnts': sampling_rate,
            'levelFilter': 400 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'TransitionsAboveLimit': 300 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'TransitionsBelowLimit': 30 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'MeanTransitionsAboveLimit': 300 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'MeanTransitionsBelowLimit': 20 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'PerStdTrnc': 0.8 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'EdgCrcTrncMed': 1.7
        },
        'Estimation': {
            'Min_LAPs_Reqd_Estimation': 3,
            'highPercentileLevelForEstimation': 65,
            'threshold_percPtsLyingBelowLevel': 0.5,
            'minNumHourForLAP': 6.0,
            'Max_allowable_LAPs_Reqd_Estimation': 10,
            'MaxRatioLAPEnergyEstimates': 2.25,
            'MinMeanAmpTransitions': 30 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'MaxMeanAmpTransitions': 300 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'MinNumTransitionsBelow': 1.5 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'MaxNumTransitionsAbove': 1.2 * sampling_rate / Cgbdisagg.SEC_IN_HOUR,
            'MinMonthRefEstimate': 15000,
            'MaxMonthRefEstimate': 110000,
            'AmpTrnsLimit': 150 * sampling_rate / 1800,
            'MinAmptorplcWHplsLimit': 40 * sampling_rate / 1800,
            'MinClustLimit': 30 * sampling_rate / 1800,
            'Enerconvtr': 2 * 1800 / sampling_rate,
            'StdThrEst': 1,
            'MinEnrThrLmt': 4 * sampling_rate / 1800
        },
        'Seasonality': {
            'secondHottestMonthSeasonalityScaling': 1.15,
            'upperBoundSeasonalityScaling': 1.25
        }
    }

    return ref_config

