"""
Author - Prasoon Patidar
Date - 03rd June 2020
Head Git Branch:  DS-1040
Description:  Empty lifestyle profile to initiate in disagg output object
"""


def init_empty_lifestyle_profile():

    """Initialize empty lifestyle profile dictionary"""

    empty_lifestyle_profile = {

        "version": None,
        "profileList": [
            {
                "start": None,
                "end": None,
                "dataRange": {
                    "start": None,
                    "end": None,
                },
                "validity": {
                    "start": None,
                    "end": None,
                },
                "lifestyleid_1": {
                    "name": "SeasonalLoadType",
                    "value": None,
                    "attributes": {
                        "consumptionLevelWinter": None,
                        "consumptionLevelSummer": None,
                        "consumptionLevelTransition": None,
                    },
                    "debugAttributes": {
                        "winterMonths": None,
                        "summerMonths": None,
                        "transitionMonths": None,

                    }
                },
                "lifestyleid_2": {
                    "name": "OfficeGoer",
                    "value": None,
                    "attributes": {
                        "officeGoerProbability": None,
                    },
                    "debugAttributes": {
                        "lowcoolingConstant": None,
                        "seasonKeys": None,
                        "seasonalProbabilities": None,
                        "lowcoolingProbabilities": None
                    }
                },
                "lifestyleid_3": {
                    "name": "ActiveUser",
                    "value": None,
                    "attributes": {
                        "activeUserProbability": None,
                    },
                    "debugAttributes": {
                        "seasonKeys": None,
                        "seasonalProbabilities": None,
                        "seasonFractions": None,
                        "baseloadConsumption": None,
                        "totalConsumption": None,
                        "nonBaseloadFraction": None,
                        "averageActivity": None,
                        "withinDayDeviation": None,
                        "acrossDayDeviation": None,
                        "winterFeaturesNormed": None,
                        "summerFeaturesNormed": None,
                        "transitionFeaturesNormed": None,
                        "activityThreshold": None
                    }
                },
                "lifestyleid_4" : {
                    "name": "DormantUser",
                    "value": None,
                    "attributes": {
                        "dormantUserProbability": None,
                    },
                    "debugAttributes": {

                    }
                },
                "lifestyleid_5": {
                    "name": "WeekendWarrior",
                    "value": None,
                    "attributes": {
                        "weekendWarriorProbability": None,
                    },
                    "debugAttributes": {
                        "seasonKeys": None,
                        "seasonalProbabilities": None
                    }
                },
                "lifestyleid_6": {
                    "name": "HourFractions",
                    "value": None,
                    "attributes": {
                        "hourFractionWeekday": None,
                        "hourFractionWeekend": None,
                    },
                    "debugAttributes": {

                    }
                },
                "lifestyleid_7": {
                    "name": "DailyLoadType",
                    "value": None,
                    "attributes": {
                        "loadtypeConfidence": None,
                        "consumptionLevel": None,
                        "clusterNames": None,
                        "clusterFractionsAll": None,
                        "clusterFractionsWeekday": None,
                        "clusterFractionsWeekend": None,
                    },
                    "debugAttributes": {
                        "peaksInfo": None,
                    }
                },
                "lifestyleid_8": {
                    "name": "WakeUpTime",
                    "value": None,
                    "attributes": {
                        "confidenceInterval": None,
                    },
                    "debugAttributes": {

                    }

                },
                "lifestyleid_9": {
                    "name": "SleepTime",
                    "value": None,
                    "attributes": {
                        "confidenceInterval": None,
                    },
                    "debugAttributes": {

                    }
                },
                "lifestyleid_10": {
                    "name": "VacationPercentage",
                    "value": None,
                    "attributes": {

                    },
                    "debugAttributes": {

                    }
                },
                "lifestyleid_11": {
                    "name": "OperationalHoursSMB",
                    "attributes": {
                        "samplingRate": None,
                        "operationalHours": {'openhour': None,
                                             'openminute': None,
                                             'closehour': None,
                                             'closeminute': None},
                        "dayOperationalHours": None,
                    },
                    "debugAttributes": {

                    }
                }
            }
        ]
    }

    return empty_lifestyle_profile
