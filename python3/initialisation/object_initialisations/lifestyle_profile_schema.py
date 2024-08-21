"""
Author - Prasoon Patidar
Date - 3rd June 2020
Head Git Branch:  DS-1040
Description:  Fixed Schema for writing lifestyle profile
Schema Library:  https://github.com/keleshev/schema
"""

from schema import Schema, Or

lifestyle_profile_schema = \
    Schema(
        {
            "version"    : Or(None, str),
            "profileList": [
                {
                    "start"         : Or(None, int),
                    "end"           : Or(None, int),
                    "dataRange"     : {
                        "start": Or(None, int),
                        "end"  : Or(None, int)
                    },
                    "validity"      : {
                        "start": Or(None, int),
                        "end"  : Or(None, int)
                    },
                    "lifestyleid_1" : {
                        "name"           : "SeasonalLoadType",
                        "value"          : Or(None, str),
                        "attributes"     : {
                            "consumptionLevelWinter"    : Or(None, str),
                            "consumptionLevelSummer"    : Or(None, str),
                            "consumptionLevelTransition": Or(None, str),
                        },
                        "debugAttributes": {
                            "winterMonths"    : Or(None, [int]),
                            "summerMonths"    : Or(None, [int]),
                            "transitionMonths": Or(None, [int]),

                        }
                    },
                    "lifestyleid_2" : {
                        "name"           : "OfficeGoer",
                        "value"          : Or(None, bool),
                        "attributes"     : {
                            "officeGoerProbability": Or(None, float)
                        },
                        "debugAttributes": {
                            "lowcoolingConstant"     : Or(None, float),
                            "seasonKeys"             : Or(None, [str]),
                            "seasonalProbabilities"  : Or(None, [Or(None, float)]),
                            "lowcoolingProbabilities": Or(None, [Or(None, float)])
                        }
                    },
                    "lifestyleid_3" : {
                        "name"           : "ActiveUser",
                        "value"          : Or(None, bool),
                        "attributes"     : {
                            "activeUserProbability": Or(None, float)
                        },
                        "debugAttributes": {
                            "seasonKeys"              : Or(None, [str]),
                            "seasonalProbabilities"   : Or(None, [Or(None, float)]),
                            "seasonFractions"         : Or(None, [Or(None, float)]),
                            "baseloadConsumption"     : Or(None, [Or(None, float)]),
                            "totalConsumption"        : Or(None, [Or(None, float)]),
                            "nonBaseloadFraction"     : Or(None, [Or(None, float)]),
                            "averageActivity"         : Or(None, [Or(None, float)]),
                            "withinDayDeviation"      : Or(None, [Or(None, float)]),
                            "acrossDayDeviation"      : Or(None, [Or(None, float)]),
                            "winterFeaturesNormed"    : Or(None, [Or(None, float)]),
                            "summerFeaturesNormed"    : Or(None, [Or(None, float)]),
                            "transitionFeaturesNormed": Or(None, [Or(None, float)]),
                            "activityThreshold"       : Or(None, float)
                        }
                    },
                    "lifestyleid_4" : {
                        "name"           : "DormantUser",
                        "value"          : Or(None, bool),
                        "attributes"     : {
                            "dormantUserProbability": Or(None, float)
                        },
                        "debugAttributes": {

                        }
                    },
                    "lifestyleid_5" : {
                        "name"           : "WeekendWarrior",
                        "value"          : Or(None, bool),
                        "attributes"     : {
                            "weekendWarriorProbability": Or(None, float)
                        },
                        "debugAttributes": {
                            "seasonKeys"           : Or(None, [str]),
                            "seasonalProbabilities": Or(None, [Or(None, float)])
                        }
                    },
                    "lifestyleid_6" : {
                        "name"           : "HourFractions",
                        "value"          : Or(None, [Or(None, float)]),
                        "attributes"     : {
                            "hourFractionWeekday": Or(None, [Or(None, float)]),
                            "hourFractionWeekend": Or(None, [Or(None, float)])
                        },
                        "debugAttributes": {

                        }
                    },
                    "lifestyleid_7" : {
                        "name"           : "DailyLoadType",
                        "value"          : Or(None, str),
                        "attributes"     : {
                            "loadtypeConfidence"     : Or(None, float),
                            "consumptionLevel"       : Or(None, str),
                            "clusterNames"           : Or(None, [str]),
                            "clusterFractionsAll"    : Or(None, [Or(None, float)]),
                            "clusterFractionsWeekday": Or(None, [Or(None, float)]),
                            "clusterFractionsWeekend": Or(None, [Or(None, float)])
                        },
                        "debugAttributes": {
                            "peaksInfo": Or(None, str)
                        }
                    },
                    "lifestyleid_8" : {
                        "name"           : "WakeUpTime",
                        "value"          : Or(None, float),
                        "attributes"     : {
                            "confidenceInterval": Or(None, float)
                        },
                        "debugAttributes": {

                        }

                    },
                    "lifestyleid_9" : {
                        "name"           : "SleepTime",
                        "value"          : Or(None, float),
                        "attributes"     : {
                            "confidenceInterval": Or(None, float)
                        },
                        "debugAttributes": {

                        }
                    },
                    "lifestyleid_10": {
                        "name"           : "VacationPercentage",
                        "value"          : Or(None, float),
                        "attributes"     : {

                        },
                        "debugAttributes": {

                        }
                    },
                    "lifestyleid_11": {
                        "name"           : "OperationalHoursSMB",
                        "attributes"     : {
                            "samplingRate"       : Or(None, int),
                            "operationalHours"   : {'openhour'   : Or(None, int),
                                                    'openminute' : Or(None, int),
                                                    'closehour'  : Or(None, int),
                                                    'closeminute': Or(None, int)},
                            "dayOperationalHours": Or(None, [Or(None, dict)]),
                        },
                        "debugAttributes": {

                        }
                    }

                }
            ]
        }
    )
