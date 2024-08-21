"""
Author - Prasoon Patidar
Date - 28/04/20
Head Git Branch:  AIAAS-1
Description:  Fixed Schema for writing appliance profile
Schema Library:  https: //github.com/keleshev/schema
"""

from schema import Schema, Or

appliance_profile_schema = \
    Schema(
        {
            "version"    : Or(None, str),
            "profileList": [
                {
                    "start"    : Or(None, int),
                    "end"      : Or(None, int),
                    "dataRange": {
                        "start": Or(None, int),
                        "end"  : Or(None, int)
                    },
                    "2"        : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "ppConsumption": Or(None, float),
                                "appType"      : Or(None, str),
                                "fuelType"     : Or(None, "Electric", "Gas"),
                                "numberOfRuns" : Or(None, [float]),
                                "amplitude"    : Or(None, [float]),
                                "schedule"     : Or(None, [[float]]),
                                "timeOfUsage"  : Or(None, [float])
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "3"        : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "heatingConsumption"       : Or(None, float),
                                "inWorkingHoursAtMonth"    : Or(None, float),
                                "inOffHoursAtMonth"        : Or(None, float),
                                "fuelType"                 : Or(None, "Electric", "Gas"),
                                "secondaryFuelType"        : Or(None, "Electric", "Gas"),
                                "usageModeCount"           : Or(None, int),
                                "usageModeCountConfidence" : Or(None, float),
                                "modeAmplitudes"           : Or(None, [float]),
                                "regressionCoeff"          : Or(None, [float]),
                                "regressionCoeffConfidence": Or(None, [float]),
                                "regressionType"           : Or(None, [str]),
                                "aoHeating"                : Or(None, float),
                                "onDemandHeating"          : Or(None, float),
                                "heatingMeans"             : Or(None, [float]),
                                "heatingStd"               : Or(None, [float]),
                                "timeOfUsage"              : Or(None, [float]),

                                "hasMultipleAppliances"    : Or(None, bool),
                                "numberOfAppliances"       : Or(None, int),

                                "hasSaturationIneff"       : Or(None, bool),
                                "saturationTemperature"    : Or(None, int),
                                "fractionOfSaturatedPts"   : Or(None, float),
                                "hasPreSaturation"         : Or(None, bool),
                                "preSaturationTemperature" : Or(None, int),

                                "hasAbruptConsumption"     : Or(None, bool),
                                "abruptConsumptionSeverity": Or(None, [float]),
                                "abruptAmplitude"          : Or(None, [float]),
                                "abruptHours"              : Or(None, [float]),
                                "abruptTouStarts"          : Or(None, [float]),
                                "abruptTouEnds"            : Or(None, [float]),

                                "hasAbruptAoHvac"          : Or(None, bool),
                                "abruptAoHvacSeverity"     : Or(None, [float]),

                                "hasShortCycling"          : Or(None, bool),
                                "shortCyclingStarts"       : Or(None, [float]),
                                "shortCyclingEnds"         : Or(None, [float]),

                                "hasBehaviorChange"        : Or(None, bool),
                                "behaviorChangeType"       : Or(None, str),
                                "behaviorChangeSeverity"   : Or(None, [float]),

                                "hasApplianceChange"        : Or(None, bool),
                                "applianceChangeDirection"  : Or(None, str),
                                "applianceChangeConfidence" : Or(None, float),

                                "hasApplianceDegradation"   : Or(None, bool),
                                "applianceDegradeConfidence": Or(None, float),

                                "applianceOfficeTimeOverlap": Or(None, bool),
                                "officeTimeOverlapSeverity" : Or(None, [float]),

                                "hasSmartThermostat"        : Or(None, bool),
                                "hasSmartThermostatConf"    : Or(None, float),

                                "applianceUsageShape"            : Or(None, [float]),
                                "applianceUsageHours"            : Or(None, [float])
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "4"        : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "coolingConsumption"       : Or(None, float),
                                "inWorkingHoursAtMonth"    : Or(None, float),
                                "inOffHoursAtMonth"        : Or(None, float),
                                "fuelType"                 : Or(None, "Electric", "Gas"),
                                "secondaryFuelType"        : Or(None, "Electric", "Gas"),
                                "usageModeCount"           : Or(None, int),
                                "usageModeCountConfidence" : Or(None, float),
                                "modeAmplitudes"           : Or(None, [float]),
                                "regressionCoeff"          : Or(None, [float]),
                                "regressionCoeffConfidence": Or(None, [float]),
                                "regressionType"           : Or(None, [str]),
                                "aoCooling"                : Or(None, float),
                                "onDemandCooling"          : Or(None, float),
                                "coolingMeans"             : Or(None, [float]),
                                "coolingStd"               : Or(None, [float]),
                                "timeOfUsage"              : Or(None, [float]),
                                "coolingType"              : Or(None, str),

                                "hasMultipleAppliances"    : Or(None, bool),
                                "numberOfAppliances"       : Or(None, int),

                                "hasSaturationIneff"       : Or(None, bool),
                                "saturationTemperature"    : Or(None, int),
                                "fractionOfSaturatedPts"   : Or(None, float),
                                "hasPreSaturation"         : Or(None, bool),
                                "preSaturationTemperature" : Or(None, int),

                                "hasAbruptConsumption"     : Or(None, bool),
                                "abruptConsumptionSeverity": Or(None, [float]),
                                "abruptAmplitude"          : Or(None, [float]),
                                "abruptHours"              : Or(None, [float]),
                                "abruptTouStarts"          : Or(None, [float]),
                                "abruptTouEnds"            : Or(None, [float]),

                                "hasAbruptAoHvac"          : Or(None, bool),
                                "abruptAoHvacSeverity"     : Or(None, [float]),

                                "hasShortCycling"          : Or(None, bool),
                                "shortCyclingStarts"       : Or(None, [float]),
                                "shortCyclingEnds"         : Or(None, [float]),

                                "hasBehaviorChange"        : Or(None, bool),
                                "behaviorChangeType"       : Or(None, str),
                                "behaviorChangeSeverity"   : Or(None, [float]),

                                "hasApplianceChange"        : Or(None, bool),
                                "applianceChangeDirection"  : Or(None, str),
                                "applianceChangeConfidence" : Or(None, float),

                                "hasApplianceDegradation"   : Or(None, bool),
                                "applianceDegradeConfidence": Or(None, float),

                                "applianceOfficeTimeOverlap": Or(None, bool),
                                "officeTimeOverlapSeverity" : Or(None, [float]),

                                "hasSmartThermostat"             : Or(None, bool),
                                "hasSmartThermostatConf"         : Or(None, float),

                                "applianceUsageShape"            : Or(None, [float]),
                                "applianceUsageHours"            : Or(None, [float])
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "7"        : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "whConsumption"       : Or(None, float),
                                "appType"             : Or(None, str),
                                "fuelType"            : Or(None, "Electric", "Gas"),
                                "runsCount"           : Or(None, int),
                                "amplitude"           : Or(None, float),
                                "amplitudeConfidence" : Or(None, float),
                                "dailyThinPulseCount" : Or(None, int),
                                "passiveUsageFraction": Or(None, float),
                                "activeUsageFraction" : Or(None, float),
                                "timeOfUsage"         : Or(None, [float])
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "8"        : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "fuelType"     : Or(None, "Electric", "Gas"),
                                "aoConsumption": Or(None, float)
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "9"        : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "refConsumption"     : Or(None, float),
                                "amplitude"          : Or(None, float),
                                "multipleRef"        : Or(None, bool),
                                "summerAmplitude"    : Or(None, float),
                                "winterAmplitude"    : Or(None, float),
                                "transitionAmplitude": Or(None, float)
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "16"       : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "solarPropensity"        : Or(None, float),
                                "solarGeneration"        : Or(None, float),
                                "requiredPanelCapacity"  : Or(None, float),
                                "breakEvenPeriod"        : Or(None, float),
                                "chunkStart"             : Or(None, [int]),
                                "chunksEnd"              : Or(None, [int]),
                                "chunksConfidence"       : Or(None, [float]),
                                "excessGeneration"       : Or(None, float),
                                "solarCapacity"          : Or(None, float),
                                "solarCapacityConfidence": Or(None, float),
                                "timeOfUsage"            : Or(None, [float]),
                                "isNewUser"            : Or(None, bool),
                                "isOldUser"            : Or(None, bool),
                                "startDate"             : Or(None, int),
                                "endDate"               : Or(None, int)
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "18"       : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "evPropensity": Or(None, float),
                                "evConsumption"          : Or(None, float),
                                "chargerType"            : Or(None, "L1", "L2", "L3"),
                                "amplitude"              : Or(None, float),
                                "chargingInstanceCount"  : Or(None, int),
                                "averageChargingDuration": Or(None, float),
                                "timeOfUsage"            : Or(None, [float])
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "71"       : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "lightingConsumption": Or(None, float),
                                "morningCapacity"    : Or(None, float),
                                "eveningCapacity"    : Or(None, float),
                                "timeOfUsage"        : Or(None, [float])
                            },
                            "debugAttributes"    : Or({str: object}, {})
                        }
                    ],
                    "81"       : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "operationalLoadConsumption": Or(None, float),
                                "samplingRate"              : Or(None, int),
                                "operationalLoadAtHour"     : Or(None, [float]),
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    "82"       : [
                        {
                            "validity"           : Or(None, {
                                "start": Or(None, int),
                                "end"  : Or(None, int)
                            }),
                            "isPresent"          : Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count"              : Or(None, int),
                            "attributes"         : {
                                "anomalousLoadConsumption": Or(None, float),
                                "dayOfUsage"              : Or(None, [float]),
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    "66": [
                        {
                            "validity": Or(None, {
                                "start": Or(None, int),
                                "end": Or(None, int)
                            }),
                            "isPresent": Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count": Or(None, int),
                            "attributes": {
                                "timeOfUsage"            : Or(None, [float]),
                                "entertainmentConsumption"  : Or(None, float),
                            },
                            "debugAttributes": {

                            }
                        }
                    ],
                    "5": [
                        {
                            "validity": Or(None, {
                                "start": Or(None, int),
                                "end": Or(None, int)
                            }),
                            "isPresent": Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count": Or(None, int),
                            "attributes": {
                                "timeOfUsage": Or(None, [float]),
                                "cookingConsumption": Or(None, float),
                            },
                            "debugAttributes": {

                            }
                        }
                    ],
                    "59": [
                        {
                            "validity": Or(None, {
                                "start": Or(None, int),
                                "end": Or(None, int)
                            }),
                            "isPresent": Or(None, bool),
                            "detectionConfidence": Or(None, float),
                            "count": Or(None, int),
                            "attributes": {
                                "timeOfUsage": Or(None, [float]),
                                "laundryConsumption": Or(None, float),
                            },
                            "debugAttributes": {

                            }
                        }
                    ]
                }
            ]
        })
