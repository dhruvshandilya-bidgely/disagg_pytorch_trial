"""
Author - Prasoon Patidar
Date - 03rd June 2020
Head Git Branch:  DS-1040
Description:  Empty appliance profile to initiate in disagg output object
"""

# Import functions from within the project
from python3.config.mappings.get_app_id import get_app_id


def init_empty_appliance_profile():

    """Initialize empty appliance profile dictionary"""

    empty_appliance_profile = \
        {
            "version"    : None,
            "profileList": [
                {
                    "start"    : None,
                    "end"      : None,
                    "dataRange": {
                        "start": None,
                        "end"  : None
                    },
                    str(get_app_id('pp'))        : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "ppConsumption": None,
                                "appType"      : None,
                                "fuelType"     : None,
                                "numberOfRuns" : None,
                                "amplitude"    : None,
                                "schedule"     : None,
                                "timeOfUsage"  : None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('sh'))        : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "heatingConsumption"       : None,
                                "inWorkingHoursAtMonth"    : None,
                                "inOffHoursAtMonth"        : None,
                                "fuelType"                 : None,
                                "secondaryFuelType"        : None,
                                "usageModeCount"           : None,
                                "usageModeCountConfidence" : None,
                                "modeAmplitudes"           : None,
                                "regressionCoeff"          : None,
                                "regressionCoeffConfidence": None,
                                "regressionType"           : None,
                                "aoHeating"                : None,
                                "onDemandHeating"          : None,
                                "heatingMeans"             : None,
                                "heatingStd"               : None,
                                "timeOfUsage"              : None,

                                "hasMultipleAppliances": None,
                                "numberOfAppliances": None,

                                "hasSaturationIneff": None,
                                "saturationTemperature": None,
                                "fractionOfSaturatedPts": None,
                                "hasPreSaturation": None,
                                "preSaturationTemperature": None,

                                "hasAbruptConsumption": None,
                                "abruptConsumptionSeverity": None,
                                "abruptAmplitude": None,
                                "abruptHours": None,
                                "abruptTouStarts": None,
                                "abruptTouEnds": None,

                                "hasAbruptAoHvac": None,
                                "abruptAoHvacSeverity": None,

                                "hasShortCycling": None,
                                "shortCyclingStarts": None,
                                "shortCyclingEnds": None,

                                "hasBehaviorChange": None,
                                "behaviorChangeType": None,
                                "behaviorChangeSeverity": None,

                                "hasApplianceChange": None,
                                "applianceChangeDirection": None,
                                "applianceChangeConfidence": None,

                                "hasApplianceDegradation": None,
                                "applianceDegradeConfidence": None,

                                "applianceOfficeTimeOverlap": None,
                                "officeTimeOverlapSeverity": None,

                                "hasSmartThermostat": None,
                                "hasSmartThermostatConf": None,

                                "applianceUsageShape": None,
                                "applianceUsageHours": None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('ac'))        : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "coolingConsumption"       : None,
                                "inWorkingHoursAtMonth"    : None,
                                "inOffHoursAtMonth"        : None,
                                "fuelType"                 : None,
                                "secondaryFuelType"        : None,
                                "usageModeCount"           : None,
                                "usageModeCountConfidence" : None,
                                "modeAmplitudes"           : None,
                                "regressionCoeff"          : None,
                                "regressionCoeffConfidence": None,
                                "regressionType"           : None,
                                "aoCooling"                : None,
                                "onDemandCooling"          : None,
                                "coolingMeans"             : None,
                                "coolingStd"               : None,
                                "coolingType"              : None,
                                "timeOfUsage"              : None,

                                "hasMultipleAppliances": None,
                                "numberOfAppliances": None,

                                "hasSaturationIneff": None,
                                "saturationTemperature": None,
                                "fractionOfSaturatedPts": None,
                                "hasPreSaturation": None,
                                "preSaturationTemperature": None,

                                "hasAbruptConsumption": None,
                                "abruptConsumptionSeverity": None,
                                "abruptAmplitude": None,
                                "abruptHours": None,
                                "abruptTouStarts": None,
                                "abruptTouEnds": None,

                                "hasAbruptAoHvac": None,
                                "abruptAoHvacSeverity": None,

                                "hasShortCycling": None,
                                "shortCyclingStarts": None,
                                "shortCyclingEnds": None,

                                "hasBehaviorChange": None,
                                "behaviorChangeType": None,
                                "behaviorChangeSeverity": None,

                                "hasApplianceChange": None,
                                "applianceChangeDirection": None,
                                "applianceChangeConfidence": None,

                                "hasApplianceDegradation": None,
                                "applianceDegradeConfidence": None,

                                "applianceOfficeTimeOverlap": None,
                                "officeTimeOverlapSeverity": None,

                                "hasSmartThermostat": None,
                                "hasSmartThermostatConf": None,

                                "applianceUsageShape": None,
                                "applianceUsageHours": None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('wh'))        : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "whConsumption"       : None,
                                "appType"             : None,
                                "fuelType"            : None,
                                "runsCount"           : None,
                                "amplitude"           : None,
                                "amplitudeConfidence" : None,
                                "dailyThinPulseCount" : None,
                                "passiveUsageFraction": None,
                                "activeUsageFraction" : None,
                                "timeOfUsage"         : None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('ao'))        : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "fuelType"     : None,
                                "aoConsumption": None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('ref'))        : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "refConsumption"     : None,
                                "amplitude"          : None,
                                "multipleRef"        : None,
                                "summerAmplitude"    : None,
                                "winterAmplitude"    : None,
                                "transitionAmplitude": None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('solar'))       : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "solarPropensity"        : None,
                                "solarGeneration"        : None,
                                "requiredPanelCapacity"  : None,
                                "breakEvenPeriod"        : None,
                                "chunkStart"             : None,
                                "chunksEnd"              : None,
                                "chunksConfidence"       : None,
                                "excessGeneration"       : None,
                                "solarCapacity"          : None,
                                "solarCapacityConfidence": None,
                                "timeOfUsage"            : None,
                                "isNewUser"            : None,
                                "isOldUser"            : None,
                                "startDate"             : None,
                                "endDate"               : None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('ev'))       : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "evPropensity"           : None,
                                "evConsumption"          : None,
                                "chargerType"            : None,
                                "amplitude"              : None,
                                "chargingInstanceCount"  : None,
                                "averageChargingDuration": None,
                                "timeOfUsage"            : None
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('li'))       : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "lightingConsumption": None,
                                "morningCapacity"    : None,
                                "eveningCapacity"    : None,
                                "timeOfUsage"        : []
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],

                    "66": [
                        {
                            "validity": None,
                            "isPresent": None,
                            "detectionConfidence": None,
                            "count": None,
                            "attributes": {
                                "timeOfUsage": None,
                                "entertainmentConsumption": None,
                            },
                            "debugAttributes": {

                            }
                        }
                    ],
                    "5": [
                        {
                            "validity": None,
                            "isPresent": None,
                            "detectionConfidence": None,
                            "count": None,
                            "attributes": {
                                "timeOfUsage": None,
                                "cookingConsumption": None,
                            },
                            "debugAttributes": {

                            }
                        }
                    ],
                    "59": [
                        {
                            "validity": None,
                            "isPresent": None,
                            "detectionConfidence": None,
                            "count": None,
                            "attributes": {
                                "timeOfUsage": None,
                                "laundryConsumption": None,
                            },
                            "debugAttributes": {

                            }
                        }
                    ],

                    str(get_app_id('op'))       : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "operationalLoadConsumption": None,
                                "samplingRate"              : None,
                                "operationalLoadAtHour"     : None,
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ],
                    str(get_app_id('x-ao'))       : [
                        {
                            "validity"           : None,
                            "isPresent"          : None,
                            "detectionConfidence": None,
                            "count"              : None,
                            "attributes"         : {
                                "anomalousLoadConsumption": None,
                                "dayOfUsage"              : None,
                            },
                            "debugAttributes"    : {

                            }
                        }
                    ]

                }
            ]
        }
    return empty_appliance_profile
