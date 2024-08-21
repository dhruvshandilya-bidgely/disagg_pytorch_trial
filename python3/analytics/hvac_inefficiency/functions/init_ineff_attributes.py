"""
Author - Anand Kumar Singh
Date - 19th Feb 2021
Call the HVAC inefficiency module and get output
"""

def initialize_ineff_attributes():

    """
    Function to initialize inefficiency attributes of ac and sh

    Returns:
        sh_attributes (dict) : Dictionary containing initialized sh ineff attributes
        ac_attributes (dict) : Dictionary containing initialized ac ineff attributes
    """

    sh_attributes = {
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
    }

    ac_attributes = {
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
    }

    return sh_attributes, ac_attributes
