"""
Author - Abhinav Srivastava
Date - 22nd Sep 2020
Fill cooling profile for user"
"""

# Import python packages


def get_cooling_profile(run_attributes, base, month_consumptions, billcycle_start, billcycle_end, col_idx, tou_attributes):

    """Fill cooling profile for user"""

    cooling_profile = base['4'][0]
    ac_ao_idx = col_idx.get('ac_ao_idx')
    ac_od_idx = col_idx.get('ac_od_idx')

    if len(month_consumptions) > 0:
        on_demand_cooling = month_consumptions[:, ac_od_idx][0]
        ao_cooling = month_consumptions[:, ac_ao_idx][0]
        total_cooling = on_demand_cooling + ao_cooling
    else:
        on_demand_cooling = 0
        ao_cooling = 0
        total_cooling = 0

    ac_run = run_attributes.get('ac')

    cooling_profile['validity'] = dict()
    cooling_profile['validity']['start'] = int(billcycle_start)
    cooling_profile['validity']['end'] = int(billcycle_end)
    cooling_profile['isPresent'] = ac_run.get('isPresent')
    cooling_profile['detectionConfidence'] = ac_run.get('detectionConfidence')
    cooling_profile['count'] = ac_run.get('count')
    cooling_profile['attributes']['coolingConsumption'] = float(total_cooling)
    cooling_profile['attributes']['fuelType'] = ac_run.get('fuelType')
    cooling_profile['attributes']['secondaryFuelType'] = ac_run.get('secondaryFuelType')
    cooling_profile['attributes']['usageModeCount'] = ac_run.get('usageModeCount')
    cooling_profile['attributes']['usageModeCountConfidence'] = ac_run.get('usageModeCountConfidence')
    cooling_profile['attributes']['modeAmplitudes'] = ac_run.get('modeAmplitudes')
    cooling_profile['attributes']['regressionCoeff'] = ac_run.get('regressionCoeff')
    cooling_profile['attributes']['regressionCoeffConfidence'] = ac_run.get('regressionCoeffConfidence')
    cooling_profile['attributes']['regressionType'] = ac_run.get('regressionType')
    cooling_profile['attributes']['aoCooling'] = float(ao_cooling)
    cooling_profile['attributes']['onDemandCooling'] = float(on_demand_cooling)
    cooling_profile['attributes']['coolingMeans'] = ac_run.get('coolingMeans')
    cooling_profile['attributes']['coolingStd'] = ac_run.get('coolingStd')
    cooling_profile['attributes']['timeOfUsage'] = tou_attributes.get('sh').get(billcycle_start)
    cooling_profile['attributes']['coolingType'] = ac_run.get('coolingType')

    return cooling_profile
