"""
Author - Abhinav Srivastava
Date - 22nd Sep 2020
Fill heating profile for user"
"""

# Import python packages


def get_heating_profile(run_attributes, base, month_consumptions, billcycle_start, billcycle_end, col_idx, tou_attributes):

    """Fill heating profile for user"""

    heating_profile = base['3'][0]

    sh_ao_idx = col_idx.get('sh_ao_idx')
    sh_od_idx = col_idx.get('sh_od_idx')

    if len(month_consumptions) > 0:
        on_demand_heating = month_consumptions[:, sh_od_idx][0]
        ao_heating = month_consumptions[:, sh_ao_idx][0]
        total_heating = on_demand_heating + ao_heating
    else:
        on_demand_heating = 0
        ao_heating = 0
        total_heating = 0

    sh_run = run_attributes.get('sh')

    heating_profile['validity'] = dict()
    heating_profile['validity']['start'] = int(billcycle_start)
    heating_profile['validity']['end'] = int(billcycle_end)
    heating_profile['isPresent'] = sh_run.get('isPresent')
    heating_profile['detectionConfidence'] = sh_run.get('detectionConfidence')
    heating_profile['count'] = sh_run.get('count')
    heating_profile['attributes']['heatingConsumption'] = float(total_heating)
    heating_profile['attributes']['fuelType'] = sh_run.get('fuelType')
    heating_profile['attributes']['secondaryFuelType'] = sh_run.get('secondaryFuelType')
    heating_profile['attributes']['usageModeCount'] = sh_run.get('usageModeCount')
    heating_profile['attributes']['usageModeCountConfidence'] = sh_run.get('usageModeCountConfidence')
    heating_profile['attributes']['modeAmplitudes'] = sh_run.get('modeAmplitudes')
    heating_profile['attributes']['regressionCoeff'] = sh_run.get('regressionCoeff')
    heating_profile['attributes']['regressionCoeffConfidence'] = sh_run.get('regressionCoeffConfidence')
    heating_profile['attributes']['regressionType'] = sh_run.get('regressionType')
    heating_profile['attributes']['aoHeating'] = float(ao_heating)
    heating_profile['attributes']['onDemandHeating'] = float(on_demand_heating)
    heating_profile['attributes']['heatingMeans'] = sh_run.get('heatingMeans')
    heating_profile['attributes']['heatingStd'] = sh_run.get('heatingStd')
    heating_profile['attributes']['timeOfUsage'] = tou_attributes.get('sh').get(billcycle_start)

    return heating_profile
