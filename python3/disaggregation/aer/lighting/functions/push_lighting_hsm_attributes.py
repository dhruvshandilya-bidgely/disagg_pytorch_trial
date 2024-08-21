"""
Author - Mayank Sharan
Date - 26/11/2019
detect seasonal tries to detect seasonal consumption patterns in the data
"""

# No imports in this module


def push_lighting_hsm_attributes(results):

    """Push Lighting HSM Attributes"""

    attributes = {
        'lighting_band': results.get('lighting_band').astype(float),
        'morning_capacity': results.get('morning_capacity'),
        'evening_capacity': results.get('evening_capacity'),
        'morning_interpolation': results.get('morning_interpolation'),
        'evening_interpolation': results.get('evening_interpolation'),
        'smoothing_noise_bound': results.get('smoothing_noise_bound'),
        'period': results.get('period'),
        'DarkestMonth_nightHours': results.get('DarkestMonth_nightHours'),
        'secondLightestMonth_nightHours': results.get('secondLightestMonth_nightHours'),
        'maxLightingPerDay': results.get('maxLightingPerDay'),
        'scaling': results.get('scaling'),
    }

    return attributes
