"""
Author - Nisha Agarwal
Date - 9th Oct 2020
Initialize config for clean day masking
"""

# No imports


def get_clean_day_score_config():

    """ Initialize config for clean day masking """

    config = dict()

    mask_clean_days_config = {
        'rolling_avg_window': 3,
        'min_high_cons_limit': 8,
        'min_cons': 1000,
        'bucket_size_factor': 0.003,
        'clustering_fraction': 5,
        "max_bucket_size": 10,
        "points_count_limit": 100
    }

    # 'rolling_avg_window': window size for rolling sun array ,
    # 'min_high_cons_limit': minimum high consumption hours required for masking clean days,
    # 'min_cons': minimum day consumption for masking clean days,
    # 'bucket_size_factor': multiplier to calculate bucket size for resampling,
    # 'clustering_fraction': fraction of resampled data to perform clustering,
    # "max_bucket_size": Max size of bucket used for reducing sample size
    # "points_count_limit": Min number of points required in the reduced sampled set

    config.update({

        "mask_clean_days_config": mask_clean_days_config

    })

    clean_day_score_config = {
        'temp_dev': 20,
        'temp_setpoint': 65,
        'perc_for_mean': 35,
        'perc_for_dev': 25,
        'high_cons_count_factor': 18,
        'cont_high_cons_count_factor': 9,
        "threshold": 15,
        "perc_cap": 95
    }

    # 'temp_dev': deviation in temperature,
    # 'temp_setpoint': mean temperature for calculating temperature score ,
    # 'perc_for_mean': percentile used for calculating mean consumption,
    # 'perc_for_dev': percentile used for calculating std dev consumption,
    # 'high_cons_count_factor': factor used calculation of high consumption points score,
    # 'cont_high_cons_count_factor': factor used calculation of continuous high consumption points score,
    # "threshold": percentile threshold to remove low consumption points

    config.update({

        "clean_day_score_config": clean_day_score_config

    })

    clean_day_score_weightage_config = {
        'temperature_score': 0.1,
        'day_level_energy_score': 0.15,
        'continuous_high_consumption_score': 0.2,
        'continuous_segments_score': 0.2,
        'high_consumption_score': 0.35
    }

    # Weightage of respective score for final cleanliness score calculation
    # 'temperature_score',
    # 'day_level_energy_score',
    # 'continuous_high_consumption_score',
    # 'continuous_segments_score',
    # 'high_consumption_score'

    config.update({

        "clean_day_score_weightage_config": clean_day_score_weightage_config

    })

    return config
