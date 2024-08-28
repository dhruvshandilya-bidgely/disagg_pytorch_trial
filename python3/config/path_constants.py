"""
Contains Folder related constants
"""


class PathConstants:

    """Class containing all path constants"""

    # Folder related constants

    CACHE_DIR = '../cache_data/'
    VIP_DIR = '/tmp/vip_results/'
    VIP_FILE = 'vip_disagg.csv'
    LOG_DIR = '../plots/'
    LOG_DIR_PRIORITY = '/var/log/bidgely/pyamidisagg/pydisaggpriority/'
    WEATHER_DATA_CACHE_DIR = "tmp/weatherDataCache"

    # Constants needed for pre and post disagg ops

    MIN_TEMP_NOT_INTERPOLATE = 90

    # Configuration variables for downloading and storing trained models from s3 buckets

    FILES_LOCAL_ROOT_DIR = '../disagg_files/'
    HYBRID_V2_CONFIG_FILES_LOCAL_ROOT_DIR = '/tmp/disagg_static_files/'
    HYBRID_V2_CONFIG_FILES_LOCAL_PATH = '/models/itemization/pilot_config/'

    PATH_SUFFIX_FOR_HYBRID_V2_CONFIG = 'v2/'

    VIP_FILE_S3_BUCKET = 'bidgely-ds'
    VIP_FILE_S3_PREFIX = 'disagg_vip/'

    FILES_S3_BUCKET_DICT = {
        'dev': 'disagg-static-files-dev',
        'ds': 'disagg-static-files-ds',
        'nonprodqa': 'disagg-static-files-nonprodqa',
        'prod-na': 'disagg-static-files-prod-na',
        'prod-eu': 'disagg-static-files-prod-eu',
        'prod-jp': 'disagg-static-files-prod-jp',
        'prod-ca': 'disagg-static-files-prod-ca',
        'prod-na-2': 'disagg-static-files-prod-na',
        'preprod-na': 'disagg-static-files-preprod-na',
        'qaperfenv': 'disagg-static-files-qaperfenv',
        'uat': 'disagg-static-files-uat',
        'productqa': 'disagg-static-files-productqa'
    }

    HYBRID_V2_CONFIG_FILES_S3_BUCKET_DICT = {
        'dev': 'hybrid-models-dev',
        'ds': 'hybrid-models-ds',
        'nonprodqa': 'hybrid-models-nonprodqa',
        'prod-na': 'hybrid-models-prod-na',
        'prod-eu': 'hybrid-models-prod-eu',
        'prod-jp': 'hybrid-models-prod-jp',
        'prod-ca': 'hybrid-models-prod-ca',
        'prod-na-2': 'hybrid-models-prod-na',
        'preprod-na': 'hybrid-models-preprod-na',
        'qaperfenv': 'hybrid-models-qaperfenv',
        'uat': 'hybrid-models-uat',
        'productqa': 'hybrid-models-productqa',
    }

    MODULE_FILES_ROOT_DIR = {
        'wh': 'models/wh/',
        'ev': 'models/ev/',
        'ref': 'models/ref/',
        'ev_propensity': 'models/ev_propensity/',
        'solar': 'models/solar/',
        'lf': 'models/lifestyle/',
        'item': 'models/itemization/pilot_config/',
        'hvac_inefficiency': 'models/hvac_inefficiency/'
    }

    MODULE_OUTPUT_FILES_DIR = {
        'wh': 'wh_data/',
        'solar': 'solar_data/',
        'ev': 'ev_data/'
    }
