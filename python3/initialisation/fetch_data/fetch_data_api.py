"""
Author - Mayank Sharan
Date - 22/09/18
Fetches all data for the user using APIs and returns in a single dictionary
"""

# Import python packages
import logging
from datetime import datetime

# Import functions from within the project
from python3.utils.time.get_time_diff import get_time_diff
from python3.initialisation.fetch_data.cache_input_data import cache_input_data
from python3.initialisation.fetch_data.fetch_meta_object import fetch_meta_object
from python3.initialisation.fetch_data.backend_data_provider import BackendDataProvider
from python3.initialisation.fetch_data.fetch_request_object import get_backend_data_fetch_api
from python3.utils.weather_utils.weather_data_prep_utils import combine_raw_data_and_weather_data
from python3.initialisation.object_initialisations.init_pipeline_input_objects import init_pipeline_input_objects


def get_empty_list_status(input_data, pipeline_run_data, weather_analytics_data):
    """
    Function to identify which data is an empty list
    Parameters:
         input_data                 (np.ndarray)        : Input data
         pipeline_run_data          (np.ndarray)        : Pipeline run data containing all run mode infos
         weather_analytics_data     (list)              : Weather data + Derived weather data
    Returns:
        empty_list_raw_data         (Boolean)           : Boolean whether input data is empty
        empty_list_pipeline_run_data (Boolean)          : Boolean whether pipeline_run_data is empty
        empty_list_weather_data     (Boolean)           : Boolean whether weather_analytics_data is empty
    """
    # Assigning variables for empty lists checks

    empty_list_raw_data = empty_list_pipeline_run_data = empty_list_weather_data = False
    if input_data is not None and len(input_data) == 0:
        empty_list_raw_data = True
    if pipeline_run_data is not None and len(pipeline_run_data) == 0:
        empty_list_pipeline_run_data = True
    if weather_analytics_data is not None and len(weather_analytics_data) == 0:
        empty_list_weather_data = True

    return empty_list_raw_data, empty_list_pipeline_run_data, empty_list_weather_data


def fetch_data_api(fetch_params, config_params, root_logger):

    """
    Parameters:
        fetch_params        (dict)              : Contains uuid, t_start, t_end and disagg_mode
        config_params       (dict)              : Dictionary with all custom run parameters provided
        root_logger         (dict)              : The root logger from which to get the child logger

    Returns:
        pipeline_input_object (dict)              : Dictionary containing all inputs to run the pipeline
        delete_message        (bool)              : Boolean containing the status of data fetch
    """

    logger_base = root_logger.get('logger').getChild('fetch_data_api')
    logger = logging.LoggerAdapter(logger_base, root_logger.get('logging_dict'))
    logging_dict = root_logger.get('logging_dict')

    # Initialize dictionary to pass logger to functions

    logger_pass = {
        'logger_base': logger_base,
        'logging_dict': logging_dict,
    }

    # Initialise variables for the data fetch

    t_start = fetch_params.get('t_start')
    t_end = fetch_params.get('t_end')

    # Fetch request object

    logger.info('Data requested | from %d to %d', t_start, t_end)

    t_before_fetch = datetime.now()

    # Get the URL for data fetch

    url, max_tries_primary, url_sec, max_tries_secondary = get_backend_data_fetch_api(fetch_params, logger_pass)

    # Fetch the required user data

    input_data, pipeline_run_data, weather_analytics_data, status_code, req_object = \
        BackendDataProvider(fetch_params, url, url_sec, max_tries_primary, max_tries_secondary, logger).fetch_data()
    t_after_fetch = datetime.now()

    logger.info('Data fetch took | %.3f s', get_time_diff(t_before_fetch, t_after_fetch))

    # Get empty list status of the data fetched

    empty_list_raw_data, empty_list_pipeline_run_data, empty_list_weather_data = \
        get_empty_list_status(input_data, pipeline_run_data, weather_analytics_data)

    # Data fetch issues checks and handling

    if input_data is None or pipeline_run_data is None or weather_analytics_data is None or \
            empty_list_raw_data or empty_list_pipeline_run_data or empty_list_weather_data:

        logger.info('Data fetch failed | HTTP status %s', str(status_code))

        if status_code == 400:
            delete_message = True
            logger.info('Backend data fetch rejection reason | %s',
                        str(req_object.get('error').get('message')).replace('\n', ' '))

        elif status_code in [500, 501, 502, 503, 504, 505]:
            delete_message = False

        elif empty_list_raw_data or empty_list_pipeline_run_data or empty_list_weather_data:
            delete_message = True
            logger.info('Received Empty list in Raw data : %s | Empty list in Weather data : %s | Empty list in '
                        'Pipeline run data : %s | ', str(empty_list_raw_data), str(empty_list_weather_data),
                        str(empty_list_pipeline_run_data))
        else:
            delete_message = True

        return None, delete_message

    # Log all pipeline events received to get a better understanding of how the code is going to run

    num_pipeline_events = len(pipeline_run_data.get('gbDisaggEvents'))
    logger.info('Number of pipeline events received | %d', num_pipeline_events)

    for idx in range(num_pipeline_events):
        pipeline_event = pipeline_run_data.get('gbDisaggEvents')[idx]
        logger.info('Pipeline event | %d : %s', idx + 1, str(pipeline_event).replace('\n', ' '))

    # Fetch all static data for the user

    t_before_meta = datetime.now()
    home_meta_data, app_profile, hsm_appliances, status_code = fetch_meta_object(fetch_params, logger_pass)
    t_after_meta = datetime.now()

    if home_meta_data is None or app_profile is None or hsm_appliances is None:

        logger.info('Meta Data fetch failed | HTTP status %s', str(status_code))
        return None, False

    logger.info('Meta data fetch took | %.3f s', get_time_diff(t_before_meta, t_after_meta))

    # If cache mode is true, cache data

    if fetch_params.get('cache_mode'):

        t_before_cache = datetime.now()
        cache_status = cache_input_data(input_data, fetch_params, pipeline_run_data, weather_analytics_data, home_meta_data,
                                        app_profile, hsm_appliances)
        t_after_cache = datetime.now()

        logger.info('Caching Status : | %s', str(cache_status))
        logger.info('Caching data took | %.3f s', get_time_diff(t_before_cache, t_after_cache))

    # Combine input data and weather data

    input_data = combine_raw_data_and_weather_data(input_data, weather_analytics_data, logger_pass)

    # Package inputs together to avoid excessive parameters

    pipeline_object_params = {
        'input_data': input_data,
        'disagg_run_data': pipeline_run_data,
        'home_meta_data': home_meta_data,
        'app_profile': app_profile,
        'hsm_appliances': hsm_appliances,
        'logging_dict': logging_dict,
        'weather_analytics_data': weather_analytics_data,
    }

    # Initialize pipeline input objects

    try:
        pipeline_input_objects = init_pipeline_input_objects(fetch_params, pipeline_object_params, config_params,
                                                             logger_pass)
    except IndexError:
        logger.warning('Pipeline will not be run for the following reasons | Input data fetched is empty')
        pipeline_input_objects = None

    return pipeline_input_objects, True
