"""
Author - Shubham Maddhashiya
Date - 19/09/2023
Class for Backend data fetch with weather data caching (disk-based caching)
"""

# Import standard python packages
import sys
import time

# Import external python packages
from diskcache import Cache

# Import local python packages 
from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants
from python3.initialisation.fetch_data.fetch_request_object import attempt_fetch_request_object


class BackendDataProvider:
    """Attributes and methods for Backend data fetch with weather data caching"""

    def __init__(self, fetch_params, url, url_sec, max_tries_primary, max_tries_secondary, logger):
        self.fetch_params = fetch_params
        self.url = url
        self.url_sec = url_sec
        self.max_tries_primary = max_tries_primary
        self.max_tries_secondary = max_tries_secondary
        self.logger = logger

        self.cache_store_log_line = "Caching store state for key {} : start={}, end={}, actual_start={} actual_end={}, length={}"

        self.cache = Cache(PathConstants.WEATHER_DATA_CACHE_DIR, size_limit=Cgbdisagg.ON_DISK_WEATHER_DATA_CACHE_SIZE_LIMIT)
        self.country = fetch_params['country']
        self.zipcode = fetch_params['zipcode']

    def fetch_data(self):

        # Check if Cache storage is enabled or not

        self.logger.info("Backend data fetch request for country = {} and zipcode = {}".format(self.country, self.zipcode))

        if (not Cgbdisagg.ENABLE_WEATHER_DATA_CACHE) or (not self.country) or (not self.zipcode):
            return self.backend_data_fetch(self.url, self.url_sec)

        # Fill and get weather data from caching layer
        try:
            # Load the cache
            start_time = time.time()
            cache = self.cache.get(self.get_weather_data_cache_key(), default={})
            self.logger.info("Time taken to get the weather data from the cache | {} seconds".format(time.time() - start_time))

            api_param_start = 0
            api_param_end = 0
            weather_data_from_cache = []
            weather_data_from_cache_actual_start = -1
            weather_data_from_cache_actual_end = -1

            # Get the start and end timestamps of the already present weather cache data along with weather data

            if len(cache) != 0:
                api_param_start = cache.get('start')
                api_param_end = cache.get('end')
                weather_data_from_cache = cache.get('weather_data')
                if len(weather_data_from_cache) != 0:
                    weather_data_from_cache_actual_start = weather_data_from_cache[0]['timestamp']
                    weather_data_from_cache_actual_end = weather_data_from_cache[-1]['timestamp']

            self.logger.info(self.cache_store_log_line.format(self.get_weather_data_cache_key(), api_param_start, api_param_end,
                                                              weather_data_from_cache_actual_start, weather_data_from_cache_actual_end, len(weather_data_from_cache)));
            # Get the updated URL with the new timestamps

            url, url_sec = self.get_url_with_cache_api_params(api_param_start, api_param_end)

            self.logger.info('Primary backend data fetch url with weather data cache params | %s', url)
            self.logger.info('Secondary backend data fetch url with weather data cache params | %s', url_sec)

            # Fetch the data with the updated URL

            input_data, pipeline_run_data, weather_analytics_data, status_code, req_object = \
                self.backend_data_fetch(url, url_sec)

            if input_data is None or pipeline_run_data is None:
                return input_data, pipeline_run_data, weather_analytics_data, status_code, req_object

            # Get fetch and required weather data duration from the payload

            returned_data_duration = req_object['payload']['weatherDataDuration']
            required_duration = req_object['payload']['requiredWeatherDataDuration']

            self.logger.info("Returned weather data duration is | {}".format(returned_data_duration))
            self.logger.info("Required weather data duration is | {}".format(required_duration))

            # Populate the weather data for this zipcode in the cache

            weather_analytics_data = self.fill_and_get_weather_data_cache(weather_data_from_cache, weather_analytics_data, returned_data_duration,
                                                                          required_duration)
            if len(weather_analytics_data) == 0:
                self.logger.info("Got empty filtered weather data for key {}, calling direct backend data fetch".format(self.get_weather_data_cache_key()))
                return self.backend_data_fetch(self.url, self.url_sec)

        except Exception as ex:
            self.logger.warning("Got exception while trying to fill and get the weather data from cache: {}".format(ex))
            self.logger.exception(ex)

            # Get weather data from backend directly rather than from the cache
            input_data, pipeline_run_data, weather_analytics_data, status_code, req_object = \
                self.backend_data_fetch(self.url, self.url_sec)

        return input_data, pipeline_run_data, weather_analytics_data, status_code, req_object

    def backend_data_fetch(self, url, url_sec):
        """
        Function to fetch the raw data
        Parameters:
            url                         (string)                : Primary URL
            url_sec                     (string)                : Secondary URL
        Returns:
            input_data                  (np.ndarray)            : Input consumption data
            pipeline_run_data           (dict)                  : Pipeline run data
            weather_analytics_data      (np.ndarray)            : Derived weather data
            status_code                 (int)                   : API response status code
            req_object                  (dict)                  : Response object
        """

        # Attempt data fetch from the primary data server

        input_data, pipeline_run_data, weather_analytics_data, status_code, req_object = \
            attempt_fetch_request_object(url, self.max_tries_primary, self.fetch_params, self.logger,
                                         server_type='primary')

        if input_data is None and not (status_code == 400):

            # Attempt data fetch from secondary data server not to be attempted if we get 400 from primary server

            input_data, pipeline_run_data, weather_analytics_data, status_code, req_object = \
                attempt_fetch_request_object(url_sec, self.max_tries_secondary, self.fetch_params, self.logger,
                                             server_type='secondary')
        return input_data, pipeline_run_data, weather_analytics_data, status_code, req_object

    def fill_and_get_weather_data_cache(self, weather_data_from_cache, weather_analytics_data, returned_data_duration, required_duration):

        """
        Function to Store / Override weather data to the existing weather data cache
        Parameters:
            weather_data_from_cache         (np.ndarray)        : Derived weather data from cache
            weather_analytics_data          (np.ndarray)        : Derived weather data from backend fetch
            returned_data_duration          (dict)              : Weather data fetch duration
            required_duration               (dict)              : Weather data required duration
        Returns:
            final_weather_analytics_data    (np.ndarray)        : Refined weather data
        """

        if len(weather_analytics_data) != 0:
            # Backend returns data for full duration, so directly override the cache value
            returned_data_start = returned_data_duration['first']
            returned_data_end = returned_data_duration['last']
            returned_data_actual_start = weather_analytics_data[0]['timestamp']
            returned_data_actual_end = weather_analytics_data[-1]['timestamp']
            cache_value = {
                'start' : returned_data_start,
                'end' : returned_data_end,
                'weather_data' : weather_analytics_data
            }
            self.logger.info("Got some weather data for country = {} and zipcode = {} from backend, updating cache".format(self.country, self.zipcode))
            self.logger.info(self.cache_store_log_line.format(self.get_weather_data_cache_key(), returned_data_start, returned_data_end,
                                                                                                            returned_data_actual_start, returned_data_actual_end, len(weather_analytics_data)));
            self.cache.set(self.get_weather_data_cache_key(), cache_value, expire=Cgbdisagg.WEATHER_DATA_CACHE_EXPIRE_TIME)
            full_weather_analytics_data = weather_analytics_data
        else:
            # Fetching directly from the cache

            self.logger.info("Full weather data is present in cache for country = {} and zipcode = {}, serving from caching layer".format(self.country, self.zipcode))
            full_weather_analytics_data = weather_data_from_cache

        self.cache.close()
        # Filter the weather data for the required duration

        final_weather_analytics_data = self.filter_weather_data(full_weather_analytics_data, required_duration['first'],
                                                                required_duration['last'])
        return final_weather_analytics_data

    def filter_weather_data(self, weather_data_from_cache, start_time, end_time):
        """
        Function to filter the weather data for the required duration
        Parameters:
            weather_data_from_cache         (np.ndarray)            : Derived weather data
            start_time                      (float)                 : Requested start time
            end_time                        (float)                 : Requested end time
        Returns:
            weather_analytics_data          (np.ndarray)            : Derived weather data
        """
        weather_analytics_data = []
        for x in weather_data_from_cache:
            if start_time <= int(x['timestamp']) <= end_time:
                weather_analytics_data.append(x)
        return weather_analytics_data

    def get_weather_data_cache_key(self):
        key = str(self.country) + " | " + str(self.zipcode)
        return key

    def get_url_with_cache_api_params(self, api_param_start, api_param_end):
        weather_args = '&weatherDataCacheStart=' + str(api_param_start) + '&weatherDataCacheEnd=' + str(api_param_end)
        url = self.url + weather_args
        url_sec = self.url_sec + weather_args
        return url, url_sec
