"""
Author - Mayank Sharan
Date - 19th Sep 2018
run gb local runs the pipeline on a local machine for one/multiple users
"""

# Import python packages

import time
import logging
import warnings
import traceback
import numpy as np
import pandas as pd
import concurrent.futures

from datetime import datetime
from multiprocessing import Lock
from multiprocessing import Value
from multiprocessing import Queue
from logging.handlers import WatchedFileHandler

# Import packages from within the projects

from python3.master_pipeline.master_director import master_director
from python3.initialisation.load_files.fetch_files import fetch_files
from python3.initialisation.load_files.fetch_files import fetch_hybrid_v2_model_files
from python3.initialisation.load_files.load_files_wrapper import load_files_wrapper

# To suppress warnings being printed in the console
warnings.simplefilter('ignore')

# Initializing the global variables needed to allow multi-processing
logger_master = None
message_queue = None
queue_lock = Lock()
read_5_min = Value('d', 0)
deleted_5_min = Value('d', 0)


def populate_message_queue(run_params_dict):

    """
    Parameters:
        run_params_dict         (dict)      : Contains params to handle how multiple users will run
    """

    global message_queue

    # Initialize the queue

    message_queue = Queue()

    # Load the user csv

    csv_path = run_params_dict.get('user_list_path')
    user_list_df = pd.read_csv(csv_path, header=None)
    user_list_arr = user_list_df.values

    # Initialize variables with column names for message array

    user_idx = 0
    t_start_idx = 1
    t_end_idx = 2
    api_env_idx = 3

    # Based on number of columns populate the default message array

    message_arr = np.full(shape=(user_list_arr.shape[0], 4), fill_value='', dtype=object)
    message_arr[:, user_idx] = user_list_arr[:, 0]

    if user_list_arr.shape[1] == 1:

        # uuid came here
        message_arr[:, t_start_idx] = run_params_dict.get('default_t0')
        message_arr[:, t_end_idx] = run_params_dict.get('default_t1')
        message_arr[:, api_env_idx] = run_params_dict.get('default_api_env')

    elif user_list_arr.shape[1] == 2:

        # uuid, api_env came here

        message_arr[:, t_start_idx] = run_params_dict.get('default_t0')
        message_arr[:, t_end_idx] = run_params_dict.get('default_t1')
        message_arr[:, api_env_idx] = user_list_arr[:, 1]

    elif user_list_arr.shape[1] == 3:

        # uuid, t_start, t_end came here

        message_arr[:, t_start_idx] = user_list_arr[:, 1].astype(int)
        message_arr[:, t_end_idx] = user_list_arr[:, 2].astype(int)
        message_arr[:, api_env_idx] = run_params_dict.get('default_api_env')

    elif user_list_arr.shape[1] == 4:

        # uuid, t_start, t_end, api_env came here

        message_arr[:, t_start_idx] = user_list_arr[:, 1].astype(int)
        message_arr[:, t_end_idx] = user_list_arr[:, 2].astype(int)
        message_arr[:, api_env_idx] = user_list_arr[:, 3]

    # Post the messages to the queue

    num_messages = message_arr.shape[0]
    print('Starting posting of {} messages to the queue'.format(num_messages))

    for msg_idx in range(num_messages):
        message_queue.put(message_arr[msg_idx, :])

        if (msg_idx + 1) % 100 == 0:
            print(datetime.now(),  '- {} messages posted'.format(msg_idx + 1))

    print('Message posting completed')


def run_user(uuid, t_start, t_end, api_env, loaded_files_dict, user_params_dict):

    """
    Run the pipeline locally for a user

    Parameters:
        uuid                    (str)               : The user for which the pipeline has to be run
        t_start                 (int)               : The start timestamp for which the data needs to be pulled
        t_end                   (int)               : The end timestamp for which the data needs to be pulled
        api_env                 (str)               : The api environment the pipeline will be operating in
        loaded_files_dict       (dict)              : Dictionary containing all loaded static files
        user_params_dict        (dict)              : Dictionary containing all pipeline run params

    Returns:
        delete_message          (bool)              : Boolean indicating if the processing was as expected
    """

    global logger_master

    logging_dict = {
        'uuid': uuid,
        't_start': -1,
        't_end': -1,
        'pilot_id': -1,
        'disagg_mode': 'N.A.',
    }

    logger_pass = {
        'logger': logger_master,
        'logging_dict': logging_dict
    }

    # By default we do not write any results in a local run
    # DO NOT CHANGE THIS. Disables write of results

    user_params_dict['write_results'] = False

    # Initialize cache mode from the parameters. Default value for cache mode is False

    cache_mode = user_params_dict.get('cache_mode')

    if cache_mode is None:
        cache_mode = False

    # Initialize run mode from the parameters. Default value for run mode is prod

    run_mode = user_params_dict.get('run_mode')

    if run_mode is None:
        run_mode = 'prod'

    pipeline_mode = user_params_dict.get('disagg_mode')

    # Fetch data using the api into the disagg input object

    fetch_params = {
        'api_env': api_env,
        'cache_mode': cache_mode,
        'disagg_mode': pipeline_mode,
        'override_mode': "",
        'run_mode': run_mode,
        't_start': t_start,
        't_end': t_end,
        'uuid': uuid,
        'priority': False,
        'cache_update': False,
        'loaded_files': loaded_files_dict,
        'trigger_id': '',
        'trigger_name': '',
        'retry_count': 0,
        'zipcode': '',
        'country': '',
        'build_info': user_params_dict.get('build_info')
    }

    delete_message = master_director(fetch_params, user_params_dict, logger_pass)

    return delete_message


def run_multiple_users_process_0(run_params):

    """
    Parameters:
        run_params          (tuple)              : Contains all parameters needed to initialise and run the pipeline
    """

    # Declare usage of global variable

    global logger_master
    global message_queue
    global queue_lock
    global read_5_min
    global deleted_5_min

    # Extract parameters from the tuple

    user_params_dict = run_params[0]

    # By default we do not write any results in a local run
    # DO NOT CHANGE THIS. Disables write of results

    user_params_dict['write_results'] = False

    # Initialize the logger for each process

    logging_dict = {
        'uuid': 'N.A',
        't_start': -1,
        't_end': -1,
        'pilot_id': -1,
        'disagg_mode': 'N.A.',
    }

    logger_master_adapter = logging.LoggerAdapter(logger_master, logging_dict)

    # Load all static files and models

    logger_pass = {
        'logger': logger_master,
        'logging_dict': logging_dict,
    }

    disagg_version = user_params_dict.get('disagg_version', '1.0.1477')
    job_tag = user_params_dict.get('job_tag', 'custom')
    build_info = disagg_version.split('.')[-1] + '_' + job_tag
    user_params_dict['build_info'] = build_info

    api_env = run_params[3].get('defaul_api_env', 'ds')

    fetch_files(api_env, disagg_version, job_tag, logger_master_adapter)

    fetch_hybrid_v2_model_files(api_env, disagg_version, job_tag, logger_master_adapter)

    loaded_files_dict = load_files_wrapper(disagg_version, job_tag, logger_pass)

    # This process runs disagg after reading the messages

    while not message_queue.empty():

        # read the message and call the disagg processing function
        message = None

        # noinspection PyBroadException
        try:
            message = message_queue.get(False, 2)
            queue_lock.acquire()
            read_5_min.value = read_5_min.value + 1
            queue_lock.release()
        except Exception:
            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_master_adapter.warning('Could not get messages from queue | %s', error_str)

        if message is not None:
            logger_master_adapter.info('Received message | %s', str(message).replace('\n', ' '))
            delete_message = run_user(message[0], message[1], message[2], message[3], loaded_files_dict,
                                      user_params_dict)
        else:
            continue

        # If processing is successful log deletion else re-post the message

        if delete_message:
            queue_lock.acquire()
            deleted_5_min.value = deleted_5_min.value + 1
            queue_lock.release()
            logger_master_adapter.info('Deleting message | %s', str(message).replace('\n', ' '))
        else:
            message_queue.put(message)


def run_multiple_users_process_1(run_params):

    """
    Parameters:
        run_params          (tuple)              : Contains all parameters needed to initialise and run the pipeline
    """

    # Declare usage of global variable

    global logger_master
    global message_queue
    global queue_lock
    global read_5_min
    global deleted_5_min

    throughput_file = run_params[1]

    # This process logs the statistics of the run

    while not message_queue.empty():

        # Open file to log the run statistics of the run

        f = open(throughput_file, 'a')

        # Prepare the statistics

        queue_lock.acquire()
        log_statement = \
            str(datetime.now()) + ' | ' + str(int(read_5_min.value)) + ' | ' + str(int(deleted_5_min.value))

        # Write the details in file and print them

        f.write(log_statement + '\n')
        print(log_statement)

        # Reset variables

        read_5_min.value = 0
        deleted_5_min.value = 0
        queue_lock.release()

        f.close()

        # To ensure that logging happens every 5 minutes

        sleep_count = 0

        while sleep_count < 12:

            # Check every 5 seconds to see if the queue is empty

            time.sleep(5)
            sleep_count += 1

            if message_queue.empty():
                break

    # Wait 20 seconds to let any pending messages complete processing

    time.sleep(5)

    # Write final stats before exiting

    f = open(throughput_file, 'a')

    # Prepare the statistics
    log_statement = str(datetime.now()) + ' | ' + str(int(read_5_min.value)) + ' | ' + str(int(deleted_5_min.value))

    # Write the details in file and print them

    f.write(log_statement + '\n')
    print(log_statement)

    # Reset variables

    read_5_min.value = 0
    deleted_5_min.value = 0

    f.close()


def run_multiple_users_process(run_params):

    """
    Parameters:
        run_params          (tuple)              : Contains all parameters needed to initialise and run the pipeline
    """

    # Extract necessary variable from the tuple

    function_type = run_params[2]

    # Based on function type call the further function

    if function_type == 0:
        run_multiple_users_process_0(run_params)
    elif function_type == 1:
        run_multiple_users_process_1(run_params)


def run_multiple_users(run_params_dict, user_params_dict):

    """
    Parameters:
        run_params_dict         (dict)      : Contains params to handle how multiple users will run
        user_params_dict        (dict)      : Contains params to handle the specific user run
    """

    global logger_master

    # Initialize root logger

    logger_master = logging.getLogger('run_multiple_users')
    logger_master.setLevel(run_params_dict.get('logging_level'))

    # Prepare and set handler and formatter to the logger
    # Watched file Handler ensures that we always open log file in append mode

    logging_path = '../' + run_params_dict.get('log_file_name')

    handler = WatchedFileHandler(logging_path)
    formatter = logging.Formatter('%(asctime)s | %(process)d | %(name)s | %(levelname)s | %(uuid)s | %(t_start)d | '
                                  '%(t_end)d | %(pilot_id)d | %(disagg_mode)s | %(message)s')

    handler.setFormatter(formatter)
    logger_master.addHandler(handler)

    # Initialize and populate the queue with disagg messages

    populate_message_queue(run_params_dict)

    # Initiate a pool of processes each of which reads messages from the queue and runs the disagg pipeline

    num_processes = int(run_params_dict.get('num_processes')) + 1

    process_idx = np.arange(num_processes)
    process_params = (user_params_dict, '', 0, run_params_dict)

    process_params_list = list([])
    process_params_list.append((user_params_dict, run_params_dict.get('throughput_file'), 1, run_params_dict))

    for idx in range(num_processes - 1):
        process_params_list.append(process_params)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        zip(process_idx, executor.map(run_multiple_users_process, process_params_list))
