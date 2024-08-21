#!/usr/bin/env python3
"""
Author - Mayank Sharan
Date - 26/10/18
queue mode runs the pipeline on servers or local machine by reading sqs queue messages and using multiprocessing
"""

# Import python packages

import os
import time
import json
import boto3
import logging
import warnings
import traceback
import numpy as np
import concurrent.futures
from datetime import datetime
import xml.etree.ElementTree as elT
from logging.handlers import WatchedFileHandler

# Import functions from within the project

from python3.utils.time.get_time_diff import get_time_diff

from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.path_constants import PathConstants
from python3.config.mappings.get_env_properties import get_env_properties

from python3.initialisation.load_files.fetch_files import fetch_files
from python3.initialisation.load_files.fetch_files import fetch_hybrid_v2_model_files

from python3.initialisation.load_files.load_files_wrapper import load_files_wrapper

from python3.master_pipeline.master_director import master_director

# To suppress warnings being printed in the console
warnings.simplefilter('ignore')

# Initializing as a global variable to allow multiprocessing to work with boto3
sqs = None
logger_master = None


def get_queue_name(priority_flag, api_env, queue_suffix, queue_partition_suffix):

    """Utility function to get the name of the queue to initialize"""

    if priority_flag:
        base_name = Cgbdisagg.QUEUE_BASE_NAME_PRIORITY
    else:
        base_name = Cgbdisagg.QUEUE_BASE_NAME

    if api_env == "":
        env_name = ""
    else:
        env_name = "-" + api_env

    if queue_suffix == "":
        suffix_name = ""
    else:
        suffix_name = "-" + queue_suffix

    if queue_partition_suffix == "":
        queue_partition_suffix_name = ""
    else:
        queue_partition_suffix_name = "-" + queue_partition_suffix

    queue_name = base_name + env_name + suffix_name + queue_partition_suffix_name

    return queue_name


def extract_run_config(message_dict, params_dict):

    """Utility function to extract parameters from sqs message"""

    # Try to extract dump_csv

    dump_csv = message_dict.attrib.get('dump_csv')

    if dump_csv is not None:
        params_dict['dump_csv'] = dump_csv

    # Try to extract dump_debug

    dump_debug = message_dict.attrib.get('dump_debug')

    if dump_debug is not None:
        params_dict['dump_debug'] = dump_debug

    # Try to extract generate_plots

    generate_plots = message_dict.attrib.get('generate_plots')

    if generate_plots is not None:
        params_dict['generate_plots'] = generate_plots

    return params_dict


def process_message(message_dict, params_dict, loaded_files_dict, api_env):

    """
    Process a SQS message

    Parameters:
        message_dict        (elT.Element)       : Dictionary containing contents of sqs message
        params_dict         (dict)              : Dictionary containing parameters to configure the run
        loaded_files_dict   (dict)              : Dictionary that contains all loaded files
        api_env             (str)               : Decides the APIs to perform all activities

    Returns:
        delete_message      (bool)              : If true the message is deleted from the queue
    """

    global logger_master

    # Initialize variables from the sqs message

    uuid = (message_dict.attrib['userId']).strip()
    store_tb_in_cassandra = (message_dict.attrib.get('storeTimebandInCassandra'))

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

    logger_master_adapter = logging.LoggerAdapter(logger_master, logging_dict)

    try:
        t_start = int(message_dict.attrib['start'])
        t_end = int(message_dict.attrib['end'])
        override_disagg_mode = message_dict.attrib.get('disaggMode')
        trigger_id = message_dict.attrib.get('triggerId', '')
        trigger_name = message_dict.attrib.get('triggerName', '')
        retry_count = message_dict.attrib.get('retryCount', 0)
        zipcode = message_dict.attrib.get('zipCode', '')
        country = message_dict.attrib.get('country', '')

    except ValueError:
        logger_master_adapter.error('Erroneous message Deleting message | uuid = %s, t0 = %s, t1 = %s ',
                                    uuid, str(message_dict.attrib['start']), str(message_dict.attrib['end']))
        return True

    logger_master_adapter.info("Trigger ID value | %s ", trigger_id)
    logger_master_adapter.info("Trigger name value | %s ", trigger_name)
    logger_master_adapter.info("retry count value | %s ", retry_count)
    logger_master_adapter.info("zipcode value | %s", zipcode)
    logger_master_adapter.info("country value | %s", country)

    # Logging build info at user-level

    build_info = message_dict.attrib.get('build_info', "Unknown")
    logger_master_adapter.info("Disagg build info is | %s ", build_info)

    # Extract parameters for run config from the sqs message

    params_dict = extract_run_config(message_dict, params_dict)

    if override_disagg_mode is None:
        override_disagg_mode = ""

    logger_master_adapter.info('Override pipeline mode received is | %s', override_disagg_mode)

    # Fetch data using the api into the disagg input object

    fetch_params = {
        'api_env': api_env,
        'override_mode': override_disagg_mode,
        't_start': t_start,
        't_end': t_end,
        'uuid': uuid,
        'run_mode': 'prod',
        'cache_mode': False,
        'cache_update': False,
        'disagg_mode': None,
        'priority': params_dict.get('priority', False),
        'loaded_files': loaded_files_dict,
        'trigger_id': trigger_id,
        'trigger_name': trigger_name,
        'retry_count': retry_count,
        'zipcode': zipcode,
        'country': country,
        'build_info': build_info,
        'store_tb_in_cassandra': store_tb_in_cassandra
    }

    # Set write results to true

    params_dict['write_results'] = True

    # Run the pipeline for the user

    t_before_message_process = datetime.now()
    delete_message = master_director(fetch_params, params_dict, logger_pass)
    t_after_message_process = datetime.now()

    logger_master_adapter.info('Message processing time combined is | %.3f s', get_time_diff(t_before_message_process,
                                                                                             t_after_message_process))

    return delete_message


def queue_mode_process(run_params):

    """
    Parameters:
        run_params          (tuple)              : Contains all parameters needed to initialise and run the pipeline
    """

    # Declare usage of global variable

    global logger_master

    # Initialize the logger for each process

    logging_dict = {
        'uuid': 'N.A',
        't_start': -1,
        't_end': -1,
        'pilot_id': -1,
        'disagg_mode': 'N.A.',
    }

    logger_master_adapter = logging.LoggerAdapter(logger_master, logging_dict)

    # Extract parameters from the dictionary

    disagg_version = run_params[4]
    queue_suffix = run_params[0]
    params_dict = run_params[3]
    priority = run_params[2]
    api_env = run_params[1]
    job_tag = run_params[5]
    queue_partition_suffix = run_params[6]

    params_dict['priority'] = priority

    # Load all static files and models

    logger_pass = {
        'logger': logger_master,
        'logging_dict': logging_dict,
    }

    loaded_files_dict = load_files_wrapper(disagg_version, job_tag, logger_pass)

    # Initialize the queue we are going to read from

    queue = None
    queue_name = ""

    # noinspection PyBroadException
    try:
        queue_name = get_queue_name(priority, str.lower(api_env), str.lower(queue_suffix), queue_partition_suffix)
        logger_master_adapter.info('Trying to initialize | ' + queue_name)

        queue = sqs.get_queue_by_name(QueueName=queue_name)
    except Exception:
        error_str = (traceback.format_exc()).replace('\n', ' ')
        logger_master_adapter.error('Queue Initialization Failed. Exiting process | queue - %s %s',
                                    queue_name, error_str)
        exit(1)

    while True:

        message_list = None

        # noinspection PyBroadException
        try:
            message_list = queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=Cgbdisagg.QUEUE_RETRY_TIME)
        except Exception:
            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger_master_adapter.warning('Could not connect to queue | %s', error_str)

        if len(message_list) == 0:
            logger_master_adapter.info('Queue is empty |')
            time.sleep(Cgbdisagg.QUEUE_RETRY_TIME)
            continue

        message = message_list[0]
        message_string = (json.dumps(message.body, indent=2)).replace('\n', ' ')

        # noinspection PyBroadException
        try:
            logger_master_adapter.info('Received event | %s' + message_string)
            message_load = elT.fromstring(message.body)

        except Exception:

            error_str = (traceback.format_exc()).replace('\n', ' ')

            logger_master_adapter.error('Failed message | %s', message_string)
            logger_master_adapter.error('Unable to extract message | %s', error_str)

            logger_master_adapter.info('Deleting message | %s', message_string)
            message.delete()

            continue

        # Passing build info for logging at user level
        message_load.attrib['build_info'] = str(disagg_version) + '_' + str(job_tag)

        delete_message = process_message(message_load, params_dict, loaded_files_dict, api_env)

        if delete_message:
            logger_master_adapter.info('Deleting message | %s', message_string)
            message.delete()


def queue_mode(queue_suffix, queue_partition_suffix, api_env, priority_flag, process_multiplier, **kwargs):

    """
    Parameters:
        queue_suffix                (str)               : The queue to read messages from
        queue_partition_suffix      (str)               : The Partition suffix or Group ID of the queue
        api_env                     (str)               : The API env to run the code in
        priority_flag               (bool)              : The env to decide whether to use normal queue or priority queue
        process_multiplier          (float)             : The ratio of number of processes to be spawned to number of cores
    """

    # To make sure boto3 works with multiprocessing

    global sqs
    global logger_master

    # Initialize some important variables we need here
    # Number of processes we will spawn as a multiplier to number of cores

    num_cpu_cores = os.cpu_count()
    num_processes = int(process_multiplier * num_cpu_cores)

    # Extract aws region as per the api env

    env_prop = get_env_properties(api_env)
    aws_region = env_prop.get('aws_region')

    # Initialize boto3 sqs resource

    sqs = boto3.resource('sqs', region_name=aws_region)

    # Initialize the logger to be used

    # Setup the logger to be used while processing this message

    logger_master = logging.getLogger('process_message')

    # Use env variable to decide the level of the logger

    logger_level = os.getenv('LOGGER_LEVEL')

    # Use this dictionary to define logger level

    logging_level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    if logger_level is None:
        logger_master.setLevel(logging.INFO)

    logger_level = logging_level_map.get(logger_level)

    if logger_level is None:
        logger_master.setLevel(logging.INFO)
    else:
        logger_master.setLevel(logger_level)

    # /var/log/bidgely - location where we are supposed to store logs on the server locally - "../gb_disagg.log"
    if priority_flag:
        logging_path = PathConstants.LOG_DIR_PRIORITY + "gb_disagg.log"
    else:
        logging_path = PathConstants.LOG_DIR + "gb_disagg.log"

    # Watched file Handler ensures that we always open log file in append mode
    handler = WatchedFileHandler(logging_path)

    formatter = logging.Formatter('%(asctime)s | %(process)d | %(name)s | %(levelname)s | %(uuid)s | %(t_start)d | '
                                  '%(t_end)d | %(pilot_id)d | %(disagg_mode)s | %(message)s')

    handler.setFormatter(formatter)
    logger_master.addHandler(handler)

    # Logger adapter for this function

    logging_dict = {
        'uuid': 'N.A.',
        't_start': -1,
        't_end': -1,
        'pilot_id': -1,
        'disagg_mode': 'N.A.',
    }

    logger_queue_mode = logging.LoggerAdapter(logger_master, logging_dict)

    # Know the installed version of the PyAmi pipeline

    disagg_version_command = 'dpkg -s pyamidisagg'

    package_status_info_read = os.popen(disagg_version_command).read()
    logger_queue_mode.info('Output of package status is : | %s', str(package_status_info_read).replace('\n', ' '))

    package_status_info = package_status_info_read.split('\n')

    version_idx = 7
    job_idx = 10

    if len(package_status_info) < 8:
        disagg_version = "No package installed on machine"
        job_tag = "No package installed on machine"
    else:
        disagg_version = package_status_info[version_idx].split(':')[-1]
        disagg_version = disagg_version.strip()

        job_tag = package_status_info[job_idx].split(':')[-1]
        job_tag = job_tag.strip()

    logger_queue_mode.info('Package version on machine : | %s ', str(disagg_version))
    logger_queue_mode.info('Build job is : | %s ', str(job_tag))

    # Copy the files needed for disagg to a local location

    fetch_files(api_env, disagg_version, job_tag, logger_queue_mode)

    fetch_hybrid_v2_model_files(api_env, disagg_version, job_tag, logger_queue_mode, queue_suffix)

    # Initiate a pool of processes each of which reads messages from the queue and runs the disagg pipeline

    process_idx = np.arange(num_processes)
    process_params = (queue_suffix, api_env, priority_flag, kwargs, disagg_version, job_tag, queue_partition_suffix)
    process_params_list = []

    for idx in range(num_processes):
        process_params_list.append(process_params)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        zip(process_idx, executor.map(queue_mode_process, process_params_list))


if __name__ == '__main__':

    # We will get the variables API environment and queue suffix from the environment

    # Queue suffix

    queue_suffix_var = os.getenv('QUEUE_SUFFIX')

    if queue_suffix_var is None:
        print('QUEUE_SUFFIX environment variable missing')
        queue_suffix_var = ""
    else:
        print('QUEUE_SUFFIX is', queue_suffix_var)

    # Queue partition suffix

    queue_partition_suffix_var = os.getenv('GB_TEMP_DATA_READY_EVENT_PYAMI_GROUP_ID')

    if queue_partition_suffix_var is None:
        print('GB_TEMP_DATA_READY_EVENT_PYAMI_GROUP_ID environment variable missing')
        queue_partition_suffix_var = ""
    else:
        print('GB_TEMP_DATA_READY_EVENT_PYAMI_GROUP_ID is', queue_partition_suffix_var)

    # API environment

    api_env_var = os.getenv('BIDGELY_ENV')

    if api_env_var is None:
        print('BIDGELY_ENV environment variable missing')
        api_env_var = ""
    else:
        print('BIDGELY_ENV is', api_env_var)

    # Priority flag

    priority_env_var = os.getenv('PRIORITY_FLAG')

    if priority_env_var is not None:
        if str.lower(priority_env_var) == 'true':
            priority_env_var = True
        else:
            priority_env_var = False

    if priority_env_var is None:
        print('PRIORITY_FLAG environment variable missing initialising to False')
        priority_env_var = False
    else:
        print('PRIORITY_FLAG is', priority_env_var)

    # Process Multiplier

    process_multiplier_var = os.getenv('PROCESS_MULTIPLIER')

    if process_multiplier_var is None:
        if priority_env_var:
            print('PROCESS_MULTIPLIER environment variable missing',
                  'initializing to {}'.format(Cgbdisagg.PROCESSES_MULTIPLIER_PRIORITY_DEFAULT))
            process_multiplier_var = Cgbdisagg.PROCESSES_MULTIPLIER_PRIORITY_DEFAULT
        else:
            print('PROCESS_MULTIPLIER environment variable missing',
                  'initializing to {}'.format(Cgbdisagg.PROCESSES_MULTIPLIER_DEFAULT))
            process_multiplier_var = Cgbdisagg.PROCESSES_MULTIPLIER_DEFAULT
    else:
        process_multiplier_var = float(process_multiplier_var)
        print('PROCESS_MULTIPLIER is', process_multiplier_var)

    # Initialize any other parameters that are to configured for the run, then add as input to queue_mode call

    queue_mode(queue_suffix_var, queue_partition_suffix_var, api_env_var, priority_env_var, process_multiplier_var)
