"""
Author - Mayank Sharan
Date - 12/08/19
Fetch files copies files from an s3 location to a local folder
"""

# Import python packages

import os
import boto3
import traceback

from datetime import datetime

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg

from python3.utils.time.get_time_diff import get_time_diff
from python3.config.path_constants import PathConstants


def fetch_vip_file(s3_paginator, s3_resource, logger):

    """
    Fetch VIP disagg file

    Parameters:
        s3_paginator        (boto3.client.paginator)    : The boto3 client for s3
        s3_resource         (boto3.resource)            : The boto3 resource for s3
        logger              (logger)                    : Logger to log information
    """

    vip_bucket = PathConstants.VIP_FILE_S3_BUCKET
    vip_prefix = PathConstants.VIP_FILE_S3_PREFIX

    local_vip_path = PathConstants.VIP_DIR

    # Create folder if it does not exist

    if not os.path.exists(os.path.dirname(local_vip_path)):
        os.makedirs(os.path.dirname(local_vip_path))

    for result in list(s3_paginator.paginate(Bucket=vip_bucket, Delimiter='/', Prefix=vip_prefix)):
        for file in result.get('Contents', []):

            file_key = file.get('Key')

            # Prepare local path

            file_name = file_key.split('/')[-1]

            if file_name == PathConstants.VIP_FILE:

                local_dest_path = local_vip_path + file_name

                # Copy the file from s3 location

                t_file_load_start = datetime.now()
                s3_resource.meta.client.download_file(vip_bucket, file_key, local_dest_path)
                t_file_load_end = datetime.now()

                logger.info('Time taken to fetch %s | %.3f s', file_key, get_time_diff(t_file_load_start,
                                                                                       t_file_load_end))


def fetch_files_rec(s3_paginator, s3_resource, s3_bucket, s3_prefix, local_path, logger):

    """
    Parameters:
        s3_paginator        (boto3.client.paginator)    : The boto3 client for s3
        s3_resource         (boto3.resource)            : The boto3 resource for s3
        s3_bucket           (string)                    : The s3 bucket where the files are located
        s3_prefix           (string)                    : The s3 path where the files are located
        local_path          (string)                    : The local folder where we place the downloaded files
        logger              (logger)                    : Logger to log information
    """

    # List everything in the location we need

    for result in list(s3_paginator.paginate(Bucket=s3_bucket, Delimiter='/', Prefix=s3_prefix)):

        # This will pull sub-directories if available

        if result.get('CommonPrefixes') is not None:
            for sub_dir in result.get('CommonPrefixes'):
                last_folder = sub_dir.get('Prefix').split('/')[-2]
                fetch_files_rec(s3_paginator, s3_resource, s3_bucket, sub_dir.get('Prefix'),
                                local_path + last_folder + '/', logger)

        # Pull files from this location

        for file in result.get('Contents', []):

            file_key = file.get('Key')

            # Prepare local path

            file_name = file_key.split('/')[-1]
            local_dest_path = local_path + file_name

            # Create folder if it does not exist

            if not os.path.exists(os.path.dirname(local_dest_path)):
                os.makedirs(os.path.dirname(local_dest_path))

            # Copy the file from s3 location

            if not os.path.exists(local_dest_path):

                t_file_load_start = datetime.now()
                s3_resource.meta.client.download_file(s3_bucket, file_key, local_dest_path)
                t_file_load_end = datetime.now()

                logger.info('Time taken to fetch %s | %.3f s', file_key,
                            get_time_diff(t_file_load_start, t_file_load_end))


def fetch_hybrid_v2_files_rec(s3_paginator, s3_resource, s3_bucket, s3_prefix, local_path, logger):

    """
    Parameters:
        s3_paginator        (boto3.client.paginator)    : The boto3 client for s3
        s3_resource         (boto3.resource)            : The boto3 resource for s3
        s3_bucket           (string)                    : The s3 bucket where the files are located
        s3_prefix           (string)                    : The s3 path where the files are located
        local_path          (string)                    : The local folder where we place the downloaded files
        logger              (logger)                    : Logger to log information
    """

    # List everything in the location we need

    for result in list(s3_paginator.paginate(Bucket=s3_bucket, Delimiter='/', Prefix=s3_prefix)):
        # Pull files from this location

        for file in result.get('Contents', []):

            file_key = file.get('Key')

            # Prepare local path

            file_name = file_key.split('/')[-1]
            local_dest_path = local_path + file_name

            # Create folder if it does not exist

            if not os.path.exists(os.path.dirname(local_dest_path)):
                os.makedirs(os.path.dirname(local_dest_path))

            # Copy the file from s3 location

            t_file_load_start = datetime.now()
            s3_resource.meta.client.download_file(s3_bucket, file_key, local_dest_path)
            t_file_load_end = datetime.now()

            logger.info('Time taken to fetch %s | %.3f s', file_key,
                        get_time_diff(t_file_load_start, t_file_load_end))


def fetch_files(api_env, disagg_version, job_tag, logger):

    """
    Parameters:
        api_env             (string)            : The environment for which the files are to be pulled
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger              (logger)            : Logger to log information
    """

    version_number = disagg_version.split('.')[-1]

    # Create the local folder needed as per the api_env

    local_path = PathConstants.FILES_LOCAL_ROOT_DIR

    s3_bucket = PathConstants.FILES_S3_BUCKET_DICT.get(api_env)
    s3_prefix = version_number + '_' + job_tag

    logger.info('Static files will be loaded locally at | %s ', local_path)
    logger.info('Static files will be loaded from s3 location | %s ', s3_bucket + '/' + s3_prefix)

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # Initialize the boto3 client and resource to pull the files

    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    # Initialize the paginator that will allow us to list all the objects amd use it to download the files in a given
    # s3 location

    s3_paginator = s3_client.get_paginator('list_objects')

    # Call the recursive function to copy the files

    t_files_load_start = datetime.now()
    fetch_files_rec(s3_paginator, s3_resource, s3_bucket, s3_prefix, local_path, logger)
    t_files_load_end = datetime.now()

    logger.info('Total time taken to fetch files | %.3f s ', get_time_diff(t_files_load_start, t_files_load_end))

    # Fetch the file for VIP user results if it exists. This will only be done if API environment is ds. This
    # is only to be used for POCs

    if api_env == 'ds':

        # noinspection PyBroadException
        try:
            fetch_vip_file(s3_paginator, s3_resource, logger)
        except Exception:
            error_str = (traceback.format_exc()).replace('\n', ' ')
            logger.warning('Error in fetching VIP file | %s', error_str)


def fetch_hybrid_v2_model_files(api_env, disagg_version, job_tag, logger, queue_suffix=''):

    """
    Parameters:
        api_env             (string)            : The environment for which the files are to be pulled
        disagg_version      (string)            : String containing the version information of the build
        job_tag             (string)            : String containing information regarding the build process used
        logger              (logger)            : Logger to log information
    """

    # Create the local folder needed as per the api_env

    version_number = disagg_version.split('.')[-1]

    s3_prefix = PathConstants.PATH_SUFFIX_FOR_HYBRID_V2_CONFIG

    if queue_suffix != '':
        s3_prefix = PathConstants.PATH_SUFFIX_FOR_HYBRID_V2_CONFIG + str(queue_suffix) + '/'

    local_path = PathConstants.HYBRID_V2_CONFIG_FILES_LOCAL_ROOT_DIR + version_number + '_' + job_tag + PathConstants.HYBRID_V2_CONFIG_FILES_LOCAL_PATH

    s3_bucket = PathConstants.HYBRID_V2_CONFIG_FILES_S3_BUCKET_DICT.get(api_env)

    logger.info('Hybrid v2 model files will be loaded locally at | %s ', local_path)
    logger.info('Hybrid v2 model files will be loaded from s3 location | %s ', s3_bucket + '/' + s3_prefix)

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # Initialize the boto3 client and resource to pull the files

    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    # Initialize the paginator that will allow us to list all the objects amd use it to download the files in a given
    # s3 location

    s3_paginator = s3_client.get_paginator('list_objects')

    # Call the recursive function to copy the files

    t_files_load_start = datetime.now()
    fetch_hybrid_v2_files_rec(s3_paginator, s3_resource, s3_bucket, s3_prefix, local_path, logger)
    t_files_load_end = datetime.now()

    logger.info('Total time taken to fetch files | %.3f s ', get_time_diff(t_files_load_start, t_files_load_end))
