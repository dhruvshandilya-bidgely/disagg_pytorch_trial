"""
Author - Nikhil Singh Chauhan
Date - 02/11/18
This module zip/unzip the output files for every user to save space
"""

# Import python packages

import os
import s3fs
import pickle
import zipfile


def file_unzip(logger, filename='data.zip'):
    """
    Unzipping the user files before processing

    Parameters:
        filename    (str)           : Name of the zip file to be unzipped
        logger      (logger)        : Logger object
    """

    # If the zip file already exists on the path, unzip the files else skip

    if os.path.exists(filename):
        with zipfile.ZipFile(filename, 'r') as myzip:
            myzip.extractall()
            os.remove(filename)

        myzip.close()

        logger.info('User files unzipped successfully | ')
    else:
        logger.info('No user files to unzip | ')

    return


def file_zip(logger, filename='data.zip'):
    """
    Zipping the user files after processing

    Parameters:
        filename    (str)           : Name of the zip file to be created
        logger      (logger)        : Logger object
    """

    # Get the list of files present in the user directory

    file_list = [file for file in os.listdir() if '.csv' in file]

    # Add each file to the zip

    if len(file_list) > 0:
        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as myzip:
            for f in file_list:
                myzip.write(f)
                os.remove(f)

        myzip.close()

        logger.info('User files zipped successfully | ')
    else:
        logger.info('No user files to zip | ')

    return


def stream_file_to_s3(s3_bucket, s3_key, stream_variable):
    """
        This function streams pickle files to s3

        Parameters:
            s3_bucket:             (str)               s3 bucket name
            s3_key:          (str)               s3 key where pickle needs to be stored.
            stream_variable:       (object)            Object to be pickled and pushed to s3
        Returns:
    """
    # Create s3 file system and stream pickle to s3
    file_system = s3fs.S3FileSystem(anon=False)

    # Pickling variable to s3
    with file_system.open('s3://{}/{}.pkl'.format(s3_bucket, s3_key), 'wb') as out_file:
        pickle.dump(stream_variable, out_file)
