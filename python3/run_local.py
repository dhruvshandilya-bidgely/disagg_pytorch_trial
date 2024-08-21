"""
Author - Mayank Sharan
Date - 19th Sep 2018
run local runs the pipeline on a local machine for one/multiple users
"""

# Import python packages
import logging

# Import packages from within the pipeline

from python3.run_single_user import run_single_user
from python3.run_multiple_users import run_multiple_users


if __name__ == '__main__':
    # Set this boolean to decide between the way we will run. By default set to false so that you can run 1 user
    multiple_users = False

    if not multiple_users:

        # To run a single user like we were, use the following command
        run_single_user(uuid='9689d6e1-1787-4946-9e8b-3797ff6928fa',
                        smb_type='all',
                        t_start=0,
                        t_end=1800000000,
                        api_env='ds',
                        cache_mode=True,
                        disagg_version='1.0.1477',
                        job_tag='release')

    else:

        # A list of parameters we offer for you to be able to run the pipeline locally for multiple users

        # Number of columns we allow in the user list file and associated assumption of column data
        # Please do not use a csv containing headers
        # 1 column : uuid
        # 2 columns : uuid, api_env
        # 3 columns : uuid, t0, t1
        # 4 columns : uuid, t0, t1, api_env
        # Please stick to these formats only, If there is any other format that might be needed please suggest.

        multi_run_params_dict = {
            'num_processes': 1,
            'user_list_path': 'disagg_trial.csv',
            'log_file_name': 'gb_disagg_multiple_users.log',
            'throughput_file': 'test_throughput.txt',
            'default_t0': 0,
            'default_t1': 1800000000,
            'default_api_env': 'ds',
            'logging_level': logging.DEBUG,
            'cache_update': False
        }

        # Parameter dictionary to configure the run. You can configure the following parameters
        # 'cache_mode': False,
        # 'disagg_mode': None,
        # 'downsample_rate': None,
        # 'dump_csv': [],
        # 'dump_debug': [],
        # 'generate_plots': [],
        # 'module_seq': 'ref>ao>pp>va>li>wh>hvac',
        # 'run_mode': None,

        params_dict = {
            'cache_mode': False,
        }

        """WRITING HAS BEEN DISABLED"""

        # Call the function that will populate the queue and spawn the processes
        run_multiple_users(multi_run_params_dict, params_dict)

        print('Run completed!')
