"""
Author - Mayank Sharan
Date - 19/09/18
initialisation disagg input object initiates single instance of disagg input object
"""

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.master_pipeline.init_global_config import init_global_config


def init_pipeline_input_object(init_params, fetch_params):

    """
    Parameters:
        init_params         (dict)              : Dictionary containing all parameters needed to initialize the object

    Returns:
        pipeline_input_object (dict)              : Dictionary containing all parameters to run the pipeline
    """

    pipeline_input_object = dict({
        'app_profile': init_params.get('app_profile'),
        'global_config': None,
        'gb_pipeline_event': init_params.get('pipeline_event'),
        'home_meta_data': init_params.get('home_meta_data'),
        'appliances_hsm': init_params.get('hsm_dict'),
        'input_data': init_params.get('input_data'),
        'logging_dict': init_params.get('logging_dict'),
        'logger': None,
        'original_input_data': None,
        'out_bill_cycles': init_params.get('billing_cycles'),
        'out_bill_cycles_by_module': init_params.get('billing_cycles_by_module'),
        'data_quality_metrics': None,
        'loaded_files': init_params.get('loaded_files'),
        'build_info': fetch_params.get('build_info'),
        'store_tb_in_cassandra': fetch_params.get('store_tb_in_cassandra')
    })

    pipeline_input_object['global_config'] = init_global_config(init_params)

    return pipeline_input_object
