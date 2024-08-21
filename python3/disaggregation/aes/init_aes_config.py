"""
This config file contains SMB constants
"""


class SmbConstants :
    """
    This class contains all the constants for the SMB run
    """

    # For manual control on SMB in absence of tag from backend
    SMB_ID_KEY = 'userSegment'
    user_segment = ''


def init_aes_config(pipeline_input_object):

    """
    Parameters:
        pipeline_input_object (dict)              : Contains information used to fetch the data

    Returns:
        config              (dict)              : Dictionary containing all parameters to run the pipeline
    """

    # Initialize fields in config to default values

    # api_env:          Set to the environment to be used for the APi calls, DS/NA/EU
    # cache_mode:       Save and load data locally
    # disagg_mode:      Pipeline runs according to this
    # downsample_rate:  Allows custom downsampling of data
    # dump_csv:         List of modules for which to dump csv with information
    # dump_debug:       List of modules for which to dump debug as pickle
    # generate_plots:   List of modules for which to dump plots
    # module_seq:       Sequence to execute modules in
    # pilot_id:         Pilot id of the user
    # run_mode:         What mode are we running in prod or custom
    # sampling_rate:    Sampling rate of the data in seconds
    # uuid:             UUID for the user
    # user type :       Identification of type of user : SMB or residential or anything else
    # write_vip_user:   Parameter to allow the current run to be written in the VIP user sheet for future use

    config = {
        'api_env': pipeline_input_object.get('global_config').get('api_env'),
        'cache_mode': pipeline_input_object.get('global_config').get('cache_mode'),
        'disagg_mode': pipeline_input_object.get('global_config').get('disagg_mode'),
        'downsample_rate': pipeline_input_object.get('global_config').get('downsample_rate'),
        'dump_csv': pipeline_input_object.get('global_config').get('dump_csv'),
        'dump_debug': pipeline_input_object.get('global_config').get('dump_debug'),
        'generate_plots': pipeline_input_object.get('global_config').get('generate_plots'),
        'module_seq': pipeline_input_object.get('global_config').get('disagg_aes_seq'),
        'pilot_id': pipeline_input_object.get('global_config').get('pilot_id'),
        'priority': pipeline_input_object.get('global_config').get('priority'),
        'run_mode': pipeline_input_object.get('global_config').get('run_mode'),
        'sampling_rate': pipeline_input_object.get('global_config').get('sampling_rate'),
        'uuid': pipeline_input_object.get('global_config').get('uuid'),
        'user_type': pipeline_input_object.get('global_config').get('user_type'),
        'write_results': pipeline_input_object.get('global_config').get('write_results'),
        'write_vip_user': pipeline_input_object.get('global_config').get('write_vip_user'),
        'trigger_id': pipeline_input_object.get('global_config').get('trigger_id'),
        'trigger_name': pipeline_input_object.get('global_config').get('trigger_name'),
        'retry_count': pipeline_input_object.get('global_config').get('retry_count'),
        'zipcode': pipeline_input_object.get('global_config').get('zipcode'),
        'country': pipeline_input_object.get('global_config').get('country')
    }

    return config





