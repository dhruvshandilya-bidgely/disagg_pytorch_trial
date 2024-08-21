"""
Author - Mayank Sharan
Date - 18th Sep 2018
initialisation config initializes fields in config to prepare for the run
"""

# Import functions from within the project
from python3.config.Cgbdisagg import Cgbdisagg
from python3.config.pilot_constants import PilotConstants
from python3.disaggregation.aes.init_aes_config import SmbConstants
from python3.config.pipeline_constants import GlobalConfigParams


def initiaize_lifestyle_config(config, is_smb_user, is_smb_active_pilot):
    """
    Function to initialize analytics config for smb and non smb users

    Parameters:
        config              (dict): Dictionary containing config info
        is_smb_user         (bool): Boolean indicating if user is smb
        is_smb_active_pilot (bool): Boolean indicating if pilot is smb enabled

    Returns:
        config              (dict): Dictionary containing config info
    """

    if 'analytics' in config.get('pipeline_seq'):

        if is_smb_user & is_smb_active_pilot:
            config['analytics_seq'] = GlobalConfigParams.analytics_aes_seq

        else:
            config['analytics_seq'] = GlobalConfigParams.analytics_aer_seq

    return config


def init_global_config(init_params):

    """
    Parameters:
        init_params         (dict)              : Contains information used to fetch the data

    Returns:
        config              (dict)              : Dictionary containing all parameters to run the pipeline
    """

    config_params = init_params['config_params']
    fetch_params = init_params['fetch_params']
    pipeline_run_data = init_params['disagg_run_data']

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
    # Add 'tou_all' in dump_csv list to dump appliance tou in a csv file at the log directory location

    config = {
        'analytics_seq': [],
        'api_env': str.lower(fetch_params['api_env']),
        'cache_mode': False,
        'data_type': GlobalConfigParams.data_type.get('ami'),
        'disagg_mode': None,
        'disagg_aer_seq': [],
        'disagg_aes_seq': [],
        'downsample_rate': None,
        'dump_csv': [],
        'dump_debug': [],
        'fuel_type': GlobalConfigParams.fuel_type.get('electric'),
        'generate_plots': [],
        'itemization_seq': [],
        'pilot_id': pipeline_run_data.get('pilotId'),
        'pipeline_seq': GlobalConfigParams.pipeline_seq.get('disagg_itemization_analytics'),
        'priority': False,
        'run_mode': None,
        'user_type': GlobalConfigParams.user_type.get('residential'),
        'uuid': fetch_params.get('uuid'),
        'write_results': True,
        'write_vip_user': False,
        'trigger_id': fetch_params.get('trigger_id'),
        'trigger_name': fetch_params.get('trigger_name'),
        'retry_count': fetch_params.get('retry_count'),
        'zipcode': fetch_params.get('zipcode'),
        'country': fetch_params.get('country'),
        'enable_ev_by_default': pipeline_run_data.get('enableEV'),
        'enable_hybrid_v2': pipeline_run_data.get('enableHybridV2'),
        'hybrid_model_files': pipeline_run_data.get('hybridModelFiles'),
        'build_info': fetch_params.get('build_info'),
    }

    # Set fields in config as per the parameters given in the function

    config_arg_name = config_params.keys()
    config_arg_arr = config.keys()

    for arg_name in config_arg_name:
        if arg_name in config_arg_arr:
            set_value = config_params[arg_name]

            if arg_name == 'dump_csv' or \
               arg_name == 'dump_debug' or arg_name == 'generate_plots':
                set_value = set_value.split('>')

            config[arg_name] = set_value
        else:
            print(arg_name, 'not in config variable')

    # Set run as production run unless mentioned as parameter and do not override mode set in case of custom run

    if config['run_mode'] is None or config['run_mode'] == 'prod':

        config['run_mode'] = 'prod'

        # Backend can give us 5 possible disagg mode values, we are equipped to handle all of them

        config['disagg_mode'] = init_params.get('disagg_mode')

    elif config['run_mode'] == 'custom'and config['disagg_mode'] is None:
        config['disagg_mode'] = 'historical'

    default_aer_sequence = ['ao', 'ref', 'pp', 'wh', 'ev', 'va', 'hvac', 'li']

    # Override default Disagg AER sequence with custom requested module sequence in case of local run

    requested_module_sequence = GlobalConfigParams.disagg_aer_seq
    if len(requested_module_sequence) > 0:
        default_aer_sequence = requested_module_sequence
    config['disagg_aer_seq'] = default_aer_sequence

    # Get the Itemization sequence required

    requested_module_sequence = GlobalConfigParams.itemization_seq
    config['itemization_seq'] = requested_module_sequence

    # Check if this user is an SMB from Manual input

    if init_params.get('home_meta_data').get(SmbConstants.SMB_ID_KEY) is None:
        init_params['home_meta_data'][SmbConstants.SMB_ID_KEY] = SmbConstants.user_segment

    # Check if this user is an SMB

    is_smb_user = init_params.get('home_meta_data').get(SmbConstants.SMB_ID_KEY, '').lower() == 'smb'
    is_smb_active_pilot = config.get('pilot_id') in PilotConstants.SMB_PILOT_LIST

    # Initialise a default SMB module sequence

    default_aes_sequence = ['ao_smb', 'li_smb', 'work_hours', 'hvac_smb']

    # If the user is SMB then initialize a SMB module sequence

    if is_smb_user & is_smb_active_pilot:

        # Get the user configured SMB module sequence
        requested_module_sequence = GlobalConfigParams.disagg_aes_seq
        if len(requested_module_sequence) > 0:
            default_aes_sequence = requested_module_sequence
        config['disagg_aes_seq'] = default_aes_sequence

        config['user_type'] = 'smb'

    # Initialise Lifestyle module sequence

    config = initiaize_lifestyle_config(config, is_smb_user, is_smb_active_pilot)

    return config
