"""
Author - Mayank Sharan
Date - 20/09/18
run gb pipeline is a wrapper to call all functions as needed to run the gb diasggregation pipeline
"""

# Import python packages

import os
import boto3
import base64
import requests
import threading

# Import functions from within the project

from python3.config.Cgbdisagg import Cgbdisagg
from python3.utils.oauth_token import OauthToken
from python3.config.mappings.get_env_url import get_base_url
from python3.config.mappings.get_env_properties import get_env_properties


def update_oauth_credentials(client_id_dict, client_secret_dict, api_env):

    """
    Utility to populate the latest credentials from simpleDB for oauth token fetch
    """

    # Extract aws region as per the api env

    env_prop = get_env_properties(api_env)
    aws_region = env_prop.get('aws_region')

    # Pull the client id and secret from simple DB, hard coded to us-east-1 since that is how it is

    client_sdb = boto3.client('sdb', region_name='us-east-1')

    client_id_response = client_sdb.get_attributes(DomainName='springPropertiesMapping',
                                                   ItemName=api_env + ':oauth.admin.clientid',
                                                   AttributeNames=['value'])

    client_secret_response = client_sdb.get_attributes(DomainName='springPropertiesMapping',
                                                       ItemName=api_env + ':oauth.admin.clientsecret',
                                                       AttributeNames=['value', ])

    # Decrypt the client id and secret and populate them in the dictionary

    client_kms = boto3.client('kms', region_name=aws_region)

    client_id_cipher = base64.b64decode(client_id_response['Attributes'][0]['Value'])
    decrypted_client_id = client_kms.decrypt(CiphertextBlob=client_id_cipher)

    client_secret_cipher = base64.b64decode(client_secret_response['Attributes'][0]['Value'])
    decrypted_client_secret = client_kms.decrypt(CiphertextBlob=client_secret_cipher)

    # Populated the decrypted client id and secret

    client_id_dict[api_env] = decrypted_client_id['Plaintext'].decode()
    client_secret_dict[api_env] = decrypted_client_secret['Plaintext'].decode()

    return client_id_dict, client_secret_dict


class OauthTokenManager:

    """
    Manager class to get auth token by env
    """

    # Initialize the base url to be used to extract the Oauth token

    url_string = '{0}{1}:{2}@{3}/oauth/token?grant_type=client_credentials&scope=all'

    # Initialize a dictionary of token objects that are accessible by the key API environment

    token = {}

    # Initialize client id and client secret dictionaries to access the credentials

    client_id = Cgbdisagg.CLIENT_ID_DICT
    client_secret = Cgbdisagg.CLIENT_SECRET_DICT

    env = os.getenv('BIDGELY_ENV')

    if env is not None:
        client_id, client_secret = update_oauth_credentials(client_id, client_secret, env)

    # Initialize the lock to be used

    lock = threading.Lock()

    @staticmethod
    def get_access_token(api_env):

        """
        Method to get access token based on env
        """

        env_token = OauthTokenManager.token.get(api_env)

        if env_token:
            if env_token.is_expired():
                OauthTokenManager.renew_access_token(api_env)
            else:
                return env_token.access_token

        else:
            OauthTokenManager.renew_access_token(api_env)

        return OauthTokenManager.token.get(api_env).access_token

    @staticmethod
    def renew_access_token(api_env):
        """
        Method to renew access token
        """

        OauthTokenManager.lock.acquire()

        # noinspection PyBroadException
        try:

            # Attempt to fetch Oauth token using the primary api server

            base_url, protocol = get_base_url(api_env)

            url = OauthTokenManager.url_string.format(protocol, OauthTokenManager.client_id.get(api_env),
                                                      OauthTokenManager.client_secret.get(api_env), base_url)

            response = requests.get(url)
            message_load = response.json()

            # If token fetched is empty try the secondary api server

            if message_load.get('access_token') is not None:

                OauthTokenManager.token[api_env] = OauthToken(message_load['access_token'],
                                                              message_load['token_type'],
                                                              message_load['expires_in'],
                                                              message_load['scope'],
                                                              message_load['url'])

            else:

                base_url, protocol = get_base_url(api_env, server_type='secondary')

                url_sec = OauthTokenManager.url_string.format(protocol, OauthTokenManager.client_id.get(api_env),
                                                              OauthTokenManager.client_secret.get(api_env), base_url)

                response = requests.get(url_sec)
                message_load = response.json()
                OauthTokenManager.token[api_env] = OauthToken(message_load['access_token'],
                                                              message_load['token_type'],
                                                              message_load['expires_in'],
                                                              message_load['scope'],
                                                              message_load['url'])

        except Exception:

            # If an error is raised try the secondary api server

            base_url, protocol = get_base_url(api_env, server_type='secondary')

            url_sec = OauthTokenManager.url_string.format(protocol, OauthTokenManager.client_id.get(api_env),
                                                          OauthTokenManager.client_secret.get(api_env), base_url)

            response = requests.get(url_sec)
            message_load = response.json()
            OauthTokenManager.token[api_env] = OauthToken(message_load['access_token'],
                                                          message_load['token_type'],
                                                          message_load['expires_in'],
                                                          message_load['scope'],
                                                          message_load['url'])
        finally:
            OauthTokenManager.lock.release()
