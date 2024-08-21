"""
Author - Mayank Sharan
Date - 20/09/18
run gb pipeline is a wrapper to call all functions as needed to run the gb diasggregation pipeline
"""

# Import python packages

import time


class OauthToken:
    """
    Gets token for api specified
    """

    def __init__(self, access_token, token_type, expires_in, scope, url):
        self.access_token = access_token
        self.token_type = token_type
        self.scope = scope
        self.url = url
        self.expiry_time = int(time.time()) + expires_in

    def get_access_token(self):
        return self.access_token

    def get_token_type(self):
        return self.token_type

    def get_scope(self):
        return self.scope

    def get_url(self):
        return self.url

    def is_expired(self):
        return int(time.time()) > self.expiry_time

    def __str__(self):
        return "oauth_token[access_token = {0}, token_type =  {1}, scope =  {2}, url =  {3}, expiry_time =  {4}".format(
            self.access_token, self.token_type,
            self.scope, self.url, self.expiry_time)

    def __repr__(self):
        return self.__str__()
