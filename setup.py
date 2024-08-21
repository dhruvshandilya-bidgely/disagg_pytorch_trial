#!/usr/bin/env python3

from distutils.core import setup
import os
from setuptools import setup, find_packages

setup(name='slytherin',
        version='1.0',
        packages=find_packages(),
        install_requires=[],
        include_package_data=True,
        scripts=['python3/queue_mode.py'],
        description='Python based disagg',
        author='Eng Bidgely',
        author_email='eng@bidgely.com',
        url='https://www.bidgely.com',
    )
