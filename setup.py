import json
import os
import sys
from setuptools import setup

#import version

setup_args = {
    'name': 'VisModel',
    'author': 'Phil Bull',
    'url': 'https://github.com/philbull/VisModel',
    'license': 'MIT',
    'version': '0.0.1',
    'description': 'Visibility forward modelling and statistical sampling.',
    'packages': ['VisModel'],
    'package_dir': {'VisModel': 'VisModel'},
    'install_requires': [
        'numpy>=1.15',
        'scipy',
        'matplotlib>=2.2'
        'pyuvdata',
    ],
    'zip_safe': False,
}

if __name__ == '__main__':
    setup(**setup_args)
