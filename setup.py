# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


requirements = [
    'coord-sim==2.2.1',
    'click==7.0',
    'numpy>=1.16.5,<1.19',
    'keras==2.2.5',
    'keras-rl==0.4.2',
    'tensorflow==1.14.0',
    'cloudpickle==1.2',
    'gym[atari]==0.14.0',
    'pandas==1.1.5',
    'h5py==2.10',
    'protobuf==3.19.6'
]

test_requirements = [
    'flake8',
    'nose2'
]

setup(
    name='drlsim',
    version='1.1.2',
    description='drlsim: Self-Learning Network and Service Coordination Using Deep Reinforcement Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/brajeshumrao84/DrlSim',
    author='Brajesh Kumar Umrao',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    package_data={'rlsp.agents': ['logging.conf']},
    python_requires=">=3.6, <3.8",
    install_requires=requirements + test_requirements,
    tests_require=test_requirements,
    test_suite='nose2.collector.collector',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            "drlsim=rlsp.agents.main:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
