# -*- coding: utf-8 -*-
from distutils.core import setup

setup(
    name='ta',
    packages=['ta'],
    version='0.5.25',
    description='Technical Analysis Library in Python',
    long_description='It is a Technical Analysis library to financial time series datasets. You can use to do feature engineering. It is builded on Python Pandas library.',
    author='Dario Lopez Padial (Bukosabino)',
    author_email='Bukosabino@gmail.com',
    url='https://github.com/bukosabino/ta',
    maintainer='Dario Lopez Padial (Bukosabino)',
    maintainer_email='Bukosabino@gmail.com',
    install_requires=[
        'numpy',
        'pandas',
    ],
    download_url='https://github.com/bukosabino/ta/tarball/0.5.25',
    keywords=['technical analysis', 'python3', 'pandas'],
    license='The MIT License (MIT)',
    classifiers=[
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License'
    ],
    project_urls={
        'Documentation': 'https://technical-analysis-library-in-python.readthedocs.io/en/latest/',
        'Bug Reports': 'https://github.com/bukosabino/ta/issues',
        'Source': 'https://github.com/bukosabino/ta',
    },
)
