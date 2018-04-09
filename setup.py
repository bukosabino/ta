from distutils.core import setup
setup(
    name = 'ta',
    packages = ['ta'],
    version = '0.1.4',
    description='Technical Analysis Library in Python',
    long_description='It is a technical analysis library to financial time series datasets. You can use to do feature engineering. It is builded on pandas python library.',
    author = 'Dario Lopez Padial (Bukosabino)',
    author_email = 'bukosabino@gmail.com',
    url = 'https://github.com/bukosabino/ta',
    maintainer='Dario Lopez Padial (Bukosabino)',
    maintainer_email='bukosabino@gmail.com',
    install_requires=[
        'numpy',
        'pandas'
    ],
    download_url = 'https://github.com/bukosabino/ta/tarball/0.1.4',
    keywords = ['technical analysis', 'python3', 'pandas'],
    license='The MIT License (MIT)',
    classifiers = [],
)
