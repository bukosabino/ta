from distutils.core import setup
setup(
    name = 'ta',
    packages = ['ta'],
    version = '0.1',
    description='Technical Analysis Library in Python',
    long_description=open('README.rst').read(),
    author = 'Dario Lopez Padial (Bukosabino)',
    author_email = 'bukosabino@gmail.com',
    url = 'https://github.com/bukosabino/ta',
    maintainer='Dario Lopez Padial (Bukosabino)',
    maintainer_email='bukosabino@gmail.com',
    install_requires=[
        'python3',
        'numpy',
        'pandas'
    ],
    # download_url = 'https://github.com/bukosabino/ta/tarball/0.1',
    keywords = ['testing', 'logging', 'example'],
    license='The MIT License (MIT)',
    classifiers = [],
)
