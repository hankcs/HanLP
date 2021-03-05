# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 19:26
from os.path import abspath, join, dirname
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()
version = {}
with open(join(this_dir, "hanlp", "version.py")) as fp:
    exec(fp.read(), version)

setup(
    name='hanlp',
    version=version['__version__'],
    description='HanLP: Han Language Processing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/hankcs/HanLP',
    author='hankcs',
    author_email='hankcshe@gmail.com',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Development Status :: 3 - Alpha",
        'Operating System :: OS Independent',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Topic :: Text Processing :: Linguistic"
    ],
    keywords='corpus,machine-learning,NLU,NLP',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=[
        'termcolor',
        'pynvml',
        'alnlp',
        'toposort==1.5',
        'transformers>=4.1.1',
        'sentencepiece>=0.1.91'
        'torch>=1.6.0',
        'hanlp-common>=0.0.6',
        'hanlp-trie>=0.0.2',
        'hanlp-downloader',
    ],
    extras_require={
        'full': [
            'fasttext==0.9.1',
            'tensorflow==2.3.0',
            'bert-for-tf2==0.14.6',
            'py-params==0.9.7',
            'params-flow==0.8.2',
            'penman==0.6.2',
        ],
    },
    python_requires='>=3.6',
    # entry_points={
    #     'console_scripts': [
    #         'hanlp=pyhanlp.main:main',
    #     ],
    # },
)
