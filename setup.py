# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 19:26
import sys
from os.path import abspath, join, dirname
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()
version = {}
with open(join(this_dir, "hanlp", "version.py")) as fp:
    exec(fp.read(), version)

FASTTEXT = 'fasttext-wheel==0.9.2'
sys_version_info = sys.version_info

EXTRAS = []
if sys.platform in {'darwin', 'win32'}:
    if (sys_version_info.major, sys_version_info.minor) == (3, 6):
        EXTRAS = ['tokenizers==0.10.3']
    elif (sys_version_info.major, sys_version_info.minor) == (3, 7):
        EXTRAS = ['safetensors<0.5']  # Failed to build safetensors

extras_require = {
    'amr': [
        'penman==1.2.1',
        'networkx>=2.5.1',
        'perin-parser>=0.0.12',
    ],
    'fasttext': [FASTTEXT],
    'tf': [FASTTEXT,
           'tensorflow>=2.6.0,<2.14; python_version < "3.12"',
           'tensorflow>=2.16,<2.17; python_version >= "3.12" and python_version < "3.13"',
           'tf-keras>=2.16,<2.17; python_version >= "3.12" and python_version < "3.13"',
           'tensorflow>=2.21,<2.22; python_version >= "3.13" and python_version < "3.14" and (sys_platform != "darwin" or platform_machine != "x86_64")',
           'tf-keras>=2.21,<2.22; python_version >= "3.13" and python_version < "3.14" and (sys_platform != "darwin" or platform_machine != "x86_64")',
           'transformers<4.55; python_version < "3.14"']  # TF is deprecated in Transformers and no longer maintained
}
extras_require['full'] = list(set(sum(extras_require.values(), [])))

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
        "Development Status :: 4 - Beta",
        'Operating System :: OS Independent',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Topic :: Text Processing :: Linguistic"
    ],
    keywords='corpus,machine-learning,NLU,NLP',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=[
        'termcolor',
        'nvidia-ml-py',
        'toposort==1.5',
        'transformers>=4.1.1',
        'sentencepiece>=0.1.91',  # Essential for tokenization_bert_japanese
        'torch>=1.6.0',
        'hanlp-common>=0.0.23',
        'hanlp-trie>=0.0.4',
        'hanlp-downloader',
        *EXTRAS,
    ],
    extras_require=extras_require,
    python_requires='>=3.6',
)
