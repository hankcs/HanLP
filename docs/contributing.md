# Contributing Guide

Thank you for being interested in contributing to `HanLP`! You
are awesome ✨.

This guideline contains information about our conventions around coding style, pull request workflow, commit messages and more.

This page also contains information to help you get started with development on this
project.

## Development

### Set-up

Get the source code of this project using git:

```bash
git clone https://github.com/hankcs/HanLP
cd HanLP
pip install -e plugins/hanlp_trie
pip install -e plugins/hanlp_common
pip install -e plugins/hanlp_restful
pip install -e .
```

To work on this project, you need Python 3.6 or newer.

### Running Tests

This project has a test suite to ensure certain important APIs work properly. The tests can be run using:

```console
$ python -m unittest discover ./tests
```

:::{tip}
It's hard to cover every API especially those of deep learning models, due to the limited computation resource of CI. However, we suggest all inference APIs to be tested at least.

:::

## Repository Structure

This repository is a split into a few critical folders:

hanlp/
: The HanLP core package, containing the Python code.

plugins/
: Contains codes shared across several individual packages or non core APIs.

docs/
: The documentation for HanLP, which is in markdown format mostly.
: The build configuration is contained in `conf.py`.

tests/
: Testing infrastructure that uses `unittest` to ensure the output of API is what we expect it to be.

.github/
: Contains Continuous-integration (CI) workflows, run on commits/PRs to the GitHub repository.

