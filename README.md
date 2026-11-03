# Introduction

This research repo studies the potential of a neural LLR-head.

## Getting Started

### Installation process

- Clone this repo on azure.
- Ensure you have configured the username/token for the `repo.minjus.nl/repository/pip-group/simple` Nexus repository.
  - see https://confluence.dev.holmes.nl/pages/viewpage.action?pageId=196619335

#### Option 1: Use PDM to install and manage requirements

- Follow the instructions to configure PDM to use the correct Nexus repository and authentication credentials.
  - see https://confluence.dev.holmes.nl/pages/viewpage.action?pageId=196619335
- Configure PDM to use python 3.12: `pdm use 3.12`
- Run `pmd install` to install all dependencies

#### Option 2: create a virtualenv with python and use pip to manage requirements

If you'd rather not use `pdm`, you can use `pip` as well.

- Create a virtualenv with py3.12 (py3.10 may also work).
- Activate the virtualenv and install the requirements from `requirements.txt` (`pip install -r requirements.txt`)
- Install the test dependencies with `pip install -r test-requirements.txt`

Note that there are versioned `requirements.txt` for python 3.10 as well, but mileage may vary, as the project is
build with python 3.12 in mind.

## Running the tests

The tests can be run manually using `pdm run tox`.

## Structure of the repo

All AI-code is located in the folder DL_modules, which holds 7 folders.
- criterion: holds code for several losses
  - brier.py: brier score for LRs.
  - cllr.py: several versions of cllr.
  - KL_div_to_normal.py: KL Divergence when target distribution is standard normal.
  - pav_loss.py: RootMeanSquare loss of PAV transform to identity line.
- custom_layers: holds code for custom AI-layers
  - contract_axes.py: transforms axes by shrinking values towards 0. Shrinkage increases when input is farther away
  from zero. This may be useful to prevent outliers for the between distribution of a TwoLevelModel (TLM).
  - dense_sequential.py: several combinations of Dense layers and activation functions. Can be used as building blocks
  when designing AI-models. We will probably not use them since we will focus on the top/head.
  - gaussian_mixture.py: holds a Gaussian mixture AI-layer. We will use this to implement KDE when we can train TLMs
  on single instances. GMM holds functions to calculate SS and CS LRs.
  - pair_to_output.py: when predicting LRs, input is a pair and output is a single value. Holds wrapper code that
  facilitates building an AI model from pair to single output.
  - top_layers.py: holds TLM and Sigmoid (logistic regression) AI-layers to be used as top/head.
  - two_level_normal_normal_sub_modules: holds sub_modules that are used by TLM layers in top_layers.py
- database: holds code for database connection. Not needed when you want to run experiments on existing data in the
  resources folder
- dataloader: holds code for dataloaders
- plots: holds code for several plots for evaluation and validateing
  - calibration_plot.py: graph of PAV calibration
  - ece_plot_llrs.py: ece plot when ece is calculated in log domain (e.g. for llrs)
  - kde.py: compares KDE of data to normal approximation
  - loss_plot.py: holds loss plots (loss versus epoch), plots for llr distributions, and qnorm-plots
  (comparing quantiles of data to quantiles of standard normal distribution)
- resources: data downloaded from the database: the glass data. Stored in splits.
  - glas_1rep_train_test: split-in-two. We will not use this.
  - glas_1rep_train_validate_test: split-in-three.
    - all_train_data: train pairs with all negatives
    - data_by_source: single instances
    - selected_train_data: training pairs with easy negatives removed (about 3% of the negatives).
    This data, in combination with validate_data is used when training AI-models.
    - validate_data: validate pairs
    - test_data: test pairs. The hold_out set for final evaluation.
- train_validate_loops: code for AI-model training by gradient descent, one function for TLM head, one function for Sigmoid
head

## Training an AI-model

A training experiment is defined by two files, both located in the root: a 'train'.py and a 'train'.yaml file.
For example: train-benchmark-glass-1-rep-DL-logreg-model-train-validate-test.py and its yaml with the same stem.
When you define an output directory in the yaml, you also need te create that directory in the root of the repo.

Run the .py file to train and validate a model.
Train new models by creating new 'train'.py and 'train'.yaml files.
If you run a 'train.py' repetitively, the output is saved in an automatically updated output folder.

## Working with git

See: https://dev.azure.com/NetherlandsForensicInstitute/CFS-Data/_git/CFS-Data/?_a=files&path=/docs/onboarding/learning-resources.md

### Contribute code: git and precommit hook

[pre-commit package](https://pre-commit.com/)

The pre-commit package can be used to automate certain checks (e.g. `ruff format`, a code quality check) before every
commit or every push.
We recommend to install pre-commit globally so that you do not have to reinstall it in new environments.
Below is explained how installation and usage work.

#### Global installation of pre-commit package

Type in a command shell outside the virtual environment

```shell
pip install pre-commit
```

#### Usage of git and precommit hook when coding

In command shell (e.g. in pycharm) within your virtual environment

```shell
pre-commit install --hook-type pre-commit --hook-type pre-push
```

When coding, and you intend to commit and push: in command shell within your virtual environment

```shell
ruff check --fix
ruff format
```

If you have not 'git committed' yet or when applying ruff results in changes you do the following:

- In command shell within your virtual environment (You do this to check whether the desired files are listed for
commit and push.)

```shell
git status
```

- If there are files you want to commit and push to azure:

```shell
git add .
git commit -m '<message>'
git push
```
