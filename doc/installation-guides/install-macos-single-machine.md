## How to Install ActiveMatcher on a MacOS M1 Machine

This is a step-by-step guide to install ActiveMatcher on a single macOS machine with an M1 chip. If you are unsure whether your Mac has an M1 chip, click on the Apple in the top left corner of your screen > About This Mac. If it says Chip Apple M1, then you have an M1 chip. If it does not say Chip Apple M1, you do not have an M1 chip.

This guide has been tested on a 2020 MacBook Pro with an Apple M1 Chip, 8GB Memory, macOS version Sequoia 15.0.1, and a .zshrc profile, using Python 3.12. You can try to adapt this guide to other configurations.

If your machine has an Intel chip, this installation guide will not work for you. If your machine has an M2, M3, or M4 chip, this installation guide may work for you, but we have not tested it.

### Step 1: Installing Python

We first install Python 3.12, create a virtual environment, and then install two Python packages setuptools and build. Other versions of Python, other environments, or incorrect installations of the setuptools and build packages can cause issues with ActiveMatcher installation.

If you suspect that you may have Python downloaded on your machine already, open up your terminal. Then, run the command:

	which python

If the output path says

“/usr/local/bin/python”

run:

	python --version

If the output of this is

“Python 3.12.x”

where x is a number, you can go to Step 1C after completing Step 1A (you do not need to complete Step 1B).

If

which python

Or

python --version

did not have the outputs listed above, do all substeps A-D.

#### Step 1A: Installing Homebrew

Before installing Python, we need to ensure we have Homebrew installed. Homebrew is a popular open-source package manager for macOS and Linux, used to simplify installing, updating, and managing software and libraries.

To check if Homebrew is installed, open up a terminal. Then, type
```
brew info
```
If the output contains kegs, files, and GB, then Homebrew is installed and you can go to Step 1B. Otherwise, you need to install Homebrew.

To install Homebrew, run the following command in your terminal:
```
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com](https://raw.githubusercontent.com)
Homebrew/install/HEAD/install.sh)"
```
The installation may prompt you to follow further instructions to add this to your PATH variables; if so, follow those onscreen instructions. This will make it easier to use brew later. If you see

“Installation Successful!” 

in the output, you are done with this step. 

#### Step 1B: Installing Python 3.12

To download Python environments, we will use Homebrew. Run the following in the terminal to install Python 3.12:
```
	brew install python@3.12
```
#### Step 1C: Setting Up the Python Environment

Now we will create a Python environment with Python 3.12. This step is necessary to make sure we use the correct version of Python with the correct dependencies. In your terminal, run:
```
	python -m venv ~/active_matcher
```
This will create a virtual environment named active_matcher. To activate this environment, run the following:
```
	source ~/active_matcher/bin/activate
```
To make sure everything is correct, run:
```
	python --version
```
If the output says

“Python 3.12.x”

where x ≥ 0, then the Python environment setup is successful.

#### Step 1D: Installing Python Packages setuptools and build

Before installing these packages, make sure you are in the virtual environment. If you have just finished Step 1C, you are in the virtual environment. Otherwise, to make sure your virtual environment is active, you can run:
```
	source ~/active_matcher/bin/activate
```
To install setuptools, run:
```
	pip install setuptools
```
To install build, run:
```
	pip install build
```
If at any point during the installation you close your terminal, you will need to reactivate your virtual environment by running:
```
	source ~/active_matcher/bin/activate
```
### Step 2: Installing ActiveMatcher

To install ActiveMatcher, use one of the following two options:

#### Option 1: Pip Installing from PyPI

You can install ActiveMatcher from PyPI, using the following command:
```
	pip install active_matcher
```
This command will install ActiveMatcher and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Py_Stringmatching, PySpark, Scikit-Learn, Scipy, Threadpoolctl, TQDM, and Xgboost.

#### Option 2: Pip Installing from GitHub

Instead of pip installing from PyPI, you may want to pip install ActiveMatcher from its GitHub repo. This happens if you want to install the latest ActiveMatcher version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

To install ActiveMatcher directly from its GitHub repo, use the following command:

	pip install git+https://github.com/anhaidgroup/active_matcher.git@main

Similar to pip installing from PyPI, the above command will install ActiveMatcher and all of its dependencies.

