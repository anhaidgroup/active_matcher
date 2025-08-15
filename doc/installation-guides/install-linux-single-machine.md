## How to Install ActiveMatcher on a Linux Machine

This is a step-by-step guide to install ActiveMatcher on a single Linux machine with Ubunutu 22.0.4 and Python 3.12. You can try to adapt this guide to other configurations.

### Step 1: Installing Essential Packages

First we install packages that are neccessary for installing Java and Python and for installation in the later steps. In particular, we will install build-essential, libbz2-dev, libssl-dev, libreadline-dev, libsqlite3-dev, libffi-dev, zlib1g-dev, libncurses5-dev, libncursesw5-dev, xz-utils, tk-dev, and wget.

To do so, open up your terminal and run the following commands:

```
sudo apt update
sudo apt install -y \
    build-essential \
    libbz2-dev \
    libssl-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    wget
```

Once these have been installed, we can install Java, Python, and ActiveMatcher.

### Step 2: Installing Java

We strongly recommend installing Java Temurin JDK 17, which is a specific Java release that we have extensively experimented with. Since that release is not available from the Ubuntu package repository, we will have to install it. To do so, use the following commands.

First,

```
sudo -s
```

Then,

```
apt install wget apt-transport-https gnupg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release)     main" | tee /etc/apt/sources.list.d/adoptium.list
apt update
apt install temurin-17-jdk
```

Finally,

```
exit
```

You can check that you have successfully installed Java by running the following command. If Java is installed, it should display a version number.

```
java --version
```

### Step 3: Installing Python

We now install Python 3.12, create a virtual environment, and install two Python packages setuptools and build. Other versions of Python, other environments, or incorrect installations of the setuptools and build packages can cause issues with ActiveMatcher installation.

If you suspect that you may have Python downloaded on your machine already, open up your terminal. Then run the command:

```
    which python3
```

If the output path says

“/usr/local/bin/python3”

run:

```
    python3 --version
```

If the output of this is

“Python 3.12.x”

where x is a number, you can go to Step 3B (you do not need to complete Step 3A).

If
```
which python3
```
or
```
python3 --version
```
do not have the outputs listed above, continue to step 3A. 

#### Step 3A: Installing Python 3.12

Here we download Python 3.12, install it, and make it the default verison.
Run the following commands in the terminal to install Python 3.12:

```
    cd /usr/src
    sudo curl -O https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
    sudo tar xzf Python-3.12.3.tgz
```
```
    cd Python-3.12.3
    sudo make clean
    sudo ./configure --enable-optimizations --with-system-ffi
    sudo make -j$(nproc)
    sudo make altinstall
```
Now run the following commands to make Python 3.12 the default: 
```
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1
python3.12 -m ensurepip --default-pip
python3.12 -m pip install --upgrade pip setuptools
```

#### Step 3B: Setting Up the Python Environment

Now we will create a Python environment with Python 3.12. This step is necessary to make sure we use the correct version of Python with the correct dependencies. First, we install the venv module with the following command:
```
    sudo apt install python3-venv
```

Next, in your terminal, run:
```
    python3 -m venv ~/active_matcher
```

This will create a virtual environment named active_matcher. To activate this environment, run the following:
```
    source ~/active_matcher/bin/activate
```

To make sure everything is correct, run:
```
    python3 --version
```

If the output says

“Python 3.12.x”

where x ≥ 0, then the Python environment setup was successful.

#### Step 3C: Installing the Python Packages setuptools and build

Before installing these two packages, make sure you are in the virtual environment. If you have just finished Step 3B, you are in the virtual environment. Otherwise, to make sure your virtual environment is active, you can run:

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

### Step 4: Installing ActiveMatcher

Before installing ActiveMatcher, we should return to the root directory by running the following command in the terminal:

```
    cd
```

In the future you can install ActiveMatcher using one of the following two options. **As of now, since ActiveMatcher is still in testing, we do not yet enable Option 1 (Pip installing from PyPI). Thus you should use Option 2 (Pip installing from GitHub).**

#### Option 1: Pip Installing from PyPI

**Note that this option is not yet enabled. Please use Option 2.**

You can install ActiveMatcher from PyPI using the following command:

```
    pip install active_matcher
```

This command will install ActiveMatcher and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Py_Stringmatching, PySpark, Scikit-Learn, Scipy, Threadpoolctl, TQDM, and Xgboost.

#### Option 2: Pip Installing from GitHub

Instead of pip installing from PyPI, you may want to pip install ActiveMatcher from its GitHub repo. This happens if you want to install the latest ActiveMatcher version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

To install ActiveMatcher directly from its GitHub repo, use the following command:

```
    pip install git+https://github.com/anhaidgroup/active_matcher.git@main
```

Similar to pip installing from PyPI, the above command will install ActiveMatcher and all of its dependencies.
