# How to Install ActiveMatcher on a Linux Machine

This is a step-by-step guide to install ActiveMatcher and its necessary dependencies on a single Linux machine with Ubunutu 22.0.4

he following software versions were installed on the test machine using the steps in this guide: Python 3.12. You can try to adapt this guide to other configurations.

### **Step 1: Install Essential Packages**

This section deals with installing and updating packages that are neccessary for us to install Java and Python and later steps. We will install build-essential, libbz2-dev, libssl-dev, libreadline-dev, libsqlite3-dev, libffi-dev, zlib1g-dev, libncurses5-dev, libncursesw5-dev, xz-utils, tk-dev, and wget.

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

Once these are installed, we can install Java, Python, and ActiveMatcher.

### **Step 2: Java Installation**

We strongly recommend installing Java Temurin JDK 17, which is a specific Java release that we have extensively experimented with. As that is not available from the Ubuntu package repository, to install Java, you will need to use the following commands:

```
sudo -s
apt install wget apt-transport-https gnupg
wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release)     main" | tee /etc/apt/sources.list.d/adoptium.list
apt update
apt install temurin-17-jdk
exit
```

You can check that you have successfully installed Java by running this command. If Java is installed, it should display a version number.

```
java --version
```

### **Step 3: Python Installation**

This section deals with installing Python 3.12, creating a virtual environment, installing two Python packages (setuptools and build). Other versions of Python, other environments, or incorrect installations of the setuptools and build packages can cause issues with ActiveMatcher installation.

If you suspect that you may have Python downloaded on your machine already, open up your terminal. Then, run the command:

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

where x is a number, you can go to Step 3C after completing Step 3A (you do not need to complete Step 3B).

If

which python3

Or

python3 --version

If you did not have the outputs listed above, continue to step 3B. Otherwise, skip to 3C.

#### **Step 3B: Python Installation**

To Install Python, we will first download Python 3.12 from Python's website, then we will install it, and finally we will make it the default verison.
Run the following in the terminal to install Python 3.12:

##### **Step 3B.1: Download Python**

```
    cd /usr/src
    sudo curl -O https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
    sudo tar xzf Python-3.12.3.tgz
```

##### **Step 3B.2: Install Python**

```
    cd Python-3.12.3
    sudo make clean
    sudo ./configure --enable-optimizations --with-system-ffi
    sudo make -j$(nproc)
    sudo make altinstall
```

##### **Step 3B.3: Make Python 3.12 the Default**
```
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1
python3.12 -m ensurepip --default-pip
python3.12 -m pip install --upgrade pip setuptools
```
#### **Step 3C: Python Environment Setup**

Now, we will create a Python environment with Python 3.12. This step is necessary to make sure we use the correct version of Python with the correct dependencies. First, we need to install the venv module with the following command:
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

#### **Step 3D: Python Package Installation**

We will be downloading two packages: setuptools and build. Before installing, make sure you are in the virtual environment. If you have just finished Step 1C, you are in the virtual environment. Otherwise, to make sure your virtual environment is active, you can run:
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
### **Step 4: ActiveMatcher Installation**

Now that you have the correct version of Python installed, we can download ActiveMatcher. Before we complete the download, we should return to the root directory. We can achieve this by running the following command in the terminal:
```
    cd
```
Now, to download ActiveMatcher, use one of the following options:

#### **Option 1: Pip Installing from PyPI**

You can install ActiveMatcher from PyPI, using the following command:
```
    pip install active_matcher
```
This command will install ActiveMatcher and all of its dependencies, such as Joblib, mmh3, Numba, Numpy, Numpydoc, Pandas, Py_Stringmatching, PySpark, Scikit-Learn, Scipy, Threadpoolctl, TQDM, and Xgboost.

#### **Option 2: Pip Installing from GitHub**

Instead of pip installing from PyPI, you may want to pip install ActiveMatcher from its GitHub repo. This happens if you want to install the latest ActiveMatcher version compared to the version on PyPI. For example, the GitHub version may contain bug fixes that the PyPI version does not.

To install ActiveMatcher directly from its GitHub repo, use the following command:
```
    pip install git+https://github.com/anhaidgroup/active_matcher.git@main
```
Similar to pip installing from PyPI, the above command will install ActiveMatcher and all of its dependencies.
