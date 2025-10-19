# Evaluation of Graph Embedding Methods on Real-World Datasets
The first step is to install **Miniconda** and create a virtual enviroment for accomodating the whole project.
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # get the latest version for Linux
bash Miniconda3-latest-Linux-x86_64.sh # install miniconda
conda --version # check the correctness of the installation
conda create --name myenv python=3.9 # create an a venv with name my env and python version 3.9
conda activate myenv # activate conda enviroment to currently work on
conda install numpy pandas matplotlib # install some basic python libraries
conda deactivate myenv # exit the myenv enviroment
```
Then in order to create an enviroment with the necessary python modules we created an environment.yml file, we executed:
``` bash
# 1) Create the env
conda env create -f environment.yml
conda activate graphbench
# 2) (Optional) upgrade pip
python -m pip install -U pip
# 3) If an error occured due to mismatch of sympy please do this
pip install "sympy==1.13.1"
# 4) Install PyG + companions for cpu only
# set the variable automatically
TV=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install -U fsspec torch-geometric
pip install -U pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-${TV}+cpu.html

