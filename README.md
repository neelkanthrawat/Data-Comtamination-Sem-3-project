# Data-Comtamination-Sem-3-project

## Create the environment

Load the modules that are needed to create the environment:

```shell
module load devel/miniconda/23.9.0-py3.9.15
```

Create a conda environment, activate it and install the necessary packages:

```shell
conda create -n DataContam
conda activate DataContam
pip install transformers torch pandas evaluate datasets rouge_score git+https://github.com/google-research/bleurt.git 'accelerate>=0.26.0'

conda install -c conda-forge certificates
conda install -c conda-forge certifi
pip install --upgrade certifi
```
