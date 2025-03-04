# Data-Comtamination-Sem-3-project

## Create the conda environment

Load the modules that are needed to create the environment:

```shell
module load devel/miniconda/23.9.0-py3.9.15
```

Create a conda environment, activate it and install the necessary packages:

```shell
conda create -n DataContam
conda activate DataContam
pip install transformers torch pandas evaluate datasets rouge_score git+https://github.com/google-research/bleurt.git 'accelerate>=0.26.0'

conda install -c conda-forge ca-certificates
conda install -c conda-forge certifi
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -m certifi)
```

## Create an environment

```shell
module load devel/python/3.10.5_gnu_12.1
```

```shell
python -m venv DataContam
source DataContam/bin/activate
python -m pip install transformers torch pandas evaluate datasets rouge_score git+https://github.com/google-research/bleurt.git 'accelerate>=0.26.0'
python -m pip install --upgrade certifi
```
