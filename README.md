# Data-Comtamination-Sem-3-project

## Create the environment

Load the modules that are needed to create the environment:

```shell
module load devel/python/3.10.0_gnu_11.1
module load devel/miniconda/23.9.0-py3.9.15
```

Create a conda environment, activate it and install the necessary packages:

```shell
conda create -n DataContam
conda activate DataContam
pip3 install transformers torch pandas evaluate datasets rouge_score git+https://github.com/google-research/bleurt.git 'accelerate>=0.26.0'
```
