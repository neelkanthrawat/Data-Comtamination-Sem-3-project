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

### Environment for generating

```shell
module devel/python/3.12.3_gnu_13.3
```

```shell
python -m venv DataContam
source DataContam/bin/activate
```

Check that python and pip are pointing to the correct locations:

```shell
which python
which pip
```

```shell
python -m pip install transformers torch pandas datasets 'accelerate>=0.26.0' sentencepiece
```

### Environment for evaluating

```shell
python -m venv DataContamEval
source DataContamEval/bin/activate
```

Check that python and pip are pointing to the correct locations:

```shell
which python
which pip
```

```shell
python -m pip install pandas evaluate rouge_score git+https://github.com/google-research/bleurt.git 'accelerate>=0.26.0' 'requests==2.32.3' 'fsspec>=2023.1.0,<=2024.12.0' tensorflow[and-cuda]
python -m pip install pandas evaluate rouge_score git+https://github.com/google-research/bleurt.git accelerate requests fsspec tensorflow[and-cuda]

```
