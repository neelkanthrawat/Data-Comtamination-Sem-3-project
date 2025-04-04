#!/bin/bash
#
# Job name
#SBATCH --job-name=eval_openllama               # TODO: adjust job name

#SBATCH --time=10:00:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --partition=gpu_4
#SBATCH --mem=138GB 

# Output and error logs
#SBATCH --output="eval_openllama_out.txt"        # TODO: adjust standard output log
#SBATCH --error="eval_openllama_err.txt"         # TODO: adjust error log

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# initialize shell to work with bash
source ~/.bashrc
# load the necessary modules
module load devel/python/3.12.3_gnu_13.3
module load devel/cuda/12.4 
module load devel/cudnn/10.2

# ADJUST THESE VARIABLES TO INCLUDE EVERYTHING WE WANT TO RUN
MODELS=("OpenLlama")
TASKS=("cb" "wsc")
TYPES=("guided" "unguided")

# Activate the conda environment
ENV_NAME="$HOME/Data-Comtamination-Sem-3-project/DataContam"
ENV_NAME2="$HOME/Data-Comtamination-Sem-3-project/DataContamEval"


# Set the environment variable to allow PyTorch to allocate more memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SSL_CERT_FILE=$($ENV_NAME/bin/python -m certifi)
export SSL_CERT_FILE=$($ENV_NAME2/bin/python -m certifi)
export TF_CPP_MIN_LOG_LEVEL=2

if [ -d "$ENV_NAME2" ]; then
    source "$ENV_NAME2/bin/activate"
    echo "Environment '$ENV_NAME2' activated successfully."
else
    echo "Error: Virtual environment '$ENV_NAME2' not found."
    exit 1
fi

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running model: $model, task: $task"
        python "$HOME/Data-Comtamination-Sem-3-project/eval.py" --guided "$HOME/Data-Comtamination-Sem-3-project/results/${model}/${task}/${task}_${model}_guided.csv" --unguided "$HOME/Data-Comtamination-Sem-3-project/results/${model}/${task}/${task}_${model}_unguided.csv" --name "${task}_${model}"

        # Verify if the script executed successfully
        if [ $? -eq 0 ]; then
            echo "Python script eval.py executed successfully for model=$model, task=$task"
        else
            echo "Error: Python script eval.py failed for model=$model, task=$task"
            exit 1
        fi
    done
done

echo "Deactivating environment: $ENV_NAME2"
deactivate


COLUMNS="JobID,JobName,MaxRSS,NTasks,AllocCPUS,AllocGRES,AveDiskRead,AveDiskWrite,Elapsed,State"
sacct -l -j $SLURM_JOB_ID --format=$COLUMNS
