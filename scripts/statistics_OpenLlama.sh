#!/bin/bash
#
# Job name
#SBATCH --job-name=stats_OpenLlama             # TODO: adjust job name

#SBATCH --time=00:30:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --partition=dev_gpu_4
#SBATCH --mem=16GB 

# Output and error logs
#SBATCH --output="stats_OpenLlama_out.txt"        # TODO: adjust standard output log
#SBATCH --error="stats_OpenLlama_err.txt"         # TODO: adjust error log

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# initialize shell to work with bash
source ~/.bashrc
# load the necessary modules
module load devel/miniconda/23.9.0-py3.9.15
module load devel/cuda/11.8
module load devel/python/3.12.3_gnu_13.3

# CHANGE THESE VARIABLES FOR DIFFERENT MODELS AND TASKS
MODELS=("OpenLlama")
TASKS=("cb", "stackexchange", "wsc")


ENV_NAME="$HOME/Data-Comtamination-Sem-3-project/DataContam"
SCRIPT="$HOME/Data-Comtamination-Sem-3-project/Statistics.py"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SSL_CERT_FILE=$($ENV_NAME/bin/python -m certifi)
export TF_CPP_MIN_LOG_LEVEL=2

# Run first script with DataContam env
echo "Activating python environment: $ENV_NAME"

if [ -d "$ENV_NAME" ]; then
    source "$ENV_NAME/bin/activate"
    echo "Environment '$ENV_NAME' activated successfully."
else
    echo "Error: Virtual environment '$ENV_NAME' not found."
    exit 1
fi


for model in "${MODELS[@]}"; do       
    for task in "${TASKS[@]}"; do
        echo "Running task: $task"
        python "$SCRIPT" --diffs "$HOME/Data-Comtamination-Sem-3-project/results/${task}_${model}_differences.csv" --icl "$HOME/Data-Comtamination-Sem-3-project/results/${task}_${model}_guided_prompting.csv"

        # Verify if the script executed successfully
        if [ $? -eq 0 ]; then
            echo "Python script '$SCRIPT' executed successfully for task=$task."
        else
            echo "Error: Python script '$SCRIPT' failed for task=$task."
            exit 1
        fi
    done
done

echo "Deactivating environment: $ENV_NAME"
deactivate

COLUMNS="JobID,JobName,MaxRSS,NTasks,AllocCPUS,AllocGRES,AveDiskRead,AveDiskWrite,Elapsed,State"
sacct -l -j $SLURM_JOB_ID --format=$COLUMNS