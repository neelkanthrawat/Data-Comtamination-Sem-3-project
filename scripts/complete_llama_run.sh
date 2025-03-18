#!/bin/bash
#
# Job name
#SBATCH --job-name=complete_llama               # TODO: adjust job name

#SBATCH --time=06:00:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --partition=gpu_4
#SBATCH --mem=128GB 

# Output and error logs
#SBATCH --output="complete_llama_out.txt"        # TODO: adjust standard output log
#SBATCH --error="complete_llama_err.txt"         # TODO: adjust error log

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
MODELS=("Llama")
TASKS=("ag_news" "imdb")
TYPES=("guided" "unguided")

# Activate the conda environment
ENV_NAME="$HOME/Data-Comtamination-Sem-3-project/DataContam"
echo "Activating python environment: $ENV_NAME"
ENV_NAME2="$HOME/Data-Comtamination-Sem-3-project/DataContam"


if [ -d "$ENV_NAME" ]; then
    source "$ENV_NAME/bin/activate"
    echo "Environment '$ENV_NAME' activated successfully."
else
    echo "Error: Virtual environment '$ENV_NAME' not found."
    exit 1
fi
# Run the Python script
SCRIPT="$HOME/Data-Comtamination-Sem-3-project/run.py"

# Set the environment variable to allow PyTorch to allocate more memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SSL_CERT_FILE=$($ENV_NAME/bin/python -m certifi)
export SSL_CERT_FILE=$($ENV_NAME2/bin/python -m certifi)
export TF_CPP_MIN_LOG_LEVEL=2

# Loop through all combinations of models, tasks, and types
for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for type in "${TYPES[@]}"; do
            echo "Running model: $model, task: $task, type: $type"
            python "$SCRIPT" --model "$model" --task "$task" --type "$type"
            
            # Verify if the script executed successfully
            if [ $? -eq 0 ]; then
                echo "Python script '$SCRIPT' executed successfully for model=$model, task=$task, type=$type."
            else
                echo "Error: Python script '$SCRIPT' failed for model=$model, task=$task, type=$type."
                exit 1
            fi
        done
    done
done

SCRIPT="$HOME/Data-Comtamination-Sem-3-project/eval.py"
SCRIPT2="$HOME/Data-Comtamination-Sem-3-project/ICL.py"

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running model: $model, task: $task"
        python "$SCRIPT" --guided "$HOME/Data-Comtamination-Sem-3-project/results/${task}_${model}_guided.csv" --unguided "$HOME/Data-Comtamination-Sem-3-project/results/${task}_${model}_unguided.csv" --name "${task}_${model}"

        # Verify if the script executed successfully
        if [ $? -eq 0 ]; then
            echo "Python script '$SCRIPT' executed successfully for model=$model, task=$task, type=$type."
        else
            echo "Error: Python script '$SCRIPT' failed for model=$model, task=$task, type=$type."
            exit 1
        fi
    done
done

echo "Deactivating environment: $ENV_NAME"
deactivate

# Run second script with DataContamEval env
echo "Activating python environment: $ENV_NAME2"

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
        python "$SCRIPT2" --guided "$HOME/Data-Comtamination-Sem-3-project/results/${task}_${model}_guided.csv" --name "${task}_${model}"

        # Verify if the script executed successfully
        if [ $? -eq 0 ]; then
            echo "Python script '$SCRIPT2' executed successfully for model=$model, task=$task, type=$type."
        else
            echo "Error: Python script '$SCRIPT2' failed for model=$model, task=$task, type=$type."
            exit 1
        fi
    done
done


ENV_NAME="$HOME/Data-Comtamination-Sem-3-project/DataContam"
SCRIPT="$HOME/Data-Comtamination-Sem-3-project/Statistics.py"

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