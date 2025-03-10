#!/bin/bash
#
# Job name
#SBATCH --job-name=eval_wsc               # TODO: adjust job name

#SBATCH --time=00:30:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --partition=dev_gpu_4
#SBATCH --mem=16GB 

# Output and error logs
#SBATCH --output="eval_wsc_out.txt"        # TODO: adjust standard output log
#SBATCH --error="eval_wsc_err.txt"         # TODO: adjust error log

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# initialize shell to work with bash
source ~/.bashrc
# load the necessary modules
module load devel/miniconda/23.9.0-py3.9.15
module load devel/cuda/11.8

ENV_NAME="$HOME/Data-Comtamination-Sem-3-project/DataContamEval"
echo "Activating python environment: $ENV_NAME"

if [ -d "$ENV_NAME" ]; then
    source "$ENV_NAME/bin/activate"
    echo "Environment '$ENV_NAME' activated successfully."
else
    echo "Error: Virtual environment '$ENV_NAME' not found."
    exit 1
fi

# Run the Python script
SCRIPT="eval.py"

# Set the environment variable to allow PyTorch to allocate more memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python3 "$SCRIPT" --guided "./results/wsc_OpenLlama_guided" --unguided "./results/wsc_OpenLlama_unguided" --name "wsc_OpenLlama"

# Verify if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script '$SCRIPT' executed successfully."
else
    echo "Error: Python script '$SCRIPT' failed."
    exit 1
fi

echo "Job completed successfully."

COLUMNS="JobID,JobName,MaxRSS,NTasks,AllocCPUS,AllocGRES,AveDiskRead,AveDiskWrite,Elapsed,State"
sacct -l -j $SLURM_JOB_ID --format=$COLUMNS

echo "Deactivating environment: $ENV_NAME"
deactivate