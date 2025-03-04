#!/bin/bash
#
# Job name
#SBATCH --job-name=DataContam               # TODO: adjust job name

#SBATCH --time=00:30:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:2                 # Request 2 GPUs
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --partition=dev_gpu_4

# Output and error logs
#SBATCH --output="DataContam_out.txt"        # TODO: adjust standard output log
#SBATCH --error="DataContam_err.txt"         # TODO: adjust error log

# Email notifications
#SBATCH --mail-user=""
#SBATCH --mail-type=START,END,FAIL  # Send email when the job ends or fails

### JOB STEPS START HERE ###
# initialize shell to work with bash
source ~/.bashrc
# load the necessary modules
module load devel/miniconda/23.9.0-py3.9.15
module load devel/cuda/11.8

# Verify conda availability
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not available after loading the module."
    exit 1
else
    echo "Conda is available."
fi

# Activate the conda environment
ENV_NAME="DataContam"
echo "Activating conda environment: $ENV_NAME"
if ! conda activate "$ENV_NAME"; then
    echo "Error: Failed to activate conda environment '$ENV_NAME'."
    exit 1
else
    echo "Conda environment '$ENV_NAME' activated successfully."
fi

# Run the Python script
SCRIPT="run.py"

# Set the environment variable to allow PyTorch to allocate more memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun python3 "$SCRIPT" --model "OpenLlama" --task "wsc" --type "guided"

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

echo "Deactivating conda environment: $ENV_NAME"
conda deactivate