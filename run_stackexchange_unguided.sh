#!/bin/bash
#
# Job name
#SBATCH --job-name=DataContam               # TODO: adjust job name

#SBATCH --time=00:30:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:1                 # Request 2 GPUs
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
module load devel/python/3.10.5_gnu_12.1 
module load devel/cuda/11.8

# Activate the conda environment
ENV_NAME="$HOME/Data-Comtamination-Sem-3-project/DataContam"
echo "Activating python environment: $ENV_NAME"

if [ -d "$ENV_NAME" ]; then
    srun source "$ENV_NAME/bin/activate"
    echo "Environment '$ENV_NAME' activated successfully."
else
    echo "Error: Virtual environment '$ENV_NAME' not found."
    exit 1
fi
# Run the Python script
SCRIPT="run.py"

# Set the environment variable to allow PyTorch to allocate more memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SSL_CERT_FILE=$(python -m certifi)

srun python "$SCRIPT" --model "OpenLlama" --task "stackexchange" --type "unguided"

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