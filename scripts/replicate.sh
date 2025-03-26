#!/bin/bash
#
# Job name
#SBATCH --job-name=replication              # TODO: adjust job name

#SBATCH --time=00:30:00              # Job time limit (30 minutes)
#SBATCH --ntasks=1                   # Total number of tasks
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --partition=dev_gpu_4
#SBATCH --mem=32GB 

# Output and error logs
#SBATCH --output="replication_out.txt"        # TODO: adjust standard output log
#SBATCH --error="replication_err.txt"         # TODO: adjust error log

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

ENV_NAME="$HOME/Data-Comtamination-Sem-3-project/DataContamEval"


if [ -d "$ENV_NAME" ]; then
    source "$ENV_NAME/bin/activate"
    echo "Environment '$ENV_NAME' activated successfully."
else
    echo "Error: Virtual environment '$ENV_NAME' not found."
    exit 1

python "$HOME/Data-Comtamination-Sem-3-project/replication_study.py"

echo "Deactivating environment: $ENV_NAME"
deactivate

COLUMNS="JobID,JobName,MaxRSS,NTasks,AllocCPUS,AllocGRES,AveDiskRead,AveDiskWrite,Elapsed,State"
sacct -l -j $SLURM_JOB_ID --format=$COLUMNS