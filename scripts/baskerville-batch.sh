#!/bin/bash
#SBATCH --qos turing
#SBATCH --account  vjgo8416-ml-workload
#SBATCH --time 24:00:0
#SBATCH --nodes 1
#SBATCH --gpus 2
#SBATCH --cpus-per-gpu 36
#SBATCH --job-name mingpt-test

# Execute using:
# sbatch -o baskerville-batch-%A_%a.out --array=1-2 ./scripts/baskerville-batch.sh

module purge
module load baskerville

echo MinGPT Batch run
echo ============================================
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}
echo SLURM_JOB_ID: ${SLURM_JOB_ID}
echo SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}
echo SLURM_ARRAY_TASK_COUNT: ${SLURM_ARRAY_TASK_COUNT}
echo SLURM_ARRAY_TASK_MAX: ${SLURM_ARRAY_TASK_MAX}
echo SLURM_ARRAY_TASK_MIN: ${SLURM_ARRAY_TASK_MIN}
echo ============================================

module restore system
module load Python/3.9.5-GCCcore-10.3.0
module load GCC/10.3.0
module load CUDA/11.3.1
module load Miniconda3/4.10.3
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="/bask/homes/o/ovau2564/vjgo8416-ml-workload/pytorch-tests/minGPT/conda_env/minGPT"

pushd /bask/homes/o/ovau2564/vjgo8416-ml-workload/pytorch-tests/minGPT

#echo "Creating conda environment"
if [ -d "$CONDA_ENV_PATH" ]; then
  echo "Conda directory exists, skipping creation"
else
  echo "Creating environment"
  conda create -p "${CONDA_ENV_PATH}"
fi

# Activate the environment
echo "Activating conda environment"
conda activate "${CONDA_ENV_PATH}"

echo "Check Python version and location"
python3 --version
which python3

echo "Installing requirements"
python3 -m pip install --upgrade torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install -r requirements.txt

echo
echo "######################################"
echo "Batch configuration"
echo "######################################"
echo

CONFIG_LAYERS=2
CONFIG_HEAD=16
CONFIG_EMBD=$((1024*${SLURM_ARRAY_TASK_ID}))
CONFIG_PRECISION=16
CONFIG_BATCH_SIZE=1
CONFIG_NUM_WORKERS=36
CONFIG_GPUS=2
CONFIG_STRATEGY=deepspeed_stage_3_offload

echo "Batch number: ${SLURM_ARRAY_TASK_ID}" 
echo "n_layer: ${CONFIG_LAYERS}"
echo "n_head: ${CONFIG_HEAD}"
echo "n_embd: ${CONFIG_EMBD}"
echo "precision: ${CONFIG_PRECISION}"
echo "batch_size: ${CONFIG_BATCH_SIZE}"
echo "num_workers: ${CONFIG_NUM_WORKERS}"
echo "strategy: ${CONFIG_STRATEGY}"

echo
echo "######################################"
echo "Starting"
echo "######################################"
echo

python3 train.py \
	--n_layer ${CONFIG_LAYERS} \
	--n_head ${CONFIG_HEAD} \
	--n_embd ${CONFIG_EMBD} \
	--gpus ${CONFIG_GPUS}\
	--precision ${CONFIG_PRECISION} \
	--batch_size ${CONFIG_BATCH_SIZE} \
	--num_workers ${CONFIG_NUM_WORKERS} \
	--strategy ${CONFIG_STRATEGY} 
echo
echo "######################################"
echo "Done"
echo "######################################"
echo

conda deactivate
popd

# To fully reset:
# rm -rf /bask/homes/o/ovau2564/vjgo8416-ml-workload/pytorch-tests/minGPT/conda_env
# rm -rf /bask/homes/o/ovau2564/.local
# rm -rf /bask/homes/o/ovau2564/vjgo8416-ml-workload/.conda

