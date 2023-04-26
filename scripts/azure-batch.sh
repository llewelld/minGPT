# Script for running a single minGPT run

# SLURM_ARRAY_JOB_ID should be set to a number 1, 2, 3, ... to control the batch variables

echo MinGPT Batch run
echo ============================================
echo SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}
echo ============================================
echo

re='^[0-9]+$'
if ! [[ "${SLURM_ARRAY_TASK_ID}" =~ $re ]]; then
  echo "Please ensure SLURM_ARRAY_TASK_ID is set to an integer value"
  exit 1
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('${HOME}/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${HOME}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export EBROOTMINICONDA3=${HOME}/miniconda3
CONDA_ENV_PATH="${HOME}/minGPT/conda_env/minGPT"

pushd ${HOME}/minGPT

# Ensure conda is installed
if [ -d "$EBROOTMINICONDA3" ]; then
  echo "Conda is already installed"
else
  echo "Conda is not installed, installing."
  echo "Following process from: "
  echo "https://repo.anaconda.com/minicond/"
  wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
  SHA256="1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f"
  if echo "$SHA256 Miniconda3-py39_4.10.3-Linux-x86_64.sh" | sha256sum --check --status; then
    echo "Miniconda3 downloaded successfully. Installling.";
  else
    echo "Downloadd Miniconda3 failed sha256 check. Exiting."
    exit 1
  fi
  chmod 744 Miniconda3-py39_4.10.3-Linux-x86_64.sh
  ./Miniconda3-py39_4.10.3-Linux-x86_64.sh -b
fi

# Ensure CUDA 11.3 is installed
if dpkg-query --show cuda-11-3 > /dev/null; then
  echo "CUDA 11.3.0 is already installed."
else
  echo "CUDA 11.3.0 is not installed. Installing."
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
  sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
  sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
  sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
  sudo apt-get update
  sudo apt install -y cuda-11-3
fi

echo "Creating conda environment"
if [ -d "$CONDA_ENV_PATH" ]; then
  echo "Conda directory exists, skipping creation"
else
  echo "Creating environment"
  conda create -y -p "${CONDA_ENV_PATH}" python=3.9.5
fi

echo "Activating conda environment"
conda activate "${CONDA_ENV_PATH}"

echo "Check Python version and location"
python3 --version
which python3

echo "Ensure we have pip installed"
conda install -y pip

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
CONFIG_GPUS=0
CONFIG_STRATEGY=ddp

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

