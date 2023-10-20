#!/bin/bash

if [[ "${#}" == 1 ]]; then
	REQUIREMENTS="${1}"
else
	REQUIREMENTS="./requirements.txt"
fi

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}


if ! find_in_conda_env "calibrated-posterior"; then
  echo "Creating conda env"
  conda create -y --name calibrated-posterior python=3.9
  conda activate calibrated-posterior
  unset -v CUDA_PATH
  conda install -y -c "nvidia/label/cuda-11.7.1" cuda
  conda install -y -c conda-forge gxx_linux-64=11.4.0
  conda install -y pip
  pip install --upgrade pip
  pip install wheel==0.38.4
  pip install $(grep 'torch==' $REQUIREMENTS) $(grep 'matplotlib==' $REQUIREMENTS) --extra-index-url https://download.pytorch.org/whl/cu117
  TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing;Ampere" pip install $(grep 'torchsort' $REQUIREMENTS)
  git clone https://github.com/montefiore-ai/hypothesis.git ./src/conda/hypothesis
  RETURNDIR=$PWD
  cd ./src/conda/hypothesis
  git checkout 0.4.0
  pip install -e .
  cd $RETURNDIR
fi

conda activate calibrated-posterior

./conda/install-deps.sh $REQUIREMENTS
