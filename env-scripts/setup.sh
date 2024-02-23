#set pytorch=1.10 because newer version do not have thc.h
#cudatoolkit-dev provides cuda-runtime.h
# /user/jschnei2/miniconda3/envs/pointmae/include/thrust/system/cuda/config.h last check commented out

#conda remove --name pointmae --all -y
conda env create -f env-scripts/envtest.yml
rm  /tmp/cuda-installer.log
source activate rd-mae
conda activate rd-mae
# optional for data folder sharing
# ln -s ../data data


