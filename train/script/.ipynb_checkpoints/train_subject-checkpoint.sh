# Specify the config file path and the GPU devices to use
# export CUDA_VISIBLE_DEVICES=0,1

# Specify the config file path
export XFL_CONFIG=./train/config/subject_512.yaml

# Specify the WANDB API key
export WANDB_API_KEY='f35df6ae07ed38429412d04eb764046f132a3497'

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41353 -m src.train.train