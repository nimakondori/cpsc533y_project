#!/bin/bash
# CUDA version from the command-line argument

cuda=${1:-"0"}
filename=${2:-"run.py"}
config_filename=${3:-"default.yml"}
save_dir=${4:-"fixed_video_size"}

export CUDA_VISIBLE_DEVICES=$cuda

command="python $filename --config_path configs/$config_filename --save_dir ./logs/$save_dir"
echo "running the following command" $command

# Run the command
$command

