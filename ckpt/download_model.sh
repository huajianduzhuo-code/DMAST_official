#!/bin/bash

mkdir -p "$HOME/dmast_ckpt/pretrained"
mkdir -p "$HOME/bucket/ckpt/pretrained"

for model_name in google/gemma-3-27b-it \
    google/gemma-3-12b-it \
    Qwen/Qwen3-VL-8B-Instruct \
    Qwen/Qwen3-VL-32B-Instruct \
    # Qwen/Qwen3-VL-30B-A3B-Instruct
    # google/gemma-3-12b-it \
    # google/gemma-3-4b-it \
    # Qwen/Qwen3-4B-Base \
    # Qwen/Qwen3-8B-Base \
    # Qwen/Qwen3-14B-Base
do
    model_dir=$(basename "$model_name" | tr '[:upper:]' '[:lower:]')
    model_dir="${model_dir//-base/-pt}"

    echo "Downloading $model_name to $model_dir"

    # skip if download is already done
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$model_name" --local-dir "$HOME/dmast_ckpt/pretrained/$model_dir"

    # skip copying if already done
    # if [ ! -d "$HOME/bucket/ckpt/pretrained/$model_dir" ]; then
    #     echo "Copying $model_dir to bucket"
    #     cp -r "$HOME/dmast_ckpt/pretrained/$model_dir" "$HOME/bucket/ckpt/pretrained/$model_dir"
    # fi
done