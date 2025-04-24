#!/bin/bash
# 使用预量化的7B模型运行

echo "安装必要的依赖..."
pip install modelscope transformers accelerate tqdm psutil

echo "使用预量化的Qwen2.5-7B-Instruct-MLX-8bit模型..."
python run_model_prequantized.py --model "lmstudio-community/Qwen2.5-7B-Instruct-MLX-8bit" --samples 5 --max_tokens 512

echo "运行完成！"
