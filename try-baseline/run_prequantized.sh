#!/bin/bash
# 使用预量化模型运行

echo "安装必要的依赖..."
pip install modelscope transformers accelerate tqdm psutil

echo "使用预量化的Qwen2.5-14B-Instruct-GPTQ-Int8模型..."
python run_model_prequantized.py --samples 5 --max_tokens 512

echo "运行完成！"
