#!/bin/bash
# 使用直接量化的方式运行模型

echo "安装必要的依赖..."
pip install bitsandbytes transformers accelerate tqdm psutil

echo "使用8位量化运行Qwen-7B-Chat模型..."
python run_model_quantized.py --bits 8 --samples 5 --max_tokens 512

echo "运行完成！"
