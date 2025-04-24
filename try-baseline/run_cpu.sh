#!/bin/bash
# 使用CPU运行模型

echo "安装必要的依赖..."
pip install transformers accelerate tqdm psutil

echo "使用CPU运行Qwen-7B-Chat模型..."
python run_model_quantized.py --cpu --samples 2 --max_tokens 256

echo "运行完成！"
