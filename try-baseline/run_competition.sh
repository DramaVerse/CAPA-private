#!/bin/bash
# 使用预量化模型运行竞赛版本

echo "安装必要的依赖..."
pip install modelscope transformers accelerate tqdm psutil

echo "使用预量化的Qwen2.5-14B-Instruct-GPTQ-Int8模型进行竞赛任务..."
python run_model_competition.py --samples 5 --max_tokens 512

echo "运行完成！"
