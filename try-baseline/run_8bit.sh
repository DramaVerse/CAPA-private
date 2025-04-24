#!/bin/bash
# 使用8位量化运行模型

echo "使用8位量化模式运行Qwen-7B-Chat模型..."
python run_model_aliyun.py --use_8bit --debug --samples 5

echo "运行完成！"
