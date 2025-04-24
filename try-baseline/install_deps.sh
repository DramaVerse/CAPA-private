#!/bin/bash
# 安装运行脚本所需的依赖

echo "安装必要的Python包..."
pip install psutil tqdm modelscope transformers==4.36.2 accelerate==0.25.0 torch==2.1.2 sentencepiece==0.1.99 protobuf==4.24.4

echo "安装完成！"
