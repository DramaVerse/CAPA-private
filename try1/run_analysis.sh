#!/bin/bash

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install pandas matplotlib seaborn numpy
else
    source venv/bin/activate
fi

# 运行分析脚本
python poetry_analysis.py

# 如果生成了报告，打印报告内容
if [ -f "analysis_output/analysis_report.md" ]; then
    echo "分析报告已生成，内容如下："
    echo "========================="
    cat analysis_output/analysis_report.md
    echo "========================="
    echo "可视化结果保存在 analysis_output 目录中"
fi

# 退出虚拟环境
deactivate 