# 古诗词理解与推理评测 Baseline

本项目提供了在阿里云PAI-DSW上运行Qwen2.5-7B模型的baseline，用于古诗词理解与推理评测任务。

## 快速开始

1. 查看详细指南：`pai_dsw_guide.md`
2. 使用Transformers运行：`python run_transformers.py`
3. 使用vLLM运行（可选）：`python run_vllm.py`
4. 使用优化提示词运行：`python run_optimized.py`
5. 评估结果：`python evaluate.py`

## 文件说明

- `pai_dsw_guide.md`: 详细的操作指南
- `run_transformers.py`: 使用Transformers运行Qwen2.5-7B的脚本
- `run_vllm.py`: 使用vLLM加速运行Qwen2.5-7B的脚本
- `run_optimized.py`: 使用优化提示词运行Qwen2.5-7B的脚本
- `evaluate.py`: 评估模型输出结果的脚本
- `test_data.json`: 测试数据
- `baseline_output.json`: 基础模型输出结果
- `vllm_output.json`: vLLM模型输出结果
- `optimized_output.json`: 优化提示词后的模型输出结果

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36+
- CUDA 11.8+
- 24GB+ GPU显存（NVIDIA A10）

## 参考资料

- [Qwen2.5-7B模型](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [vLLM文档](https://docs.vllm.ai/)
- [Transformers文档](https://huggingface.co/docs/transformers/)
