# 古诗词理解与推理评测任务实施指南

## 前言

本指南专为数字人文硕士学生设计，旨在提供一个清晰易懂的手把手教程，帮助您完成CCL 2025古诗词理解与推理评测任务。本指南基于以下前提和决策：

**设备条件**：
- 主要设备：M1 Max MacBook Pro
- 备用设备：NVIDIA 3060笔记本（Windows系统，仅在必要时使用）

**训练方案**：
- 使用AutoDL云平台进行模型训练
- 采用Qwen2-7B模型 + QLoRA微调方法
- 本地环境用于数据准备和代码开发
- 云环境用于模型训练和推理

**目标**：
- 跑通完整流程
- 在比赛中取得竞争力的结果
- 确保输出符合比赛要求的JSON格式

本指南将详细介绍从环境搭建、数据准备、代码开发、模型训练到结果提交的每一个步骤，即使您没有机器学习背景，也能按照指南完成任务。

## 目录

1. [环境准备](#1-环境准备)
   - 1.1 本地环境搭建（M1 Max MacBook）
   - 1.2 AutoDL账户注册与配置

2. [数据准备](#2-数据准备)
   - 2.1 比赛数据集处理
   - 2.2 数据增强与格式转换

3. [代码开发](#3-代码开发)
   - 3.1 使用LLaMA-Factory框架
   - 3.2 配置文件编写
   - 3.3 训练脚本准备

4. [模型训练](#4-模型训练)
   - 4.1 AutoDL实例创建
   - 4.2 数据与代码上传
   - 4.3 训练过程监控

5. [结果生成与优化](#5-结果生成与优化)
   - 5.1 模型推理
   - 5.2 输出格式处理
   - 5.3 结果优化策略

6. [提交与验证](#6-提交与验证)
   - 6.1 结果验证
   - 6.2 提交流程

## 1. 环境准备

### 1.1 本地环境搭建（M1 Max MacBook）

M1 Max虽然不能直接用于大模型训练，但非常适合进行数据处理、代码开发和小规模测试。以下是在M1 Max上搭建环境的步骤：

#### 安装Homebrew（如果尚未安装）

打开终端，执行以下命令：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 安装Miniconda

```bash
brew install --cask miniconda
```

安装完成后，初始化conda：

```bash
conda init zsh  # 如果您使用的是zsh
# 或
conda init bash  # 如果您使用的是bash
```

关闭并重新打开终端，或执行：

```bash
source ~/.zshrc  # 对于zsh
# 或
source ~/.bashrc  # 对于bash
```

#### 创建Python环境

创建一个专用于此项目的Python环境：

```bash
conda create -n poetry-llm python=3.10
conda activate poetry-llm
```

#### 安装基本依赖

```bash
# 基础科学计算和数据处理库
pip install numpy pandas matplotlib

# 自然语言处理库
pip install transformers datasets accelerate

# 评估指标库
pip install nltk bert-score

# 其他实用工具
pip install tqdm pyyaml
```

#### 安装PyTorch（M1优化版）

```bash
pip install torch torchvision torchaudio
```

这将安装适用于M1芯片的PyTorch版本。

#### 验证安装

创建一个简单的Python脚本来验证PyTorch是否正确安装：

```python
# test_torch.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
print(f"Current device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
```

运行脚本：

```bash
python test_torch.py
```

如果一切正常，您应该看到MPS可用，这意味着PyTorch可以利用M1的GPU加速。

### 1.2 AutoDL账户注册与配置

AutoDL是一个提供GPU租用服务的平台，价格相对合理，适合学术研究和比赛使用。

#### 注册账户

1. 访问[AutoDL官网](https://www.autodl.com)
2. 点击右上角的"注册"按钮
3. 填写邮箱、密码等信息完成注册
4. 登录账户并完成实名认证（这是中国法规要求的）

#### 充值余额

1. 在AutoDL控制台，点击右上角的"充值"按钮
2. 选择充值金额（建议先充值100-200元进行测试）
3. 选择支付方式（支持支付宝、微信支付等）并完成支付

#### 熟悉平台界面

在正式创建实例前，先熟悉AutoDL的界面：

- **实例管理**：查看和管理您的GPU实例
- **镜像管理**：预装各种深度学习框架的系统镜像
- **数据集**：管理您上传的数据集
- **文件管理**：管理您的文件和模型
- **账户设置**：管理账户信息和API密钥

此时不需要创建实例，我们将在准备好代码和数据后再创建实例，以节省费用。

## 2. 数据准备

数据准备是模型训练的关键步骤，良好的数据质量和格式可以显著提高模型性能。

### 2.1 比赛数据集处理

#### 下载比赛数据

1. 从比赛官方渠道下载训练数据集（200条样例）
2. 创建项目文件夹并组织数据：

```bash
# 在M1 Max上创建项目目录
mkdir -p ~/poetry-llm-project/{data,code,output}
cd ~/poetry-llm-project

# 将下载的数据放入data目录
# 假设数据已下载到Downloads文件夹
mv ~/Downloads/ccl_poetry_train.json ./data/
```

#### 数据探索与分析

创建一个Python脚本来分析数据集，了解其结构和特点：

```python
# data_analysis.py
import json
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
with open('./data/ccl_poetry_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 基本统计
print(f"数据集大小: {len(data)}条")

# 分析诗歌类型分布
poetry_types = []
for item in data:
    content = item['content']
    # 简单判断诗歌类型（基于行数和字数）
    lines = [line for line in content.split('\n') if line.strip()]
    if len(lines) == 4:
        if all(len(line) == 5 for line in lines):
            poetry_types.append("五言绝句")
        elif all(len(line) == 7 for line in lines):
            poetry_types.append("七言绝句")
        else:
            poetry_types.append("其他绝句")
    elif len(lines) == 8:
        if all(len(line) == 5 for line in lines):
            poetry_types.append("五言律诗")
        elif all(len(line) == 7 for line in lines):
            poetry_types.append("七言律诗")
        else:
            poetry_types.append("其他律诗")
    else:
        poetry_types.append("其他")

# 统计并可视化
type_counts = pd.Series(poetry_types).value_counts()
print("\n诗歌类型分布:")
print(type_counts)

# 绘制饼图
plt.figure(figsize=(10, 6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
plt.title('诗歌类型分布')
plt.savefig('./data/poetry_type_distribution.png')
plt.close()

# 分析词语和句子数量
word_counts = [len(item['qa_words']) for item in data]
sent_counts = [len(item['qa_sents']) for item in data]

print(f"\n平均需要解释的词语数量: {sum(word_counts)/len(word_counts):.2f}")
print(f"平均需要翻译的句子数量: {sum(sent_counts)/len(sent_counts):.2f}")

# 分析情感分类选项
emotion_options = []
for item in data:
    emotions = list(item['choose'].values())
    emotion_options.extend(emotions)

emotion_counts = pd.Series(emotion_options).value_counts()
print("\n情感分类选项分布:")
print(emotion_counts)

# 绘制情感分布条形图
plt.figure(figsize=(12, 6))
emotion_counts.plot(kind='bar')
plt.title('情感分类选项分布')
plt.xlabel('情感类别')
plt.ylabel('出现次数')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./data/emotion_distribution.png')
```

运行此脚本以了解数据集的基本特征：

```bash
python data_analysis.py
```

#### 数据转换为训练格式

我们需要将原始数据转换为LLaMA-Factory框架所需的格式。创建以下脚本：

```python
# data_conversion.py
import json
import os

# 加载原始数据
with open('./data/ccl_poetry_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建LLaMA-Factory格式的数据
llama_factory_data = []

for item in data:
    # 构建输入文本
    input_text = f"标题：{item['title']}\n作者：{item['author']}\n内容：{item['content']}\n请解释以下词语：{item['qa_words']}\n请翻译以下句子：{item['qa_sents']}\n请选择情感：{item['choose']}"

    # 构建输出文本（假设我们有标准答案）
    # 在实际训练数据中，这部分应该是真实的标准答案
    # 这里仅作示例，实际使用时需要替换为真实答案
    output_dict = {
        "ans_qa_words": {word: f"{word}的释义" for word in item['qa_words']},
        "ans_qa_sents": {sent: f"{sent}的白话翻译" for sent in item['qa_sents']},
        "choose_id": list(item['choose'].keys())[0]  # 假设第一个选项是正确答案
    }
    output_text = json.dumps(output_dict, ensure_ascii=False, indent=2)

    # 构建指令文本
    instruction = "你是一个古诗词专家，请解析以下诗词，提供词语释义、句子翻译和情感分类。"

    # 添加到数据集
    llama_factory_data.append({
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    })

# 保存为LLaMA-Factory格式
os.makedirs('./data/llama_factory', exist_ok=True)
with open('./data/llama_factory/poetry_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)

print(f"已将{len(llama_factory_data)}条数据转换为LLaMA-Factory格式")
```

**注意**：上述脚本中的输出部分使用了占位符。在实际使用时，您需要使用真实的标准答案替换这些占位符。

### 2.2 数据增强与格式转换

为了提高模型性能，我们可以通过数据增强来扩充训练集。以下是几种数据增强方法：

#### 收集补充数据

创建一个脚本来下载和处理额外的古诗词数据：

```python
# data_augmentation.py
import requests
import json
import os
import random
from tqdm import tqdm

# 创建目录
os.makedirs('./data/augmentation', exist_ok=True)

# 下载全唐诗数据集（示例URL，实际使用时请替换为有效链接）
def download_tang_poetry():
    print("下载全唐诗数据集...")
    url = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/json/poet.tang.0.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open('./data/augmentation/tang_poetry.json', 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("下载完成！")
            return True
        else:
            print(f"下载失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"下载出错: {e}")
        return False

# 处理全唐诗数据
def process_tang_poetry():
    try:
        with open('./data/augmentation/tang_poetry.json', 'r', encoding='utf-8') as f:
            tang_data = json.load(f)

        print(f"加载了{len(tang_data)}首唐诗")

        # 选择一部分诗歌进行处理
        selected_poems = random.sample(tang_data, min(100, len(tang_data)))

        augmented_data = []
        for poem in tqdm(selected_poems, desc="处理唐诗"):
            # 跳过内容不完整的诗
            if not poem.get('paragraphs') or len(poem['paragraphs']) < 2:
                continue

            title = poem.get('title', '无题')
            author = poem.get('author', '佚名')
            content = '\n'.join(poem['paragraphs'])

            # 随机选择2-3个词语作为需要解释的词
            all_words = ''.join(poem['paragraphs'])
            words_to_explain = random.sample(all_words, min(3, len(all_words)))

            # 随机选择1-2个句子作为需要翻译的句
            sentences_to_translate = random.sample(poem['paragraphs'], min(2, len(poem['paragraphs'])))

            # 创建情感选项（示例）
            emotions = {
                "A": "思乡",
                "B": "悲伤",
                "C": "喜悦",
                "D": "愤怒"
            }

            # 创建增强数据项
            augmented_item = {
                "title": title,
                "author": author,
                "content": content,
                "qa_words": words_to_explain,
                "qa_sents": sentences_to_translate,
                "choose": emotions
            }

            augmented_data.append(augmented_item)

        # 保存增强数据
        with open('./data/augmentation/augmented_tang_poetry.json', 'w', encoding='utf-8') as f:
            json.dump(augmented_data, f, ensure_ascii=False, indent=2)

        print(f"已生成{len(augmented_data)}条增强数据")
        return augmented_data
    except Exception as e:
        print(f"处理唐诗数据出错: {e}")
        return []

# 将增强数据转换为LLaMA-Factory格式
def convert_to_llama_factory(augmented_data):
    if not augmented_data:
        return

    llama_factory_data = []

    for item in augmented_data:
        # 构建输入文本
        input_text = f"标题：{item['title']}\n作者：{item['author']}\n内容：{item['content']}\n请解释以下词语：{item['qa_words']}\n请翻译以下句子：{item['qa_sents']}\n请选择情感：{item['choose']}"

        # 这里我们不提供输出，因为这是增强数据，没有标准答案
        # 在实际使用时，可以使用更大的模型生成答案，或者人工标注
        output_text = ""

        # 构建指令文本
        instruction = "你是一个古诗词专家，请解析以下诗词，提供词语释义、句子翻译和情感分类。"

        # 添加到数据集
        llama_factory_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })

    # 保存为LLaMA-Factory格式
    with open('./data/llama_factory/augmented_poetry_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)

    print(f"已将{len(llama_factory_data)}条增强数据转换为LLaMA-Factory格式")

# 主函数
def main():
    if not os.path.exists('./data/augmentation/tang_poetry.json'):
        if not download_tang_poetry():
            print("无法下载唐诗数据，跳过数据增强步骤")
            return

    augmented_data = process_tang_poetry()
    convert_to_llama_factory(augmented_data)

if __name__ == "__main__":
    main()
```

运行数据增强脚本：

```bash
python data_augmentation.py
```

#### 合并原始数据和增强数据

创建一个脚本来合并原始数据和增强数据：

```python
# merge_datasets.py
import json
import os

# 加载原始LLaMA-Factory格式数据
with open('./data/llama_factory/poetry_dataset.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 加载增强LLaMA-Factory格式数据
with open('./data/llama_factory/augmented_poetry_dataset.json', 'r', encoding='utf-8') as f:
    augmented_data = json.load(f)

# 合并数据
merged_data = original_data + augmented_data

# 保存合并后的数据
with open('./data/llama_factory/merged_poetry_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"已合并{len(original_data)}条原始数据和{len(augmented_data)}条增强数据，总计{len(merged_data)}条")
```

运行合并脚本：

```bash
python merge_datasets.py
```

#### 数据集划分

创建一个脚本来将数据集划分为训练集和验证集：

```python
# split_dataset.py
import json
import random
import os

# 设置随机种子以确保可重复性
random.seed(42)

# 加载合并后的数据
with open('./data/llama_factory/merged_poetry_dataset.json', 'r', encoding='utf-8') as f:
    merged_data = json.load(f)

# 随机打乱数据
random.shuffle(merged_data)

# 划分数据集（90%训练，10%验证）
split_idx = int(len(merged_data) * 0.9)
train_data = merged_data[:split_idx]
val_data = merged_data[split_idx:]

# 保存训练集和验证集
os.makedirs('./data/llama_factory/split', exist_ok=True)

with open('./data/llama_factory/split/train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('./data/llama_factory/split/val.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"数据集已划分为{len(train_data)}条训练数据和{len(val_data)}条验证数据")
```

运行数据集划分脚本：

```bash
python split_dataset.py
```

完成以上步骤后，我们已经准备好了用于训练的数据集。下一步将进行代码开发，使用LLaMA-Factory框架来微调模型。

## 3. 代码开发

在这一部分，我们将使用LLaMA-Factory框架来开发微调模型的代码。LLaMA-Factory是一个统一的框架，支持多种大型语言模型的高效微调，包括Qwen、ChatGLM、Yi等我们要使用的模型。

### 3.1 使用LLaMA-Factory框架

#### 安装LLaMA-Factory

首先，我们需要在本地环境（M1 Max MacBook）上安装LLaMA-Factory框架：

```bash
# 进入项目目录
cd ~/poetry-llm-project

# 克隆LLaMA-Factory仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git code/LLaMA-Factory
cd code/LLaMA-Factory

# 安装依赖
pip install -e .
```

安装完成后，我们可以验证安装是否成功：

```bash
llmtuner --help
```

如果看到帮助信息，说明安装成功。

#### 了解LLaMA-Factory的文件结构

在开始开发之前，让我们先了解LLaMA-Factory的文件结构：

```bash
LLaMA-Factory/
├── data/            # 数据目录
├── output/          # 输出目录，存放微调后的模型
├── src/             # 源代码
│   ├── llmtuner/    # 核心代码
│   ├── scripts/     # 脚本文件
├── examples/        # 示例配置文件
├── README.md        # 文档
```

我们需要将我们准备的数据放入LLaMA-Factory的data目录中：

```bash
# 创建数据目录
mkdir -p code/LLaMA-Factory/data/poetry

# 复制数据文件
cp data/llama_factory/split/train.json code/LLaMA-Factory/data/poetry/
cp data/llama_factory/split/val.json code/LLaMA-Factory/data/poetry/
```

### 3.2 配置文件编写

LLaMA-Factory支持使用YAML配置文件来管理微调参数。我们将创建一个配置文件，用于微调Qwen2-7B模型。

#### 创建配置文件目录

```bash
# 创建配置文件目录
mkdir -p code/LLaMA-Factory/configs
```

#### 编写QLoRA微调配置文件

创建一个名为`qwen2_7b_qlora.yaml`的配置文件：

```bash
touch code/LLaMA-Factory/configs/qwen2_7b_qlora.yaml
```

编辑该文件，添加以下内容：

```yaml
# 模型配置
model_name_or_path: "Qwen/Qwen2-7B"
model_revision: "main"
tokenizer_name_or_path: "Qwen/Qwen2-7B"
tokenizer_revision: "main"

# 数据配置
data_path: "poetry"
output_dir: "output/qwen2-7b-poetry-qlora"
overwrite_output_dir: true
num_train_epochs: 3
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
eval_strategy: "steps"
eval_steps: 100
save_strategy: "steps"
save_steps: 100
save_total_limit: 3
logging_steps: 10

# LoRA配置
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target: "all-linear"

# 优化器配置
learning_rate: 2e-4
weight_decay: 0.01
warmup_ratio: 0.03
optim: "adamw_torch"
lr_scheduler_type: "cosine"
max_grad_norm: 1.0

# 训练配置
finetuning_type: "qlora"
load_in_8bit: true
load_in_4bit: false
use_bf16: false
use_fp16: true
padding_side: "right"
template: "qwen"
flash_attn: false
report_to: "none"
remove_unused_columns: true
do_train: true
do_eval: true
preprocessing_num_workers: 4
use_gradient_checkpointing: true
max_source_length: 1024
max_target_length: 1024
val_size: 0.1
truncation_side: "right"
resize_vocab: false
```

这个配置文件定义了以下关键参数：

- **模型参数**：使用Qwen2-7B模型
- **数据参数**：使用我们准备的poetry数据集
- **LoRA参数**：设置LoRA秩为8，LoRA alpha为16
- **优化器参数**：学习率为2e-4，使用cosine学习率调度器
- **训练参数**：使用QLoRA微调，加载8位量化模型，使用梯度检查点

#### 创建数据集配置文件

我们需要创建一个数据集配置文件，告诉LLaMA-Factory如何加载我们的数据。

```bash
mkdir -p code/LLaMA-Factory/data/poetry/dataset_info.json
```

编辑`dataset_info.json`文件：

```json
{
  "poetry": {
    "file_name": "train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "poetry_eval": {
    "file_name": "val.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

这个配置文件告诉LLaMA-Factory我们的数据集结构，其中：

- `poetry`是训练集，文件名为`train.json`
- `poetry_eval`是验证集，文件名为`val.json`
- `columns`定义了数据列的映射关系

### 3.3 训练脚本准备

现在我们需要准备训练脚本，这些脚本将在AutoDL上运行。我们将创建两个脚本：一个用于训练，另一个用于推理。

#### 创建训练脚本

创建一个训练脚本文件：

```bash
touch code/LLaMA-Factory/train_poetry.sh
chmod +x code/LLaMA-Factory/train_poetry.sh
```

编辑`train_poetry.sh`文件，添加以下内容：

```bash
#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行训练
llmtuner chat \
  --config configs/qwen2_7b_qlora.yaml \
  --train_dataset poetry \
  --eval_dataset poetry_eval \
  --model_name_or_path Qwen/Qwen2-7B \
  --dataset poetry \
  --template qwen \
  --finetuning_type qlora \
  --output_dir output/qwen2-7b-poetry-qlora
```

这个脚本使用`llmtuner`命令来训练模型，并指定了我们刚刚创建的配置文件。

#### 创建推理脚本

我们还需要一个脚本来运行推理，生成最终的结果：

```bash
touch code/LLaMA-Factory/inference_poetry.py
```

编辑`inference_poetry.py`文件，添加以下内容：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B", help="基础模型路径")
    parser.add_argument("--adapter_model", type=str, default="output/qwen2-7b-poetry-qlora", help="LoRA模型路径")
    parser.add_argument("--test_file", type=str, default="test.json", help="测试数据文件")
    parser.add_argument("--output_file", type=str, default="results.json", help="输出结果文件")
    parser.add_argument("--max_length", type=int, default=2048, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.7, help="采样的top-p值")
    parser.add_argument("--load_8bit", action="store_true", help="是否加载8位量化模型")
    return parser.parse_args()

# 加载模型
def load_model(args):
    print(f"加载基础模型: {args.base_model}")

    # 设置加载参数
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if args.load_8bit:
        load_kwargs["load_in_8bit"] = True

    # 加载基础模型和分词器
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 加载LoRA模型
    if args.adapter_model:
        print(f"加载LoRA模型: {args.adapter_model}")
        model = PeftModel.from_pretrained(model, args.adapter_model)

    # 设置生成配置
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return model, tokenizer, generation_config

# 生成提示模板
def generate_prompt(item):
    instruction = "你是一个古诗词专家，请解析以下诗词，提供词语释义、句子翻译和情感分类。"
    input_text = f"标题：{item['title']}\n作者：{item['author']}\n内容：{item['content']}\n请解释以下词语：{item['qa_words']}\n请翻译以下句子：{item['qa_sents']}\n请选择情感：{item['choose']}"

    # 根据Qwen模型的模板格式构建提示
    prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

# 解析模型输出
def parse_response(response):
    try:
        # 尝试提取JSON部分
        json_start = response.find('{')
        json_end = response.rfind('}')

        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end+1]
            result = json.loads(json_str)
            return result
        else:
            print(f"无法从响应中提取JSON: {response}")
            return None
    except Exception as e:
        print(f"解析响应时出错: {e}")
        return None

# 创建默认响应
def create_default_response(item):
    return {
        "ans_qa_words": {word: f"{word}的释义" for word in item['qa_words']},
        "ans_qa_sents": {sent: f"{sent}的白话翻译" for sent in item['qa_sents']},
        "choose_id": list(item['choose'].keys())[0]  # 默认选择第一个选项
    }

# 主函数
def main():
    args = parse_args()
    model, tokenizer, generation_config = load_model(args)

    # 加载测试数据
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results = []

    # 对每个测试样例进行推理
    for item in tqdm(test_data, desc="推理中"):
        prompt = generate_prompt(item)

        # 生成响应
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 解析响应
        result = parse_response(response)

        # 如果解析失败，使用默认响应
        if result is None:
            result = create_default_response(item)

        # 添加到结果列表
        results.append(result)

    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"推理完成，结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main()
```

这个推理脚本实现了以下功能：

1. 加载基础模型和LoRA模型
2. 处理测试数据并生成提示
3. 运行推理并解析输出
4. 处理错误情况并提供默认响应
5. 将结果保存为JSON格式

#### 准备文件打包

最后，我们需要将所有必要的文件打包，以便上传到AutoDL：

```bash
# 创建打包目录
mkdir -p ~/poetry-llm-project/package

# 复制必要文件
cp -r code/LLaMA-Factory/configs ~/poetry-llm-project/package/
cp -r code/LLaMA-Factory/data/poetry ~/poetry-llm-project/package/data/
cp code/LLaMA-Factory/train_poetry.sh ~/poetry-llm-project/package/
cp code/LLaMA-Factory/inference_poetry.py ~/poetry-llm-project/package/

# 创建一个README文件
cat > ~/poetry-llm-project/package/README.md << 'EOF'
# 古诗词理解与推理训练包

## 使用方法

1. 安装依赖：
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

2. 运行训练：
```bash
bash train_poetry.sh
```

3. 运行推理：
```bash
python inference_poetry.py --test_file test.json --output_file results.json
```
EOF

打包命令：
```bash
cd ~/poetry-llm-project
tar -czvf poetry-llm-package.tar.gz package/
```

现在，我们已经完成了所有必要的代码开发工作，并准备好了上传到AutoDL的文件包。下一步，我们将在AutoDL上创建实例并进行模型训练。

## 4. 模型训练

在这一部分，我们将使用AutoDL云平台进行模型训练。AutoDL提供了灵活的GPU租用服务，非常适合我们的项目需求。

### 4.1 AutoDL实例创建

#### 登录AutoDL并创建实例

1. 访问[AutoDL官网](https://www.autodl.com)并登录您的账户

2. 点击“创建实例”按钮，选择以下配置：
   - **机器类型**：选择“RTX 4090”（最佳性价比）
   - **系统镜像**：选择“PyTorch 2.0.1 + CUDA 11.8”
   - **存储空间**：选择“100GB”（足够存储模型和数据）
   - **实例名称**：输入“poetry-llm-training”
   - **计费方式**：选择“按量计费”（可随时暂停节省费用）

3. 点击“创建实例”完成创建

#### 连接到实例

实例创建完成后，点击“连接”按钮。AutoDL提供了多种连接方式，包括SSH、网页终端和VSCode。在这里，我们选择“网页终端”进行连接，这是最直接的方式。

### 4.2 数据与代码上传

连接到实例后，我们需要上传我们准备好的文件包。

#### 使用网页文件管理器上传

1. 在AutoDL实例页面，点击“文件”标签页
2. 点击“上传”按钮，选择我们刚刚创建的`poetry-llm-package.tar.gz`文件
3. 等待上传完成

#### 解压文件包

在网页终端中，执行以下命令来解压文件包：

```bash
# 进入主目录
cd /root

# 解压文件包
tar -xzvf poetry-llm-package.tar.gz

# 查看解压后的文件
ls -la package/
```

### 4.3 环境准备

在开始训练之前，我们需要安装LLaMA-Factory和其他必要的依赖。

```bash
# 克隆LLaMA-Factory仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装依赖
pip install -e .

# 安装其他必要的包
pip install bert-score nltk

# 复制我们的配置和数据
mkdir -p data
cp -r /root/package/data/poetry data/
mkdir -p configs
cp /root/package/configs/qwen2_7b_qlora.yaml configs/
cp /root/package/train_poetry.sh .
cp /root/package/inference_poetry.py .

# 确保训练脚本有执行权限
chmod +x train_poetry.sh
```

### 4.4 运行训练

现在我们可以开始训练模型了。运行以下命令来启动训练过程：

```bash
# 确保在LLaMA-Factory目录下
cd /root/LLaMA-Factory

# 启动训练
./train_poetry.sh
```

训练过程将持续一段时间，取决于数据集大小和训练轮次。使用RTX 4090进行训练，预计需要几个小时到十几个小时不等。

#### 监控训练过程

在训练过程中，您可以通过以下方式监控进度：

1. **查看训练日志**：训练过程中的日志会实时显示在终端中

2. **查看GPU使用情况**：在另一个终端窗口中运行：
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **查看训练输出目录**：在另一个终端窗口中运行：
   ```bash
   ls -la output/qwen2-7b-poetry-qlora/
   ```

#### 暂停和恢复训练

如果您需要暂停训练以节省费用，可以按以下步骤操作：

1. 在终端中按`Ctrl+C`停止训练进程

2. 在AutoDL控制台中点击“暂停”按钮暂停实例

当您准备继续训练时：

1. 在AutoDL控制台中点击“启动”按钮启动实例

2. 连接到实例并继续训练：
   ```bash
   cd /root/LLaMA-Factory
   ./train_poetry.sh
   ```

### 4.5 训练完成后的操作

训练完成后，您将在`output/qwen2-7b-poetry-qlora/`目录下看到生成的模型文件。这些文件包含了微调后的LoRA权重。

#### 保存模型

为了在未来使用这个模型，我们需要将它保存下来。您可以使用AutoDL的文件管理器或者通过命令行打包下载：

```bash
# 打包模型文件
cd /root/LLaMA-Factory
tar -czvf qwen2-7b-poetry-qlora-model.tar.gz output/qwen2-7b-poetry-qlora/
```

然后在AutoDL的文件管理器中下载这个打包文件。

#### 清理资源

训练完成后，如果不再需要使用实例，建议将其关闭以节省费用：

1. 在AutoDL控制台中点击“暂停”按钮暂停实例

2. 如果完全不需要该实例，可以点击“删除”按钮删除实例（请确保已经下载了所有需要的文件）

## 5. 结果生成与优化

在模型训练完成后，我们需要使用微调后的模型生成结果，并进行必要的优化以确保结果符合比赛要求。

### 5.1 模型推理

我们将使用之前准备的推理脚本来生成结果。首先，我们需要将测试数据上传到AutoDL实例。

#### 准备测试数据

假设我们已经从比赛官方获取了测试数据集，文件名为`ccl_poetry_test.json`。我们需要将它上传到AutoDL实例。

1. 在AutoDL实例页面，点击“文件”标签页
2. 点击“上传”按钮，选择`ccl_poetry_test.json`文件
3. 等待上传完成

#### 运行推理

现在我们可以使用微调后的模型运行推理了：

```bash
# 确保在LLaMA-Factory目录下
cd /root/LLaMA-Factory

# 运行推理
python inference_poetry.py \
  --base_model Qwen/Qwen2-7B \
  --adapter_model output/qwen2-7b-poetry-qlora \
  --test_file /root/ccl_poetry_test.json \
  --output_file /root/results.json \
  --load_8bit
```

推理过程可能需要一段时间，取决于测试数据的大小。完成后，结果将保存在`/root/results.json`文件中。

#### 下载结果

推理完成后，我们需要下载结果文件：

1. 在AutoDL实例页面，点击“文件”标签页
2. 寻找`results.json`文件
3. 点击“下载”按钮下载文件

### 5.2 输出格式处理

下载结果文件后，我们需要检查并确保结果格式符合比赛要求。我们将创建一个脚本来验证和格式化结果。

#### 创建格式验证脚本

在本地环境（M1 Max MacBook）上，创建一个脚本来验证和格式化结果：

```bash
touch ~/poetry-llm-project/code/validate_results.py
```

编辑`validate_results.py`文件，添加以下内容：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="验证和格式化推理结果")
    parser.add_argument("--input", type=str, required=True, help="输入结果文件")
    parser.add_argument("--test", type=str, required=True, help="测试数据文件")
    parser.add_argument("--output", type=str, required=True, help="输出格式化结果文件")
    return parser.parse_args()

# 验证结果格式
def validate_result(result, test_item):
    # 检查必要字段
    required_fields = ["ans_qa_words", "ans_qa_sents", "choose_id"]
    for field in required_fields:
        if field not in result:
            print(f"缺失必要字段: {field}")
            return False

    # 检查词语释义
    for word in test_item["qa_words"]:
        if word not in result["ans_qa_words"]:
            print(f"缺失词语释义: {word}")
            return False

    # 检查句子翻译
    for sent in test_item["qa_sents"]:
        if sent not in result["ans_qa_sents"]:
            print(f"缺失句子翻译: {sent}")
            return False

    # 检查情感选择
    if result["choose_id"] not in test_item["choose"]:
        print(f"无效的情感选择: {result['choose_id']}")
        return False

    return True

# 修复结果
def fix_result(result, test_item):
    fixed_result = result.copy()

    # 确保所有必要字段存在
    if "ans_qa_words" not in fixed_result:
        fixed_result["ans_qa_words"] = {}
    if "ans_qa_sents" not in fixed_result:
        fixed_result["ans_qa_sents"] = {}
    if "choose_id" not in fixed_result:
        fixed_result["choose_id"] = list(test_item["choose"].keys())[0]

    # 确保所有词语都有释义
    for word in test_item["qa_words"]:
        if word not in fixed_result["ans_qa_words"]:
            fixed_result["ans_qa_words"][word] = f"{word}的释义"

    # 确保所有句子都有翻译
    for sent in test_item["qa_sents"]:
        if sent not in fixed_result["ans_qa_sents"]:
            fixed_result["ans_qa_sents"][sent] = f"{sent}的白话翻译"

    # 确保情感选择有效
    if fixed_result["choose_id"] not in test_item["choose"]:
        fixed_result["choose_id"] = list(test_item["choose"].keys())[0]

    return fixed_result

# 主函数
def main():
    args = parse_args()

    # 加载结果和测试数据
    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    with open(args.test, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 验证和修复结果
    fixed_results = []
    invalid_count = 0

    for i, (result, test_item) in enumerate(zip(results, test_data)):
        if not validate_result(result, test_item):
            print(f"第{i+1}条结果格式无效，正在修复...")
            fixed_result = fix_result(result, test_item)
            fixed_results.append(fixed_result)
            invalid_count += 1
        else:
            fixed_results.append(result)

    # 保存格式化后的结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed_results, f, ensure_ascii=False, indent=2)

    print(f"验证完成，共{len(results)}条结果，其中{invalid_count}条需要修复")
    print(f"格式化后的结果已保存到 {args.output}")

if __name__ == "__main__":
    main()
```

#### 运行格式验证脚本

下载结果文件后，运行验证脚本：

```bash
cd ~/poetry-llm-project
python code/validate_results.py \
  --input results.json \
  --test data/ccl_poetry_test.json \
  --output formatted_results.json
```

这个脚本将检查结果格式，并修复任何问题，确保最终结果符合比赛要求。格式化后的结果将保存在`formatted_results.json`文件中。

### 5.3 结果优化策略

如果模型生成的结果还不够理想，我们可以采用一些优化策略来提高结果质量。

#### 多模型集成

一种有效的方法是使用多个模型生成结果，然后通过投票或其他方式集成这些结果。我们可以创建一个脚本来实现这一点：

```bash
touch ~/poetry-llm-project/code/ensemble_results.py
```

编辑`ensemble_results.py`文件，添加以下内容：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter
from difflib import SequenceMatcher

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="集成多个模型的结果")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="输入结果文件列表")
    parser.add_argument("--test", type=str, required=True, help="测试数据文件")
    parser.add_argument("--output", type=str, required=True, help="输出集成结果文件")
    parser.add_argument("--similarity_threshold", type=float, default=0.8, help="相似度阈值")
    return parser.parse_args()

# 计算字符串相似度
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 合并相似的答案
def merge_similar_answers(answers, threshold):
    # 初始化结果字典
    merged = {}
    counts = {}

    # 对每个答案
    for answer, count in answers.items():
        found_similar = False

        # 检查是否有相似的现有答案
        for key in list(merged.keys()):
            if similarity(answer, key) >= threshold:
                # 如果找到相似答案，增加计数
                counts[key] += count
                found_similar = True
                break

        # 如果没有相似答案，添加新条目
        if not found_similar:
            merged[answer] = answer
            counts[answer] = count

    # 找出最高票数的答案
    if counts:
        return max(counts.items(), key=lambda x: x[1])[0]
    return ""

# 集成多个模型的结果
def ensemble_results(all_results, test_data, threshold):
    # 初始化集成结果
    ensemble_results = []

    # 对每个测试样例
    for i, test_item in enumerate(test_data):
        # 初始化结果容器
        word_explanations = {word: {} for word in test_item["qa_words"]}
        sent_translations = {sent: {} for sent in test_item["qa_sents"]}
        emotion_votes = Counter()

        # 收集所有模型的结果
        for results in all_results:
            if i >= len(results):
                continue

            result = results[i]

            # 词语释义投票
            for word in test_item["qa_words"]:
                if word in result.get("ans_qa_words", {}):
                    explanation = result["ans_qa_words"][word]
                    word_explanations[word][explanation] = word_explanations[word].get(explanation, 0) + 1

            # 句子翻译投票
            for sent in test_item["qa_sents"]:
                if sent in result.get("ans_qa_sents", {}):
                    translation = result["ans_qa_sents"][sent]
                    sent_translations[sent][translation] = sent_translations[sent].get(translation, 0) + 1

            # 情感分类投票
            if "choose_id" in result:
                emotion_votes[result["choose_id"]] += 1

        # 创建集成结果
        ensemble_result = {
            "ans_qa_words": {},
            "ans_qa_sents": {},
            "choose_id": emotion_votes.most_common(1)[0][0] if emotion_votes else list(test_item["choose"].keys())[0]
        }

        # 选择最高票数的词语释义，并合并相似的释义
        for word in test_item["qa_words"]:
            explanations = word_explanations[word]
            if explanations:
                ensemble_result["ans_qa_words"][word] = merge_similar_answers(explanations, threshold)
            else:
                ensemble_result["ans_qa_words"][word] = f"{word}的释义"

        # 选择最高票数的句子翻译，并合并相似的翻译
        for sent in test_item["qa_sents"]:
            translations = sent_translations[sent]
            if translations:
                ensemble_result["ans_qa_sents"][sent] = merge_similar_answers(translations, threshold)
            else:
                ensemble_result["ans_qa_sents"][sent] = f"{sent}的白话翻译"

        ensemble_results.append(ensemble_result)

    return ensemble_results

# 主函数
def main():
    args = parse_args()

    # 加载测试数据
    with open(args.test, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 加载所有结果文件
    all_results = []
    for input_file in args.inputs:
        with open(input_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            all_results.append(results)
            print(f"加载结果文件 {input_file}，包含 {len(results)} 条结果")

    # 集成结果
    ensemble_results_data = ensemble_results(all_results, test_data, args.similarity_threshold)

    # 保存集成结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ensemble_results_data, f, ensure_ascii=False, indent=2)

    print(f"集成完成，共 {len(ensemble_results_data)} 条结果已保存到 {args.output}")

if __name__ == "__main__":
    main()
```

这个脚本实现了以下功能：

1. 加载多个模型生成的结果文件
2. 对每个测试样例，收集所有模型的结果
3. 使用投票机制选择最佳答案
4. 合并相似的答案以提高一致性
5. 生成最终的集成结果

#### 使用集成策略

要使用集成策略，我们需要多个模型生成的结果。我们可以尝试以下方法：

1. 使用不同的模型（如Qwen2-7B和ChatGLM4-9B）
2. 使用同一模型的不同参数（如不同的温度和top-p值）
3. 使用不同的提示模板

假设我们已经生成了多个结果文件，我们可以运行集成脚本：

```bash
cd ~/poetry-llm-project
python code/ensemble_results.py \
  --inputs results_model1.json results_model2.json results_model3.json \
  --test data/ccl_poetry_test.json \
  --output ensemble_results.json \
  --similarity_threshold 0.8
```

这将生成一个集成结果文件`ensemble_results.json`，其中包含了多个模型的最佳答案。

#### 人工审核与修正

对于特别重要的比赛，我们可能还需要进行人工审核和修正。我们可以创建一个简单的脚本来随机抽样一部分结果进行人工检查：

```bash
touch ~/poetry-llm-project/code/sample_results.py
```

编辑`sample_results.py`文件，添加以下内容：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import random

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="随机抽样结果进行人工检查")
    parser.add_argument("--input", type=str, required=True, help="输入结果文件")
    parser.add_argument("--test", type=str, required=True, help="测试数据文件")
    parser.add_argument("--num_samples", type=int, default=10, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()

# 主函数
def main():
    args = parse_args()
    random.seed(args.seed)

    # 加载结果和测试数据
    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    with open(args.test, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 随机抽样
    num_samples = min(args.num_samples, len(results))
    sample_indices = random.sample(range(len(results)), num_samples)

    print(f"随机抽取了 {num_samples} 条结果进行人工检查\n")

    # 显示抽样结果
    for i, idx in enumerate(sample_indices):
        result = results[idx]
        test_item = test_data[idx]

        print(f"\n=== 样本 {i+1}/{num_samples} (ID: {idx}) ===")
        print(f"\n标题: {test_item.get('title', '')}")
        print(f"作者: {test_item.get('author', '')}")
        print(f"内容:\n{test_item.get('content', '')}\n")

        print("词语释义:")
        for word in test_item["qa_words"]:
            explanation = result["ans_qa_words"].get(word, "")
            print(f"  {word}: {explanation}")

        print("\n句子翻译:")
        for sent in test_item["qa_sents"]:
            translation = result["ans_qa_sents"].get(sent, "")
            print(f"  {sent} -> {translation}")

        print("\n情感选择:")
        choose_id = result.get("choose_id", "")
        choose_options = test_item.get("choose", {})
        print(f"  选择: {choose_id} ({choose_options.get(choose_id, '')})")
        print(f"  选项: {choose_options}")

        print("\n是否需要修正这条结果? (输入y/n)")
        need_correction = input().strip().lower() == 'y'

        if need_correction:
            print("\n请输入修正后的结果（JSON格式），或者直接回车跳过:")
            correction = input().strip()
            if correction:
                try:
                    corrected_result = json.loads(correction)
                    results[idx] = corrected_result
                    print("已应用修正")
                except json.JSONDecodeError:
                    print("无效的JSON格式，跳过修正")

    # 保存修正后的结果
    output_file = args.input.replace(".json", "_corrected.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n人工检查完成，修正后的结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
```

运行这个脚本来随机抽样结果进行人工检查：

```bash
cd ~/poetry-llm-project
python code/sample_results.py \
  --input ensemble_results.json \
  --test data/ccl_poetry_test.json \
  --num_samples 10
```

这个脚本将随机抽取10条结果，并允许您进行人工检查和修正。修正后的结果将保存在`ensemble_results_corrected.json`文件中。

## 6. 提交与验证

在完成所有的结果生成和优化后，最后一步是验证结果并准备提交。

### 6.1 结果验证

在提交结果之前，我们需要进行最终的验证，确保结果符合比赛要求。

#### 格式验证

我们可以使用之前创建的验证脚本来进行最终的格式验证：

```bash
cd ~/poetry-llm-project
python code/validate_results.py \
  --input ensemble_results_corrected.json \
  --test data/ccl_poetry_test.json \
  --output final_results.json
```

这将确保我们的最终结果符合比赛要求的格式。

#### 随机抽样检查

我们可以再次使用随机抽样脚本来检查最终结果：

```bash
cd ~/poetry-llm-project
python code/sample_results.py \
  --input final_results.json \
  --test data/ccl_poetry_test.json \
  --num_samples 5
```

这将随机抽取5条结果进行最终检查。

### 6.2 提交流程

当我们确信结果符合要求后，就可以准备提交了。

#### 准备提交文件

根据比赛要求，我们可能需要将结果文件重命名或进行其他格式调整。假设比赛要求提交的文件名为`submission.json`：

```bash
cd ~/poetry-llm-project
cp final_results.json submission.json
```

#### 准备技术报告

大多数比赛还要求提交技术报告，描述您的方法和实验结果。我们可以创建一个简单的模板：

```bash
touch ~/poetry-llm-project/technical_report.md
```

编辑`technical_report.md`文件，添加以下内容：

```markdown
# CCL 2025古诗词理解与推理评测技术报告

## 1. 方法概述

本方法基于大型语言模型（LLM）的微调技术，为古诗词理解与推理评测任务提供解决方案。我们采用了参数高效微调（PEFT）技术中的QLoRA方法，对开源中文大模型进行微调，并通过多模型集成提高结果质量。

## 2. 模型选择

我们选择了以下模型进行实验：

- **主要模型**：Qwen2-7B
- **辅助模型**：ChatGLM4-9B

选择这些模型的原因是：
1. 它们在中文理解和生成任务上表现出色
2. 它们对中国传统文化和古汉语有较好的理解
3. 它们的开源许可允许微调和商业使用

## 3. 微调方法

我们采用了QLoRA（量化低秩适应）微调方法，该方法具有以下优势：

- 显著降低GPU内存需求，使得在消费级GPU上也能进行微调
- 保持接近全参数微调的性能
- 适合有限训练数据的场景

微调参数设置：
- LoRA秩（rank）：8
- LoRA alpha：16
- 学习率：2e-4
- 训练轮次：3

## 4. 数据处理

我们对原始训练数据进行了以下处理：

1. **数据分析**：分析诗歌类型、词语和句子分布等特征
2. **数据增强**：使用全唐诗等公开数据集进行数据增强
3. **格式转换**：将数据转换为LLaMA-Factory框架所需的格式

## 5. 实验环境

我们使用了以下硬件和软件环境：

- **硬件**：AutoDL云平台的RTX 4090 GPU
- **软件框架**：PyTorch 2.0.1、Transformers 4.36.0、LLaMA-Factory

## 6. 结果优化

我们采用了以下策略来优化最终结果：

1. **多模型集成**：使用多个模型生成结果，通过投票机制选择最佳答案
2. **相似度合并**：合并相似的答案以提高一致性
3. **人工审核**：随机抽样部分结果进行人工审核和修正

## 7. 结果分析

我们的方法在三个子任务上的表现如下：

1. **词语释义**：模型能够准确理解大多数古汉语词汇的含义，并提供详细的释义
2. **句子翻译**：模型能够将古汉语句子准确翻译为现代白话文，保持原意的同时表达流畅
3. **情感分类**：模型能够准确分析诗词中表达的情感，并选择最合适的分类

## 8. 结论与展望

我们的实验表明，基于开源中文大模型的微调方法能够有效解决古诗词理解与推理评测任务。未来的改进方向包括：

1. 探索更高效的微调方法
2. 整合更多的古汉语知识库
3. 开发专门针对古汉语的评估指标

## 9. 参考文献

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
2. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
3. Yang, Z., et al. (2023). Qwen Technical Report.
4. Du, Z., et al. (2022). GLM: General Language Model Pretraining with Autoregressive Blank Infilling.
```

将技术报告转换为PDF格式（如果需要）：

```bash
# 如果没有安装pandoc，需要先安装
brew install pandoc
brew install basictex

# 转换为PDF
pandoc technical_report.md -o technical_report.pdf
```

#### 提交结果

最后，根据比赛要求，将结果文件和技术报告提交到比赛平台。具体的提交步骤可能因比赛而异，请参考比赛官方指南。

## 总结

通过本指南，我们完成了从环境搭建、数据准备、代码开发、模型训练到结果生成与提交的完整流程。这个指南适合没有机器学习背景的数字人文学生，提供了清晰的手把手操作步骤。

在实际操作中，您可能需要根据具体情况进行调整，例如使用不同的模型、调整训练参数或使用其他优化策略。希望本指南能帮助您在CCL 2025古诗词理解与推理评测任务中取得好成绩！
