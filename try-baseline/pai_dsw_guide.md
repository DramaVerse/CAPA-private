# 在阿里云PAI-DSW上运行Qwen2.5-7B Baseline指南

本指南将帮助您在已开通的阿里云PAI-DSW实例（ecs.gn7i-c8g1.2xlarge，配备NVIDIA A10 GPU）上运行Qwen2.5-7B模型的baseline，用于古诗词理解与推理评测任务。

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [模型下载与加载](#3-模型下载与加载)
4. [运行Baseline](#4-运行baseline)
5. [结果分析与优化](#5-结果分析与优化)
6. [常见问题与解决方案](#6-常见问题与解决方案)

## 1. 环境准备

### 1.1 检查GPU环境

首先，确认您的PAI-DSW实例GPU环境是否正常：

```bash
# 检查GPU是否可用
nvidia-smi
```

您应该能看到NVIDIA A10 GPU的信息。

### 1.2 安装必要的依赖

```bash
# 安装必要的Python包
pip install transformers==4.36.2 accelerate==0.25.0 torch==2.1.2 tqdm==4.66.1 sentencepiece==0.1.99 protobuf==4.24.4 vllm==0.2.7
```

### 1.3 创建工作目录

我们已经创建了`try-baseline`目录并复制了必要的文件。

## 2. 数据准备

### 2.1 数据格式说明

比赛数据格式如下：

```json
{
    "title": "诗词标题",
    "content": "诗词内容",
    "keywords": {
        "词语1": "词语1的含义",
        "词语2": "词语2的含义"
    },
    "trans": "诗词的白话文翻译",
    "emotion": "诗词表达的情感"
}
```

### 2.2 准备测试数据

我们已经将`data/唐诗/七言绝句/train.json`复制为`try-baseline/test_data.json`作为测试数据。

## 3. 模型下载与加载

有两种方式运行Qwen2.5-7B模型：直接使用Transformers或使用vLLM进行加速。

### 3.1 使用Transformers（推荐初次尝试）

创建一个新的Python脚本`run_transformers.py`：

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

def load_json(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_poetry_data(data, model, tokenizer, max_samples=5):
    """处理诗词数据并使用Qwen模型生成回答"""
    results = []
    
    # 只处理指定数量的样本（用于测试）
    data = data[:max_samples]
    
    for item in tqdm(data, desc="处理诗词"):
        # 构建提示
        prompt = f"""
        你是一个古诗词专家，现在有一些古诗词需要你的帮助。
        我会给你提供一个 JSON 数据，格式如下：
        - **"title"**：古诗词的标题  
        - **"author"**：古诗词的作者（如果有）  
        - **"content"**：古诗词的内容  
        - **"keywords"**：古诗词中需要解释的词语及其含义  
        - **"trans"**：古诗词的白话文翻译  
        - **"emotion"**：古诗词表达的情感  

        这是我的数据：
        ```json
        {json.dumps(item, ensure_ascii=False)}
        ```

        请你根据提供的数据，生成如下 JSON 格式的结果：
        - **"ans_qa_words"**：对诗中的词语进行解释  
        - **"ans_qa_sents"**：对诗中的句子提供白话文翻译  
        - **"choose_id"**：选择最符合该诗词情感的选项ID  

        请确保输出是有效的JSON格式，不要包含任何额外的解释或注释。
        """

        # 构建消息
        messages = [
            {"role": "system", "content": "你是一个擅长古诗词的专家。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成回答
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 提取新生成的token
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 尝试提取JSON
        try:
            # 使用正则表达式提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                results.append(result)
                print(f"成功处理: {item.get('title', '无标题')}")
            else:
                print(f"无法提取JSON: {item.get('title', '无标题')}")
                print(f"原始响应: {response}")
        except json.JSONDecodeError:
            print(f"JSON解析错误: {item.get('title', '无标题')}")
            print(f"原始响应: {response}")
    
    return results

def main():
    # 模型路径
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 加载模型和分词器
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # 使用bfloat16以节省显存
    )
    print("模型加载成功!")
    
    # 加载测试数据
    input_file = "test_data.json"
    output_file = "baseline_output.json"
    
    data = load_json(input_file)
    print(f"成功加载数据，共{len(data)}条")
    
    # 处理数据
    result_data = process_poetry_data(data, model, tokenizer)
    
    # 保存结果
    save_json(result_data, output_file)
    print(f"处理完成。结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
```

### 3.2 使用vLLM加速（适合大规模推理）

创建一个新的Python脚本`run_vllm.py`：

```python
import json
import os
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re

def load_json(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_poetry_data_vllm(data, llm, max_samples=5):
    """使用vLLM处理诗词数据"""
    results = []
    prompts = []
    
    # 只处理指定数量的样本（用于测试）
    data = data[:max_samples]
    
    # 准备所有提示
    for item in data:
        prompt = f"""
        <|im_start|>system
        你是一个古诗词专家，擅长解释古诗词中的词语、翻译句子和分析情感。
        <|im_end|>
        <|im_start|>user
        我会给你提供一个 JSON 数据，格式如下：
        - **"title"**：古诗词的标题  
        - **"author"**：古诗词的作者（如果有）  
        - **"content"**：古诗词的内容  
        - **"keywords"**：古诗词中需要解释的词语及其含义  
        - **"trans"**：古诗词的白话文翻译  
        - **"emotion"**：古诗词表达的情感  

        这是我的数据：
        ```json
        {json.dumps(item, ensure_ascii=False)}
        ```

        请你根据提供的数据，生成如下 JSON 格式的结果：
        - **"ans_qa_words"**：对诗中的词语进行解释  
        - **"ans_qa_sents"**：对诗中的句子提供白话文翻译  
        - **"choose_id"**：选择最符合该诗词情感的选项ID  

        请确保输出是有效的JSON格式，不要包含任何额外的解释或注释。
        <|im_end|>
        <|im_start|>assistant
        """
        prompts.append(prompt)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024
    )
    
    # 批量生成
    print("正在生成回答...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 处理输出
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        
        # 尝试提取JSON
        try:
            # 使用正则表达式提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                results.append(result)
                print(f"成功处理: {data[i].get('title', '无标题')}")
            else:
                print(f"无法提取JSON: {data[i].get('title', '无标题')}")
                print(f"原始响应: {response}")
        except json.JSONDecodeError:
            print(f"JSON解析错误: {data[i].get('title', '无标题')}")
            print(f"原始响应: {response}")
    
    return results

def main():
    # 模型路径
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 加载vLLM模型
    print("正在加载vLLM模型...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # 使用1个GPU
        trust_remote_code=True,
        dtype="bfloat16"  # 使用bfloat16以节省显存
    )
    print("模型加载成功!")
    
    # 加载测试数据
    input_file = "test_data.json"
    output_file = "vllm_output.json"
    
    data = load_json(input_file)
    print(f"成功加载数据，共{len(data)}条")
    
    # 处理数据
    result_data = process_poetry_data_vllm(data, llm)
    
    # 保存结果
    save_json(result_data, output_file)
    print(f"处理完成。结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
```

## 4. 运行Baseline

### 4.1 使用Transformers运行

```bash
cd try-baseline
python run_transformers.py
```

这将加载Qwen2.5-7B-Instruct模型并处理测试数据中的前5个样本。

### 4.2 使用vLLM运行（可选）

```bash
cd try-baseline
python run_vllm.py
```

vLLM通常比Transformers快2-5倍，但需要更多的设置。

### 4.3 监控GPU使用情况

在运行过程中，您可以在另一个终端中监控GPU使用情况：

```bash
watch -n 1 nvidia-smi
```

## 5. 结果分析与优化

### 5.1 检查输出结果

```bash
# 查看输出文件
cat baseline_output.json
```

### 5.2 评估模型性能

创建一个简单的评估脚本`evaluate.py`：

```python
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_results(results, ground_truth):
    """简单评估模型输出与参考答案的匹配程度"""
    total_words = 0
    correct_words = 0
    total_sents = 0
    correct_sents = 0
    
    for result, gt in zip(results, ground_truth):
        # 评估词语解释
        if 'ans_qa_words' in result:
            for word, explanation in result['ans_qa_words'].items():
                total_words += 1
                if word in gt['keywords'] and explanation.strip() == gt['keywords'][word].strip():
                    correct_words += 1
        
        # 评估句子翻译
        if 'ans_qa_sents' in result:
            for sent, translation in result['ans_qa_sents'].items():
                total_sents += 1
                if translation.strip() == gt['trans'].strip():
                    correct_sents += 1
    
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    sent_accuracy = correct_sents / total_sents if total_sents > 0 else 0
    
    return {
        'word_accuracy': word_accuracy,
        'sent_accuracy': sent_accuracy
    }

def main():
    results = load_json('baseline_output.json')
    ground_truth = load_json('test_data.json')[:len(results)]
    
    metrics = evaluate_results(results, ground_truth)
    
    print(f"词语解释准确率: {metrics['word_accuracy']:.4f}")
    print(f"句子翻译准确率: {metrics['sent_accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

运行评估：

```bash
python evaluate.py
```

### 5.3 优化提示词

如果结果不理想，可以尝试优化提示词。创建一个新的脚本`run_optimized.py`，使用更详细的提示：

```python
# 在run_transformers.py的基础上修改prompt部分
prompt = f"""
你是一个古诗词专家，精通古诗词的解释、翻译和情感分析。

我会给你提供一个古诗词的JSON数据，包含以下信息：
- 标题: "{item.get('title', '无标题')}"
- 内容: "{item.get('content', '')}"
- 需要解释的词语: {json.dumps(list(item.get('keywords', {}).keys()), ensure_ascii=False)}
- 参考翻译: "{item.get('trans', '')}"
- 情感: "{item.get('emotion', '')}"

请你完成以下任务：
1. 解释每个词语的含义，尽量与参考解释保持一致
2. 提供整首诗的白话文翻译，尽量与参考翻译保持一致
3. 分析诗词表达的主要情感

请以下面的JSON格式返回结果：
{{
    "ans_qa_words": {{
        "词语1": "词语1的含义",
        "词语2": "词语2的含义"
    }},
    "ans_qa_sents": {{
        "{item.get('content', '')}": "整首诗的白话文翻译"
    }},
    "choose_id": "A"
}}

请确保输出是有效的JSON格式，不要包含任何额外的解释或注释。
"""
```

## 6. 常见问题与解决方案

### 6.1 显存不足

如果遇到显存不足的问题，可以尝试以下方法：

1. 使用8位量化：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True  # 使用8位量化
)
```

2. 减小批处理大小和生成的最大token数：

```python
max_new_tokens=512  # 减小最大生成token数
```

### 6.2 模型输出格式不正确

如果模型输出的JSON格式不正确，可以尝试以下方法：

1. 在提示中强调输出格式的重要性
2. 使用正则表达式提取JSON部分
3. 实现简单的后处理函数修复常见的JSON格式错误

### 6.3 模型加载失败

如果模型加载失败，可能是网络问题或缓存问题，可以尝试：

1. 检查网络连接
2. 清除Hugging Face缓存：

```bash
rm -rf ~/.cache/huggingface/
```

3. 手动下载模型并从本地加载：

```bash
# 手动下载模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

# 从本地加载
model_name = "./Qwen2.5-7B-Instruct"
```

## 总结

本指南详细介绍了如何在阿里云PAI-DSW上使用Qwen2.5-7B模型运行古诗词理解与推理评测任务的baseline。通过按照步骤操作，您可以快速搭建环境、加载模型、处理数据并获得初步结果。

后续可以尝试更多优化方法，如模型微调、提示词工程、集成多个模型等，以提高模型性能。
