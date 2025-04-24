# 在阿里云PAI-DSW上运行Qwen2.5-7B Baseline详细指南

本指南适合数字人文专业的同学，将一步步帮助您在已开通的阿里云PAI-DSW实例（ecs.gn7i-c8g1.2xlarge，配备NVIDIA A10 GPU）上运行Qwen2.5-7B模型的baseline，用于古诗词理解与推理评测任务。

## 目录

1. [登录PAI-DSW平台](#1-登录pai-dsw平台)
2. [准备代码和环境](#2-准备代码和环境)
3. [下载和准备模型](#3-下载和准备模型)
4. [运行Baseline脚本](#4-运行baseline脚本)
5. [查看和分析结果](#5-查看和分析结果)
6. [常见问题与解决方案](#6-常见问题与解决方案)

## 1. 登录PAI-DSW平台

### 1.1 连接到PAI-DSW实例

1. 登录阿里云控制台：https://pai.console.aliyun.com/

2. 在左侧菜单中选择「交互式建模（DSW）」

3. 点击您已创建的实例（ecs.gn7i-c8g1.2xlarge）旁边的「打开」按钮

4. 等待实例启动并进入JupyterLab界面

### 1.2 打开终端

1. 在JupyterLab界面中，点击左侧菜单中的「File」 > 「New」 > 「Terminal」

2. 这将打开一个终端窗口，所有后续操作都将在这个终端中进行

### 1.3 检查GPU环境

在终端中输入以下命令，确认GPU是否可用：

```bash
nvidia-smi
```

您应该能看到类似以下输出，显示一个NVIDIA A10 GPU：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10          On   | 00000000:00:07.0 Off |                    0 |
| N/A   34C    P0    N/A /  N/A |      0MiB / 24576MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

## 2. 准备代码和环境

### 2.1 克隆项目代码

如果您的代码在GitHub上，需要先克隆到PAI-DSW实例上：

```bash
# 切换到合适的目录
cd /mnt/workspace

# 克隆您的GitHub仓库
git clone https://github.com/您的用户名/您的仓库名.git

# 进入项目目录
cd 您的仓库名
```

如果您已经在PAI-DSW上有代码，可以直接进入目录：

```bash
cd /mnt/workspace/您的项目目录
```

### 2.2 创建工作目录

我们将创建一个新的目录来运行baseline：

```bash
# 创建新目录
mkdir -p try-baseline

# 进入目录
cd try-baseline
```

### 2.3 安装必要的依赖包

在终端中运行以下命令安装必要的Python库：

```bash
pip install transformers==4.36.2 accelerate==0.25.0 torch==2.1.2 tqdm==4.66.1 sentencepiece==0.1.99 protobuf==4.24.4
```

这些库的作用如下：
- transformers: Hugging Face的模型库，用于加载和使用Qwen2.5-7B模型
- accelerate: 加速模型训练和推理
- torch: PyTorch深度学习框架
- tqdm: 显示进度条
- sentencepiece和protobuf: 模型分词器所需的库

### 2.4 准备测试数据

我们需要准备一些测试数据来运行baseline。如果您的项目中已有数据，可以复制一部分作为测试：

```bash
# 回到项目根目录
cd /mnt/workspace/您的项目目录

# 复制一部分数据作为测试
cp data/唐诗/七言绝句/train.json try-baseline/test_data.json

# 返回try-baseline目录
cd try-baseline
```

数据格式应该如下：

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

## 3. 下载和准备模型

在这一步，我们将准备使用Qwen模型。由于在中国境内的PAI-DSW环境中，直接访问Hugging Face可能会受到网络限制，我们将使用阿里云自己的ModelScope平台上的Qwen模型。

### 3.1 创建适用于中国环境的运行脚本

我们已经为您准备了一个适合在中国环境运行的脚本`run_model_cn.py`，它使用ModelScope上的Qwen-7B-Chat模型：

```bash
# 确保您在try-baseline目录中
cd /mnt/workspace/CAPA-private/try-baseline

# 查看脚本内容
cat run_model_cn.py
```

这个脚本的主要功能是：
1. 加载ModelScope上的Qwen-7B-Chat模型（国内可访问）
2. 读取测试数据
3. 为每个诗词样本生成解释、翻译和情感分类
4. 将结果保存为JSON文件

### 3.2 安装必要的依赖

在运行脚本之前，我们需要确保安装了所有必要的依赖：

```bash
# 安装必要的Python包
pip install modelscope transformers==4.36.2 accelerate==0.25.0 torch==2.1.2 tqdm==4.66.1 sentencepiece==0.1.99 protobuf==4.24.4
```

这些库的作用如下：
- modelscope: 阿里云的模型平台，用于访问国内模型
- transformers: Hugging Face的模型库，用于加载和使用模型
- accelerate: 加速模型训练和推理
- torch: PyTorch深度学习框架
- tqdm: 显示进度条
- sentencepiece和protobuf: 模型分词器所需的库

### 3.3 了解模型下载过程

当您第一次运行脚本时，系统会自动从ModelScope下载Qwen-7B-Chat模型。这个过程是自动的，但需要注意以下几点：

1. **模型大小**：Qwen-7B模型约14GB，下载可能需要一些时间
2. **存储位置**：模型会被下载到`~/.cache/modelscope/hub`目录
3. **网络要求**：需要稳定的网络连接，但不需要访问国外网站
4. **显存要求**：A10 GPU有24GB显存，足够运行7B参数的模型

如果您想要手动下载模型，可以使用以下命令：

```bash
# 使用modelscope命令行工具下载模型
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen-7B-Chat')"
```

### 3.4 理解模型格式

Qwen-7B-Chat模型使用的是标准格式，包含以下主要文件：

- `config.json`：模型配置文件
- `tokenizer.json`：分词器配置
- `tokenizer_config.json`：分词器额外配置
- `model.safetensors`：模型权重（可能分为多个分片文件）

这些文件会在您首次运行脚本时自动下载，您不需要手动处理这些文件。

## 4. 运行Baseline脚本

现在我们已经准备好了所有必要的文件，可以开始运行baseline脚本了。

### 4.1 运行模型（适用于中国环境）

我们发现在PAI-DSW环境中，直接使用`run_model_cn.py`可能仍然会尝试访问Hugging Face，因此我们创建了一个专门使用阿里云ModelScope API的脚本`run_model_aliyun.py`。

确保您在try-baseline目录中，然后运行以下命令：

```bash
# 确保您在try-baseline目录中
cd /mnt/workspace/CAPA-private/try-baseline

# 安装ModelScope（如果尚未安装）
pip install modelscope

# 运行阿里云专用脚本，处理前5个样本
python run_model_aliyun.py --samples 5
```

这个命令会：
1. 使用ModelScope的pipeline API加载Qwen-7B-Chat模型
2. 如果加载失败，会自动尝试加载更小的Qwen-1.8B-Chat模型
3. 处理test_data.json中的前5个样本
4. 将结果保存到baseline_output.json文件

> **注意**：首次运行时，模型下载可能需要5-10分钟。如果遇到网络问题，脚本会自动尝试使用更小的模型。

### 4.2 监控GPU使用情况

在模型运行过程中，您可以打开另一个终端窗口来监控GPU使用情况：

1. 在JupyterLab中打开一个新的终端（File > New > Terminal）
2. 运行以下命令：

```bash
# 每秒更新一次GPU状态
watch -n 1 nvidia-smi
```

您应该能看到GPU内存使用量增加到约14-16GB（Qwen-7B模型加载后）。

### 4.3 处理更多样本

如果前5个样本处理顺利，您可以尝试处理更多样本：

```bash
# 处理前20个样本
python run_model_aliyun.py --samples 20 --output baseline_output_20.json

# 处理所有样本（可能需要较长时间）
python run_model_aliyun.py --samples 1000 --output baseline_output_all.json
```

### 4.4 理解运行过程

当您运行脚本时，会看到类似以下输出：

```
正在加载模型和分词器...
这可能需要几分钟时间...
模型加载成功!
成功加载数据，共50条
处理诗词: 100%|██████████| 5/5 [01:30<00:00, 18.12s/it]
成功处理: 为有
成功处理: 九月九日忆山东兄弟
成功处理: 静夜思
成功处理: 望庐山瀑布
成功处理: 早发白帝城
处理完成。结果已保存到 baseline_output.json
```

每个样本的处理时间约为15-20秒，这是正常的。如果您处理大量样本，可能需要几个小时。

### 4.5 如果仍然遇到网络问题

如果您在使用`run_model_aliyun.py`时仍然遇到网络问题，可以尝试以下解决方案：

1. 检查是否安装了ModelScope：
```bash
pip install modelscope -U
```

2. 尝试使用更小的模型：
```bash
# 修改run_model_aliyun.py中的模型名称
# 将model='qwen/Qwen-7B-Chat'改为
# model='qwen/Qwen-1.8B-Chat'
```

3. 如果仍然无法下载模型，可以尝试使用本地模型：
```bash
# 查看PAI-DSW环境中是否有预装模型
ls /root/share/models/
```

如果找到了预装的模型，可以修改脚本使用本地模型路径。

## 5. 查看和分析结果

### 5.1 查看输出结果

运行完成后，您可以查看生成的结果文件：

```bash
# 查看结果文件
cat baseline_output.json
```

或者在JupyterLab中打开文件：
1. 在左侧文件浏览器中导航到try-baseline目录
2. 双击baseline_output.json文件

输出结果应该类似于以下格式：

```json
[
    {
        "ans_qa_words": {
            "云屏": "雕饰着云母图案的屏风",
            "凤城": "京城",
            "无端": "没来由",
            "金龟婿": "佩带金龟的丈夫"
        },
        "ans_qa_sents": {
            "为有云屏无限娇，凤城寒尽怕春宵。无端嫁得金龟婿，辜负香衾事早朝。": "云母屏风后面的美人格外娇美，京城寒冬已过只怕春宵短暂。没来由地嫁了个做官的丈夫，不贪恋温暖香衾只想去上早朝。"
        },
        "choose_id": "A"
    },
    ...
]
```

### 5.2 创建简单的评估脚本

为了评估模型输出的质量，我们可以创建一个简单的评估脚本：

```bash
# 创建评估脚本
nano evaluate.py
```

将以下代码复制到编辑器中：

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

然后运行评估脚本：

```bash
python evaluate.py
```

### 5.3 优化提示词（可选）

如果您想尝试改进结果，可以创建一个优化版本的脚本：

```bash
# 复制原始脚本
cp run_model.py run_optimized.py

# 编辑优化版本
nano run_optimized.py
```

在编辑器中，找到`prompt`变量（约在第180行），将其替换为以下更详细的提示：

```python
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

保存文件后，运行优化版本：

```bash
python run_optimized.py --output optimized_output.json
```


## 6. 常见问题与解决方案

在运行过程中，您可能会遇到一些问题。以下是常见问题及其解决方案：

### 6.1 显存不足问题

**问题**：运行时出现"CUDA out of memory"错误。

**解决方案**：

1. 使用8位量化加载模型，修改run_model.py中的模型加载部分：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True  # 使用8位量化而不是bfloat16
)
```

2. 减小生成的最大token数：

```python
# 将max_new_tokens从1024减小到512
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,  # 减小最大生成token数
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

3. 关闭其他占用GPU内存的程序或重启实例。

### 6.2 模型下载问题（中国环境）

**问题**：ModelScope模型下载失败。

**解决方案**：

1. 检查网络连接：

```bash
ping modelscope.cn
```

2. 清除ModelScope缓存后重试：

```bash
rm -rf ~/.cache/modelscope/
```

3. 尝试使用阿里云内部镜像：

```bash
# 修改run_model_cn.py中的模型路径
# 将model_name = "qwen/Qwen-7B-Chat"改为
# model_name = "pai-models/qwen-7b-chat"
```

4. 如果您在PAI-DSW环境中，可以尝试使用预装的模型：

```bash
# 查看PAI-DSW环境中是否有预装模型
ls /root/share/models/
```

如果找到了预装的Qwen模型，可以直接使用：

```python
# 修改模型路径为预装模型路径
model_name = "/root/share/models/Qwen-7B-Chat"
```

### 6.3 JSON解析错误

**问题**：模型输出无法解析为JSON格式。

**解决方案**：

1. 修改提示词，更强调JSON格式的重要性：

```python
# 在提示词末尾添加
prompt += """
重要提醒：
1. 你的回答必须是有效的JSON格式
2. 不要包含任何额外的解释或注释
3. 确保所有引号、括号和逗号都正确匹配
"""
```

2. 使用我们已经准备好的JSON修复函数：

```python
# 我们已经为您准备了fix_json.py文件
cat fix_json.py
```

```python
import re
import json

def fix_json_string(json_str):
    """尝试修复常见的JSON格式错误"""
    # 移除可能的前缀和后缀文本
    json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)

    # 修复未转义的引号
    json_str = re.sub(r'(?<!\\)"(?=.*":)', r'\"', json_str)

    # 修复缺少逗号的情况
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)

    # 修复多余逗号
    json_str = re.sub(r',\s*}', '}', json_str)

    try:
        # 尝试解析修复后的JSON
        parsed = json.loads(json_str)
        return parsed
    except:
        # 如果仍然失败，返回None
        return None
```

3. 在主脚本中使用这个修复函数：

```python
# 在run_model_cn.py中导入修复函数
from fix_json import fix_json_string

# 修改JSON解析部分
try:
    # 使用正则表达式提取JSON部分
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试修复JSON
            result = fix_json_string(json_str)
            if result is None:
                raise
        results.append(result)
        print(f"成功处理: {item.get('title', '无标题')}")
    else:
        print(f"无法提取JSON: {item.get('title', '无标题')}")
        print(f"原始响应: {response}")
except Exception as e:
    print(f"处理错误: {str(e)}")
    print(f"原始响应: {response}")
```

### 6.4 运行时间过长

**问题**：处理大量样本需要很长时间。

**解决方案**：

1. 减少处理的样本数量，先测试一小部分：

```bash
python run_model_cn.py --samples 10
```

2. 使用更小的模型（如果可接受较低的性能）：

```python
# 使用更小的模型，如Qwen-1.8B
model_name = "qwen/Qwen-1.8B-Chat"
```

3. 将数据分批处理，每批处理一部分样本：

```bash
# 处理第1-10个样本
python run_model_cn.py --samples 10 --output batch1.json

# 处理第11-20个样本
# 可以创建一个新的测试数据文件，只包含第11-20个样本
head -n 20 test_data.json | tail -n 10 > test_data_11_20.json
python run_model_cn.py --input test_data_11_20.json --output batch2.json
```

## 7. 后续步骤

完成baseline运行后，您可以考虑以下后续步骤：

### 7.1 提高模型性能

1. **提示词工程**：优化提示词以获得更好的结果
2. **模型微调**：使用训练数据微调模型（需要更多资源）
3. **集成多个模型**：结合多个模型的输出以提高准确率

### 7.2 扩展到更多数据

1. 处理所有类型的古诗词数据
2. 创建更全面的评估指标
3. 分析不同类型诗词的性能差异

### 7.3 准备比赛提交

1. 确保输出格式符合比赛要求
2. 处理完整的测试集
3. 准备提交文件和说明文档

### 7.4 PAI-DSW环境特有问题

**问题**：在PAI-DSW环境中遇到特殊问题。

**解决方案**：

1. 如果遇到权限问题：

```bash
# 确保您有权限访问工作目录
chmod -R 755 /mnt/workspace/CAPA-private/try-baseline
```

2. 如果需要安装其他软件包：

```bash
# 使用apt安装系统软件包
apt update
apt install -y <软件包名称>

# 使用pip安装Python包
pip install <Python包名称>
```

3. 如果需要保存工作环境：

```bash
# 在PAI-DSW控制台中，点击"保存镜像"按钮
# 这将保存您的所有安装和配置
```

## 总结

恭喜！您已经成功在阿里云PAI-DSW上运行了Qwen模型的baseline，用于古诗词理解与推理评测任务。通过本指南，您学会了：

1. 登录和使用PAI-DSW平台
2. 准备代码和环境
3. 在中国环境下下载和使用大型语言模型
4. 运行推理脚本处理数据
5. 分析结果并解决常见问题

这些技能不仅适用于本次任务，也可以应用于其他自然语言处理项目。随着您对模型和数据的深入理解，您可以进一步优化性能，探索更多可能性。

祝您在古诗词理解与推理评测任务中取得好成绩！
