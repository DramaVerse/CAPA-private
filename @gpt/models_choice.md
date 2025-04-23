# 古诗词理解与推理评测任务模型选择指南

## 一、推荐模型概述

经过全面调研，以下是针对CCL 2025古诗词理解与推理评测任务的最佳开源模型推荐：

### 1. Qwen2.5系列
- **最新版本**：2024年9月发布
- **可选规模**：0.5B、1.5B、7B、14B、72B
- **主要优势**：
  - 在中文语言任务上表现卓越
  - 对文化和历史背景有深入理解
  - 能够很好地遵循复杂指令
  - 开源且许可灵活
- **适用原因**：最新的Qwen模型在理解包括古典文学在内的细微中文文本方面有显著提升。

### 2. ChatGLM4-9B
- **最新版本**：2024年6月发布
- **主要优势**：
  - 专为中文语言理解优化
  - 在对话和指令遵循方面表现良好
  - 硬件需求合理，运行效率高
  - 对中国文化背景理解深入
- **适用原因**：ChatGLM系列模型一直在中文语言任务上表现出色，最新版本延续了这一优势。

### 3. Yi-34B/Yi-6B
- **最新版本**：Yi系列有多个可用模型
- **主要优势**：
  - 强大的多语言能力，中文表现尤为出色
  - 效率和性能平衡良好
  - 开源且社区活跃
- **适用原因**：Yi模型在中文语言任务上表现良好，能够处理复杂推理。

### 4. Baichuan3
- **主要优势**：
  - 专为中文语言任务设计
  - 对文化背景理解深入
  - 微调能力高效
- **适用原因**：Baichuan模型针对中文语言理解进行了优化。

## 二、微调策略建议

### 1. 参数高效微调（PEFT）

#### QLoRA（量化低秩适应）
这可能是您的最佳微调方法，因为它：
- **显著降低GPU内存需求**：相比全参数微调，内存需求减少80-90%
- **保持接近全参数微调的性能**：在大多数任务上，性能损失很小
- **适合有限训练数据场景**：对于您的200个训练样例，能有效学习
- **硬件友好**：可在16GB-24GB消费级GPU上实施
- **实现简单**：有多个成熟框架支持（如LLaMA-Factory）

#### 具体参数建议
- **学习率**：1e-4至5e-4之间（建议从较小值开始，如2e-4）
- **训练轮次**：3-5轮（避免过拟合）
- **批量大小**：根据GPU内存调整，通常8-16
- **LoRA秩（rank）**：8-16（较小的模型可用较小的rank）
- **LoRA Alpha**：通常设为rank的2倍
- **LoRA应用层**：建议应用于注意力和前馈网络层

### 2. 数据增强策略

#### 现有数据集利用
- **古诗词数据库**：利用全唐诗、全宋词等公开数据集
- **古汉语语料库**：用于增强模型对古汉语的理解
- **注释和翻译资料**：收集已有的古诗词注释和翻译

#### 合成数据生成
- **使用更大模型生成样例**：可以使用更大的开源模型（如Qwen2-72B）生成额外训练样例
- **模板化生成**：基于现有样例的模式，生成结构相似的新样例
- **数据变体创建**：对现有样例进行小幅修改，创建变体

#### 质量控制
- **人工筛选**：对生成的数据进行人工检查，确保质量
- **一致性检查**：确保生成的数据与原始数据风格一致
- **难度平衡**：确保增强数据集包含不同难度级别的样例

### 3. 提示工程（Prompt Engineering）

#### 有效提示模板设计
- **任务明确说明**：清晰说明三个子任务的要求
- **输出格式示例**：提供期望的JSON输出格式示例
- **古诗词背景信息**：包含相关的文化和历史背景
- **思维链引导**：引导模型进行逐步推理

#### 示例提示模板
```python
你是一位古诗词专家，请解析以下诗词：

标题：{title}
作者：{author}
内容：{content}

请完成以下任务：

1. 解释以下词语的含义（请详细解释每个词的本义和在诗中的含义）：
{qa_words}

2. 将以下句子翻译成现代白话文（保持原意，使表达流畅自然）：
{qa_sents}

3. 分析这首诗表达的主要情感，并从以下选项中选择最贴切的一项：
{choose}

请按照以下JSON格式输出你的回答：
{
    "ans_qa_words": {
        "词语1": "释义1",
        "词语2": "释义2"
    },
    "ans_qa_sents": {
        "句子1": "白话翻译1",
        "句子2": "白话翻译2"
    },
    "choose_id": "A"
}
```

## 三、实现方案建议

### 1. 框架选择

#### LLaMA-Factory
- **主要优势**：
  - 统一的框架，支持多种LLM的微调
  - 支持所有推荐的模型（Qwen、ChatGLM、Yi等）
  - 高效实现QLoRA和其他PEFT方法
  - 文档完善，社区支持良好
  - 持续更新，跟进最新技术

#### 实现步骤
1. **安装环境**：
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

2. **准备数据**：将训练数据转换为以下格式：
```json
[
  {
    "instruction": "解析以下古诗词...",
    "input": "标题：登鹳雀楼\n作者：王之涣\n内容：白日依山尽，黄河入海流。欲穷千里目，更上一层楼。\n请解释以下词语：[\"依山\", \"穷\"]\n请翻译以下句子：[\"白日依山尽\", \"黄河入海流\"]\n请选择情感：{\"A\": \"豪迈\", \"B\": \"忧伤\"}",
    "output": "{\"ans_qa_words\":{\"依山\":\"靠着山，指太阳靠近西山\",\"穷\":\"穷尽，看到尽头\"},\"ans_qa_sents\":{\"白日依山尽\":\"夕阳靠近西山即将落下\",\"黄河入海流\":\"黄河水滚滚流向大海\"},\"choose_id\":\"A\"}"
  },
  // 更多样例...
]
```

3. **配置微调**：创建配置文件，指定模型和参数：
```yaml
model_name_or_path: "Qwen/Qwen2-7B"
data_path: "./data/poetry_dataset.json"
output_dir: "./output/qwen2-7b-poetry"

# QLoRA参数
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

# 训练参数
num_train_epochs: 3
per_device_train_batch_size: 4
learning_rate: 2e-4
weight_decay: 0.01
warmup_ratio: 0.03
```

4. **执行微调**：
```bash
llmtuner chat \
  --model_name_or_path Qwen/Qwen2-7B \
  --dataset poetry_dataset \
  --template qwen \
  --finetuning_type lora \
  --output_dir ./output/qwen2-7b-poetry
```

### 2. 硬件需求

#### 推荐配置
- **对于Qwen2-7B或ChatGLM4-9B（使用QLoRA）**：
  - GPU: NVIDIA RTX 3090/4090（24GB VRAM）
  - RAM: 32GB以上
  - 存储: 100GB SSD（用于模型和数据）

- **对于较小模型（Qwen2-1.5B）**：
  - GPU: NVIDIA RTX 3060/4060（16GB VRAM）可能足够
  - RAM: 16GB以上
  - 存储: 50GB SSD

#### 云服务选项
如果本地资源有限，可考虑以下云GPU服务：
- **国内服务**：
  - 阿里云PAI-DSW（支持A10、V100等）
  - 腾讯云GPU实例
  - 百度AI Studio

- **国际服务**（如果可访问）：
  - Google Colab Pro+（提供A100，但连接不稳定）
  - Lambda Labs（按小时计费，性价比高）
  - Vast.ai（社区GPU共享平台，价格较低）

### 3. 评估策略

#### 指标实现
- **BLEU值计算**：
```python
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    # 分词
    ref_tokens = list(reference)
    cand_tokens = list(candidate)
    # 计算BLEU
    return sentence_bleu([ref_tokens], cand_tokens)
```

- **中文BertScore**：
```python
from bert_score import score

def calculate_bertscore(references, candidates):
    P, R, F1 = score(candidates, references, lang="zh", rescale_with_baseline=True)
    return F1.numpy()
```

#### 验证流程
1. **格式验证**：确保输出符合要求的JSON格式
2. **完整性检查**：验证所有必需字段都已填写
3. **交叉验证**：使用不同模型和参数的结果进行比较

#### 超参数优化
- 使用小型验证集测试不同学习率（1e-4至5e-4）
- 尝试不同的LoRA秩（8、16、32）
- 测试不同的训练轮次（2-5轮）
- 记录每组参数的性能，选择最佳配置

## 四、结果优化策略

### 1. 输出格式规范化

#### JSON格式处理
- **后处理脚本**：确保输出符合要求的JSON格式
```python
import json
import re

def format_output(raw_output):
    # 提取JSON部分
    json_pattern = r'\{[\s\S]*\}'
    json_match = re.search(json_pattern, raw_output)

    if not json_match:
        return create_fallback_response()

    try:
        # 解析JSON
        json_str = json_match.group(0)
        result = json.loads(json_str)

        # 验证必要字段
        required_fields = ['ans_qa_words', 'ans_qa_sents', 'choose_id']
        for field in required_fields:
            if field not in result:
                result[field] = {} if field != 'choose_id' else 'A'

        # 格式化输出
        return json.dumps(result, ensure_ascii=False, indent=2)
    except:
        return create_fallback_response()
```

#### 空值处理
- **默认值策略**：对于模型未能给出答案的情况，提供合理的默认值
```python
def create_fallback_response(qa_words=None, qa_sents=None):
    # 创建基本结构
    response = {
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": "A"  # 默认选择
    }

    # 填充词语解释
    if qa_words:
        for word in qa_words:
            response["ans_qa_words"][word] = "该词在古汉语中指..."

    # 填充句子翻译
    if qa_sents:
        for sent in qa_sents:
            response["ans_qa_sents"][sent] = "该句的白话文意思是..."

    return json.dumps(response, ensure_ascii=False, indent=2)
```

### 2. 多模型集成策略

#### 投票机制
- **多模型结果整合**：使用多个模型生成结果，通过投票选出最佳答案
```python
def ensemble_results(model_outputs, qa_words, qa_sents):
    # 初始化结果容器
    word_explanations = {word: {} for word in qa_words}
    sent_translations = {sent: {} for sent in qa_sents}
    emotion_votes = {}

    # 收集所有模型的结果
    for output in model_outputs:
        try:
            result = json.loads(output)

            # 词语解释投票
            for word in qa_words:
                if word in result["ans_qa_words"]:
                    explanation = result["ans_qa_words"][word]
                    word_explanations[word][explanation] = word_explanations[word].get(explanation, 0) + 1

            # 句子翻译投票
            for sent in qa_sents:
                if sent in result["ans_qa_sents"]:
                    translation = result["ans_qa_sents"][sent]
                    sent_translations[sent][translation] = sent_translations[sent].get(translation, 0) + 1

            # 情感分类投票
            emotion = result["choose_id"]
            emotion_votes[emotion] = emotion_votes.get(emotion, 0) + 1
        except:
            continue

    # 整合结果
    final_result = {
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": max(emotion_votes.items(), key=lambda x: x[1])[0] if emotion_votes else "A"
    }

    # 选择最高票数的解释和翻译
    for word in qa_words:
        explanations = word_explanations[word]
        if explanations:
            final_result["ans_qa_words"][word] = max(explanations.items(), key=lambda x: x[1])[0]

    for sent in qa_sents:
        translations = sent_translations[sent]
        if translations:
            final_result["ans_qa_sents"][sent] = max(translations.items(), key=lambda x: x[1])[0]

    return final_result
```

#### 编辑距离优化
- **相似答案合并**：对于表达相似的答案，进行合并以提高一致性
```python
from difflib import SequenceMatcher

def merge_similar_answers(answers, threshold=0.8):
    # 合并相似的答案
    merged = {}
    for answer in answers:
        found_similar = False
        for key in merged:
            similarity = SequenceMatcher(None, answer, key).ratio()
            if similarity >= threshold:
                # 如果找到相似答案，增加计数
                merged[key] += answers[answer]
                found_similar = True
                break
        if not found_similar:
            # 如果没有相似答案，添加新条目
            merged[answer] = answers[answer]
    return merged
```

### 3. 错误处理机制

#### 多次重试策略
- **渐进式提示**：当模型输出格式不正确时，使用更具体的提示重试
```python
def retry_with_progressive_prompts(model, tokenizer, data, max_retries=3):
    # 初始提示
    prompt = create_base_prompt(data)

    for attempt in range(max_retries):
        try:
            # 生成答案
            response = generate_response(model, tokenizer, prompt)

            # 验证格式
            result = validate_json_format(response)
            if result:
                return result

            # 如果格式不正确，增强提示
            if attempt == 0:
                prompt = create_format_emphasis_prompt(data)
            elif attempt == 1:
                prompt = create_step_by_step_prompt(data)
            else:
                prompt = create_detailed_example_prompt(data)
        except Exception as e:
            print(f"重试原因: {e}")

    # 所有重试失败后返回默认结果
    return create_fallback_response(data["qa_words"], data["qa_sents"])
```

## 五、总结与建议

### 1. 最佳模型选择

基于当前调研和任务特点，推荐以下模型选择策略：

#### 首选方案：Qwen2-7B + QLoRA
- **优势**：
  - 在中文古文理解上有出色表现
  - 模型大小适中，平衡性能和资源需求
  - 微调效果好，对有限数据敏感
  - 社区支持度高，文档完善

#### 备选方案：ChatGLM4-9B + QLoRA
- **优势**：
  - 专门针对中文任务优化
  - 对中国文化背景理解深入
  - 对语义理解和情感分析有优势

#### 资源受限方案：Qwen2-1.5B + QLoRA
- **优势**：
  - 资源需求低，可在普通GPU上运行
  - 微调速度快，允许更多实验迭代
  - 作为基线模型或集成系统的一部分

### 2. 实施路线图

#### 第一阶段：基础准备（1-2周）
1. 环境搭建与数据准备
   - 安装必要库和框架
   - 整理训练数据为标准格式
   - 收集补充数据集

2. 基线模型测试
   - 下载并测试各个候选模型
   - 评估原始模型在样例数据上的表现
   - 确定最终模型选择

#### 第二阶段：模型微调（2-3周）
1. 数据增强
   - 生成合成数据
   - 数据清洗与质量控制
   - 准备验证集

2. QLoRA微调
   - 调整学习率、批量大小等超参数
   - 进行多组实验对比
   - 选择最佳模型检查点

#### 第三阶段：结果优化（1-2周）
1. 输出格式处理
   - 实现JSON格式规范化
   - 开发错误处理机制
   - 测试各种边界情况

2. 多模型集成（可选）
   - 实现投票机制
   - 测试集成效果
   - 优化集成策略

#### 第四阶段：测试与提交（1周）
1. 全面测试
   - 在验证集上进行评估
   - 检查输出格式合规性
   - 解决最后的问题

2. 准备提交
   - 整理最终结果
   - 准备提交文件
   - 记录实验过程与结果

### 3. 注意事项与风险防范

#### 潜在风险
- **数据泄露**：确保不使用测试集数据进行训练
- **过拟合**：由于训练数据有限，需谨慎控制训练轮次
- **输出格式错误**：必须确保输出符合要求的JSON格式

#### 防范措施
- 使用交叉验证防止过拟合
- 实现强大的格式验证和错误处理
- 保存所有实验记录，便于复现和排错
- 定期备份模型和数据

通过以上全面的模型选择、微调策略和实施计划，您将能够有效地完成CCL 2025古诗词理解与推理评测任务，并获得竞争力的结果。
