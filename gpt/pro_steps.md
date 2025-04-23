# 古诗词理解与推理评测任务高级实施指南

本指南是对基础方案的升级版本，假设您已经完成了基础环境配置和初步实验。本指南将引导您实施一个更具竞争力的解决方案，包括多阶段微调、模型集成和高级数据处理技术。

## 目录

1. [项目结构与代码组织](#1-项目结构与代码组织)
2. [高级数据处理与增强](#2-高级数据处理与增强)
3. [多阶段微调策略](#3-多阶段微调策略)
4. [模型集成与推理](#4-模型集成与推理)
5. [结果优化与后处理](#5-结果优化与后处理)
6. [提交与验证](#6-提交与验证)

## 1. 项目结构与代码组织

首先，让我们创建一个更加模块化的项目结构，以支持更复杂的训练流程：

```
poetry-pro/
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   ├── augmented/          # 增强数据
│   └── final/              # 最终训练数据
├── models/
│   ├── domain_pretrain/    # 领域预训练模型
│   ├── instruction_tune/   # 指令微调模型
│   ├── preference_align/   # 偏好对齐模型
│   └── ensemble/           # 集成模型
├── src/
│   ├── data/               # 数据处理代码
│   │   ├── process.py
│   │   ├── augment.py
│   │   └── dataset.py
│   ├── training/           # 训练代码
│   │   ├── domain_pretrain.py
│   │   ├── instruction_tune.py
│   │   ├── preference_align.py
│   │   └── trainer.py
│   ├── models/             # 模型代码
│   │   ├── modeling.py
│   │   ├── integration.py
│   │   └── ensemble.py
│   ├── inference/          # 推理代码
│   │   ├── pipeline.py
│   │   ├── postprocess.py
│   │   └── ensemble.py
│   └── utils/              # 工具函数
│       ├── metrics.py
│       ├── logging.py
│       └── config.py
├── configs/                # 配置文件
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── train_config.yaml
│   └── inference_config.yaml
├── scripts/                # 脚本文件
│   ├── prepare_data.sh
│   ├── train_all.sh
│   ├── inference.sh
│   └── submit.sh
├── notebooks/              # 分析笔记本
│   ├── data_analysis.ipynb
│   ├── model_analysis.ipynb
│   └── results_analysis.ipynb
├── requirements.txt        # 依赖项
└── README.md               # 项目说明
```

### 创建项目结构

```bash
# 创建主要目录
mkdir -p poetry-pro/{data/{raw,processed,augmented,final},models/{domain_pretrain,instruction_tune,preference_align,ensemble},src/{data,training,models,inference,utils},configs,scripts,notebooks}

# 复制原始数据
cp -r ~/poetry-llm-project/data/* poetry-pro/data/raw/

# 进入项目目录
cd poetry-pro
```

### 安装额外依赖

```bash
# 创建requirements.txt文件
cat > requirements.txt << EOF
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
trl>=0.7.4
sentencepiece>=0.1.99
protobuf>=4.24.4
tensorboard>=2.14.0
wandb>=0.16.0
jsonlines>=3.1.0
nltk>=3.8.1
bert_score>=0.3.13
jieba>=0.42.1
rouge>=1.0.1
sacrebleu>=2.3.1
EOF

# 安装依赖
pip install -r requirements.txt
```

## 2. 高级数据处理与增强

在这一部分，我们将实现更复杂的数据处理和增强策略，以提高模型性能。

### 2.1 数据收集与整合

首先，我们需要收集更多的古诗词数据来增强我们的训练集：

```python
# src/data/collect.py
import os
import json
import requests
from tqdm import tqdm

def download_chinese_poetry_corpus():
    """下载中文古诗词语料库"""
    sources = {
        "chinese-poetry": [
            "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/json/poet.tang.0.json",
            "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/json/poet.tang.1.json",
            "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/json/poet.song.0.json",
            "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/json/poet.song.1.json"
        ],
        "couplet-dataset": [
            "https://raw.githubusercontent.com/wb14123/couplet-dataset/master/train/in.txt",
            "https://raw.githubusercontent.com/wb14123/couplet-dataset/master/train/out.txt"
        ]
    }

    os.makedirs("data/raw/external", exist_ok=True)

    for source_name, urls in sources.items():
        os.makedirs(f"data/raw/external/{source_name}", exist_ok=True)

        for i, url in enumerate(urls):
            try:
                print(f"下载 {url}...")
                response = requests.get(url)
                if response.status_code == 200:
                    filename = url.split("/")[-1]
                    with open(f"data/raw/external/{source_name}/{filename}", "w", encoding="utf-8") as f:
                        f.write(response.text)
                    print(f"下载完成: {filename}")
                else:
                    print(f"下载失败: {url}, 状态码: {response.status_code}")
            except Exception as e:
                print(f"下载出错: {url}, 错误: {e}")

def integrate_poetry_data():
    """整合所有诗词数据"""
    # 加载原始比赛数据
    with open("data/raw/ccl_poetry_train.json", "r", encoding="utf-8") as f:
        competition_data = json.load(f)

    print(f"加载了 {len(competition_data)} 条比赛数据")

    # 加载外部数据
    external_data = []

    # 加载全唐诗数据
    tang_poetry_files = [f for f in os.listdir("data/raw/external/chinese-poetry") if f.startswith("poet.tang")]
    for file in tang_poetry_files:
        with open(f"data/raw/external/chinese-poetry/{file}", "r", encoding="utf-8") as f:
            tang_data = json.load(f)
            external_data.extend(tang_data)

    print(f"加载了 {len(external_data)} 条外部数据")

    # 保存整合后的数据
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/integrated_data.json", "w", encoding="utf-8") as f:
        json.dump({
            "competition_data": competition_data,
            "external_data": external_data
        }, f, ensure_ascii=False, indent=2)

    print("数据整合完成")

if __name__ == "__main__":
    download_chinese_poetry_corpus()
    integrate_poetry_data()
```

运行数据收集脚本：

```bash
python -m src.data.collect
```

### 2.2 高级数据增强

我们将实现三种高级数据增强方法：专家知识注入、对比学习样本构建和多样性增强。

```python
# src/data/augment.py
import os
import json
import random
import jieba
import numpy as np
from tqdm import tqdm

def load_expert_knowledge():
    """加载专家知识库（这里模拟一个简单的知识库）"""
    # 在实际应用中，这应该是一个从专业教材或词典中提取的大型知识库
    return {
        "诗词常用词语": {
            "春风": "指春天的风，常用来表示暖意和生机",
            "秋月": "指秋天的月亮，常用来表示清凉、明朗或思乡",
            "江南": "指长江以南的地区，常指现在的江苏、浙江一带",
            "山水": "指山和水，常用来指代自然风光或国家江山",
            # ... 更多词语
        },
        "古诗词常用句式": {
            "白日依山尽": "太阳落山，天色将暗",
            "黄河入海流": "黄河水流入海，永不停息",
            "小桥流水人家冬": "小桥旁流水，人家在冬天",
            # ... 更多句式
        },
        "古诗词情感分类": {
            "思乡": ["游子夕归", "游子身上衛", "遥瞻故园"],
            "送别": ["无可奈何花落去", "一壶浊酒尽余欢", "劫天地灵气"],
            "爱国": ["万里长城", "报国一刀", "民族英雄"],
            # ... 更多情感分类
        }
    }

def create_expert_enhanced_samples(competition_data, expert_knowledge):
    """使用专家知识增强样本"""
    enhanced_samples = []

    for item in tqdm(competition_data, desc="创建专家增强样本"):
        # 复制原始样本
        enhanced_item = item.copy()

        # 增强词语解释
        if 'qa_words' in item and 'keywords' in item:
            for word in item['qa_words']:
                if word in expert_knowledge["诗词常用词语"]:
                    # 使用专家知识库中的解释
                    enhanced_item['keywords'][word] = expert_knowledge["诗词常用词语"][word]

        # 增强句子翻译
        if 'qa_sents' in item and 'trans' in item:
            for sent in item['qa_sents']:
                if sent in expert_knowledge["古诗词常用句式"]:
                    # 将专家知识库中的翻译添加到原始翻译中
                    if not enhanced_item.get('trans_enhanced'):
                        enhanced_item['trans_enhanced'] = enhanced_item['trans']
                    enhanced_item['trans_enhanced'] = enhanced_item['trans_enhanced'].replace(
                        sent, expert_knowledge["古诗词常用句式"][sent]
                    )

        enhanced_samples.append(enhanced_item)

    return enhanced_samples

def create_contrastive_samples(competition_data):
    """创建对比学习样本（好-差对）"""
    contrastive_pairs = []

    for item in tqdm(competition_data, desc="创建对比学习样本"):
        if 'keywords' in item and 'trans' in item and 'emotion' in item:
            # 创建“好”的样本（使用原始数据）
            good_sample = {
                "title": item['title'],
                "author": item['author'],
                "content": item['content'],
                "qa_words": list(item['keywords'].keys()),
                "qa_sents": [item['content'].split('\n')[0]],  # 简化处理，取第一句
                "choose": {"A": item['emotion']},
                "ans_qa_words": item['keywords'],
                "ans_qa_sents": {item['content'].split('\n')[0]: item['trans']},
                "choose_id": "A"
            }

            # 创建“差”的样本（故意降低质量）
            bad_sample = good_sample.copy()

            # 降低词语解释质量
            bad_ans_words = {}
            for word, explanation in item['keywords'].items():
                # 简化解释，去除部分内容
                simplified_exp = explanation.split('，')[0] if '，' in explanation else explanation
                bad_ans_words[word] = simplified_exp[:len(simplified_exp)//2] + "..."

            # 降低句子翻译质量
            first_line = item['content'].split('\n')[0]
            bad_trans = item['trans'][:len(item['trans'])//3] + "..."

            bad_sample["ans_qa_words"] = bad_ans_words
            bad_sample["ans_qa_sents"] = {first_line: bad_trans}

            # 随机选择错误的情感
            emotions = ["思乡", "送别", "爱国", "写景", "哀思"]
            wrong_emotion = random.choice([e for e in emotions if e != item['emotion']])
            bad_sample["choose"] = {"A": wrong_emotion}

            contrastive_pairs.append((good_sample, bad_sample))

    return contrastive_pairs

def create_diverse_samples(external_data, competition_format=True):
    """从外部数据创建多样性样本"""
    diverse_samples = []

    # 常见情感类别
    emotions = ["思乡", "送别", "爱国", "写景", "哀思", "节日", "官场", "隔等", "山水"]

    # 随机选择一部分外部数据
    selected_data = random.sample(external_data, min(500, len(external_data)))

    for item in tqdm(selected_data, desc="创建多样性样本"):
        if 'paragraphs' in item and len(item['paragraphs']) > 1:
            content = '\n'.join(item['paragraphs'])
            title = item.get('title', '无题')
            author = item.get('author', '佚名')

            # 分词并选择可能需要解释的词语
            words = list(jieba.cut(content))
            words = [w for w in words if len(w) >= 2]  # 只选择双字及以上的词

            if len(words) >= 3:
                # 随机选择词语进行解释
                selected_words = random.sample(words, min(3, len(words)))

                # 随机选择句子进行翻译
                selected_sents = random.sample(item['paragraphs'], min(2, len(item['paragraphs'])))

                # 随机选择情感类别
                emotion = random.choice(emotions)

                if competition_format:
                    # 转换为比赛格式
                    diverse_sample = {
                        "title": title,
                        "author": author,
                        "content": content,
                        "qa_words": selected_words,
                        "qa_sents": selected_sents,
                        "choose": {"A": emotion, "B": random.choice([e for e in emotions if e != emotion])}
                    }
                else:
                    # 保持原始格式
                    diverse_sample = {
                        "title": title,
                        "author": author,
                        "content": content,
                        "keywords": {word: f"{word}的含义" for word in selected_words},
                        "trans": f"{content}的白话文译文",
                        "emotion": emotion
                    }

                diverse_samples.append(diverse_sample)

    return diverse_samples

def main():
    # 加载数据
    with open("data/processed/integrated_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    competition_data = data["competition_data"]
    external_data = data["external_data"]

    # 加载专家知识
    expert_knowledge = load_expert_knowledge()

    # 1. 专家知识注入
    expert_enhanced_samples = create_expert_enhanced_samples(competition_data, expert_knowledge)
    print(f"创建了 {len(expert_enhanced_samples)} 个专家增强样本")

    # 2. 对比学习样本构建
    contrastive_pairs = create_contrastive_samples(competition_data)
    print(f"创建了 {len(contrastive_pairs)} 对对比学习样本")

    # 3. 多样性增强
    diverse_samples = create_diverse_samples(external_data)
    print(f"创建了 {len(diverse_samples)} 个多样性样本")

    # 保存增强数据
    os.makedirs("data/augmented", exist_ok=True)

    with open("data/augmented/expert_enhanced_samples.json", "w", encoding="utf-8") as f:
        json.dump(expert_enhanced_samples, f, ensure_ascii=False, indent=2)

    with open("data/augmented/contrastive_pairs.json", "w", encoding="utf-8") as f:
        json.dump(contrastive_pairs, f, ensure_ascii=False, indent=2)

    with open("data/augmented/diverse_samples.json", "w", encoding="utf-8") as f:
        json.dump(diverse_samples, f, ensure_ascii=False, indent=2)

    print("数据增强完成")

if __name__ == "__main__":
    main()
```

运行数据增强脚本：

```bash
python -m src.data.augment
```

### 2.3 数据转换为训练格式

最后，我们需要将所有数据转换为适合多阶段训练的格式：

```python
# src/data/prepare.py
import os
import json
import random
from tqdm import tqdm

def prepare_domain_pretraining_data(external_data):
    """准备领域预训练数据"""
    pretraining_samples = []

    for item in tqdm(external_data, desc="准备领域预训练数据"):
        if 'paragraphs' in item and len(item['paragraphs']) > 0:
            content = '\n'.join(item['paragraphs'])
            # 简单的文本补全任务
            pretraining_samples.append({
                "text": content
            })

    return pretraining_samples

def prepare_instruction_tuning_data(competition_data, expert_enhanced_samples, diverse_samples):
    """准备多任务指令微调数据"""
    # 分别为三个子任务准备数据
    word_task_samples = []
    sent_task_samples = []
    emotion_task_samples = []

    # 合并所有数据源
    all_samples = competition_data + expert_enhanced_samples + diverse_samples

    for item in tqdm(all_samples, desc="准备指令微调数据"):
        # 词语解释任务
        if 'qa_words' in item and ('keywords' in item or 'ans_qa_words' in item):
            keywords = item.get('keywords', item.get('ans_qa_words', {}))
            if keywords and len(item['qa_words']) > 0:
                word_sample = {
                    "instruction": "你是一位古诗词专家，请解释以下古诗词中的词语。",
                    "input": f"标题：{item.get('title', '')}"
                             f"\n作者：{item.get('author', '')}"
                             f"\n内容：{item.get('content', '')}"
                             f"\n请解释以下词语：{', '.join(item['qa_words'])}",
                    "output": json.dumps({"ans_qa_words": {word: keywords.get(word, "") for word in item['qa_words']}}, ensure_ascii=False),
                    "task_type": "word_explanation"
                }
                word_task_samples.append(word_sample)

        # 句子翻译任务
        if 'qa_sents' in item and ('trans' in item or 'ans_qa_sents' in item):
            trans = item.get('trans', "")
            ans_sents = item.get('ans_qa_sents', {})

            if len(item['qa_sents']) > 0:
                # 如果有直接的句子翻译，使用它
                if ans_sents:
                    translations = ans_sents
                # 否则尝试从全文翻译中提取
                elif trans:
                    translations = {}
                    for sent in item['qa_sents']:
                        # 简化处理，假设翻译中包含原句的一部分
                        for part in trans.split('，'):
                            if any(w in part for w in sent if len(w) >= 2):
                                translations[sent] = part
                                break
                        # 如果没有找到匹配，使用空字符串
                        if sent not in translations:
                            translations[sent] = ""

                sent_sample = {
                    "instruction": "你是一位古诗词专家，请将以下古诗词中的句子翻译成现代白话文。",
                    "input": f"标题：{item.get('title', '')}"
                             f"\n作者：{item.get('author', '')}"
                             f"\n内容：{item.get('content', '')}"
                             f"\n请翻译以下句子：{', '.join(item['qa_sents'])}",
                    "output": json.dumps({"ans_qa_sents": translations}, ensure_ascii=False),
                    "task_type": "sentence_translation"
                }
                sent_task_samples.append(sent_sample)

        # 情感分类任务
        if 'choose' in item and ('emotion' in item or 'choose_id' in item):
            choose_options = item['choose']
            correct_id = item.get('choose_id', "")

            # 如果没有正确答案ID但有情感标签
            if not correct_id and 'emotion' in item:
                # 找到匹配的情感选项
                for id, emotion in choose_options.items():
                    if emotion == item['emotion']:
                        correct_id = id
                        break

            if correct_id:
                emotion_sample = {
                    "instruction": "你是一位古诗词专家，请分析以下古诗词表达的主要情感。",
                    "input": f"标题：{item.get('title', '')}"
                             f"\n作者：{item.get('author', '')}"
                             f"\n内容：{item.get('content', '')}"
                             f"\n请选择这首诗表达的主要情感：{json.dumps(choose_options, ensure_ascii=False)}",
                    "output": json.dumps({"choose_id": correct_id}, ensure_ascii=False),
                    "task_type": "emotion_classification"
                }
                emotion_task_samples.append(emotion_sample)

    return {
        "word_task": word_task_samples,
        "sent_task": sent_task_samples,
        "emotion_task": emotion_task_samples
    }

def prepare_preference_alignment_data(contrastive_pairs):
    """准备偏好对齐数据（DPO格式）"""
    preference_samples = []

    for good_sample, bad_sample in tqdm(contrastive_pairs, desc="准备偏好对齐数据"):
        # 构建输入提示
        prompt = f"标题：{good_sample.get('title', '')}"
                f"\n作者：{good_sample.get('author', '')}"
                f"\n内容：{good_sample.get('content', '')}"
                f"\n请解释以下词语：{', '.join(good_sample.get('qa_words', []))}"
                f"\n请翻译以下句子：{', '.join(good_sample.get('qa_sents', []))}"
                f"\n请选择这首诗表达的主要情感：{json.dumps(good_sample.get('choose', {}), ensure_ascii=False)}"

        # 构建好的回答
        good_response = json.dumps({
            "ans_qa_words": good_sample.get('ans_qa_words', {}),
            "ans_qa_sents": good_sample.get('ans_qa_sents', {}),
            "choose_id": good_sample.get('choose_id', "")
        }, ensure_ascii=False)

        # 构建差的回答
        bad_response = json.dumps({
            "ans_qa_words": bad_sample.get('ans_qa_words', {}),
            "ans_qa_sents": bad_sample.get('ans_qa_sents', {}),
            "choose_id": bad_sample.get('choose_id', "")
        }, ensure_ascii=False)

        preference_samples.append({
            "prompt": prompt,
            "chosen": good_response,
            "rejected": bad_response
        })

    return preference_samples

def split_data(data, train_ratio=0.9):
    """将数据分割为训练集和验证集"""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

def main():
    # 加载原始数据
    with open("data/processed/integrated_data.json", "r", encoding="utf-8") as f:
        integrated_data = json.load(f)

    competition_data = integrated_data["competition_data"]
    external_data = integrated_data["external_data"]

    # 加载增强数据
    with open("data/augmented/expert_enhanced_samples.json", "r", encoding="utf-8") as f:
        expert_enhanced_samples = json.load(f)

    with open("data/augmented/contrastive_pairs.json", "r", encoding="utf-8") as f:
        contrastive_pairs = json.load(f)

    with open("data/augmented/diverse_samples.json", "r", encoding="utf-8") as f:
        diverse_samples = json.load(f)

    # 1. 准备领域预训练数据
    pretraining_samples = prepare_domain_pretraining_data(external_data)
    print(f"准备了 {len(pretraining_samples)} 条领域预训练数据")

    # 2. 准备多任务指令微调数据
    instruction_data = prepare_instruction_tuning_data(
        competition_data, expert_enhanced_samples, diverse_samples
    )
    print(f"准备了 {len(instruction_data['word_task'])} 条词语解释数据")
    print(f"准备了 {len(instruction_data['sent_task'])} 条句子翻译数据")
    print(f"准备了 {len(instruction_data['emotion_task'])} 条情感分类数据")

    # 3. 准备偏好对齐数据
    preference_samples = prepare_preference_alignment_data(contrastive_pairs)
    print(f"准备了 {len(preference_samples)} 条偏好对齐数据")

    # 分割数据
    pretraining_train, pretraining_val = split_data(pretraining_samples)

    word_task_train, word_task_val = split_data(instruction_data['word_task'])
    sent_task_train, sent_task_val = split_data(instruction_data['sent_task'])
    emotion_task_train, emotion_task_val = split_data(instruction_data['emotion_task'])

    preference_train, preference_val = split_data(preference_samples)

    # 保存最终训练数据
    os.makedirs("data/final/domain_pretrain", exist_ok=True)
    os.makedirs("data/final/instruction_tune", exist_ok=True)
    os.makedirs("data/final/preference_align", exist_ok=True)

    # 保存领域预训练数据
    with open("data/final/domain_pretrain/train.json", "w", encoding="utf-8") as f:
        json.dump(pretraining_train, f, ensure_ascii=False, indent=2)

    with open("data/final/domain_pretrain/val.json", "w", encoding="utf-8") as f:
        json.dump(pretraining_val, f, ensure_ascii=False, indent=2)

    # 保存指令微调数据
    with open("data/final/instruction_tune/word_task_train.json", "w", encoding="utf-8") as f:
        json.dump(word_task_train, f, ensure_ascii=False, indent=2)

    with open("data/final/instruction_tune/word_task_val.json", "w", encoding="utf-8") as f:
        json.dump(word_task_val, f, ensure_ascii=False, indent=2)

    with open("data/final/instruction_tune/sent_task_train.json", "w", encoding="utf-8") as f:
        json.dump(sent_task_train, f, ensure_ascii=False, indent=2)

    with open("data/final/instruction_tune/sent_task_val.json", "w", encoding="utf-8") as f:
        json.dump(sent_task_val, f, ensure_ascii=False, indent=2)

    with open("data/final/instruction_tune/emotion_task_train.json", "w", encoding="utf-8") as f:
        json.dump(emotion_task_train, f, ensure_ascii=False, indent=2)

    with open("data/final/instruction_tune/emotion_task_val.json", "w", encoding="utf-8") as f:
        json.dump(emotion_task_val, f, ensure_ascii=False, indent=2)

    # 保存偏好对齐数据
    with open("data/final/preference_align/train.json", "w", encoding="utf-8") as f:
        json.dump(preference_train, f, ensure_ascii=False, indent=2)

    with open("data/final/preference_align/val.json", "w", encoding="utf-8") as f:
        json.dump(preference_val, f, ensure_ascii=False, indent=2)

    print("数据准备完成")

if __name__ == "__main__":
    main()
```

运行数据准备脚本：

```bash
python -m src.data.prepare
```

完成所有数据准备后，我们可以创建一个一键数据准备脚本：

```bash
# scripts/prepare_data.sh
#!/bin/bash

echo "开始数据准备流程..."

# 1. 数据收集
与整合
echo "步骤1: 数据收集与整合"
python -m src.data.collect

# 2. 数据增强
echo "步骤2: 数据增强"
python -m src.data.augment

# 3. 数据准备
echo "步骤3: 数据准备"
python -m src.data.prepare

echo "数据准备完成!"
```

运行一键数据准备脚本：

```bash
chmod +x scripts/prepare_data.sh
./scripts/prepare_data.sh
```

## 3. 多阶段微调策略

在这一部分，我们将实现三阶段微调策略：领域预训练、多任务指令微调和人类偏好对齐。

### 3.1 领域预训练

首先，我们将对基础模型进行领域预训练，以增强其对古汉语语法、词汇和文化背景的理解。

```python
# src/training/domain_pretrain.py
import os
import json
import torch
import logging
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """加载预训练数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 将数据转换为数据集格式
    return Dataset.from_dict({"text": [item["text"] for item in data]})

def train_domain_adaptation(config):
    """进行领域适应性预训练"""
    # 加载数据
    train_dataset = load_data(config["train_data_path"])
    val_dataset = load_data(config["val_data_path"])

    logger.info(f"加载了 {len(train_dataset)} 条训练数据和 {len(val_dataset)} 条验证数据")

    # 加载模型和分词器
    logger.info(f"加载模型: {config['model_name_or_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=True
    )

    # 确保分词器有正确的padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备模型进行量化训练
    model = prepare_model_for_kbit_training(model)

    # 配置LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 定义数据处理函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config["max_length"])

    # 处理数据集
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=10,
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard"
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # 开始训练
    logger.info("开始领域预训练...")
    trainer.train()

    # 保存模型
    logger.info(f"保存模型到 {config['output_dir']}")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    return model, tokenizer

def main():
    # 训练配置
    config = {
        "model_name_or_path": "Qwen/Qwen2.5-7B",  # 或者 "THUDM/chatglm4-9b"
        "train_data_path": "data/final/domain_pretrain/train.json",
        "val_data_path": "data/final/domain_pretrain/val.json",
        "output_dir": "models/domain_pretrain",
        "num_epochs": 3,
        "batch_size": 4,
        "max_length": 1024,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gradient_accumulation_steps": 8,
        "save_steps": 500,
        "eval_steps": 500
    }

    # 进行领域预训练
    train_domain_adaptation(config)

if __name__ == "__main__":
    main()
```

运行领域预训练脚本：

```bash
python -m src.training.domain_pretrain
```

### 3.2 多任务指令微调

第二阶段，我们将实现多任务指令微调，将任务分解为词语解释、句子翻译和情感分类三个子任务。

```python
# src/training/instruction_tune.py
import os
import json
import torch
import logging
import random
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 自定义多任务训练器
class MultiTaskTrainer(Trainer):
    def __init__(self, task_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights or {
            "word_explanation": 0.35,
            "sentence_translation": 0.35,
            "emotion_classification": 0.3
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        # 识别当前任务类型
        task_type = inputs.pop("task_type", None)

        # 计算基本损失
        outputs = model(**inputs)
        loss = outputs.loss

        # 应用任务权重
        if task_type in self.task_weights:
            loss = loss * self.task_weights[task_type]

        return (loss, outputs) if return_outputs else loss

def load_instruction_data(data_paths):
    """加载指令微调数据"""
    datasets = {}

    for task_name, path in data_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 将数据转换为数据集格式
        datasets[task_name] = Dataset.from_dict({
            "instruction": [item["instruction"] for item in data],
            "input": [item["input"] for item in data],
            "output": [item["output"] for item in data],
            "task_type": [item["task_type"] for item in data]
        })

    return datasets

def format_instruction(instruction, input_text, output_text=None):
    """格式化指令数据为模型输入格式"""
    # 构建提示模板
    if input_text:
        prompt = f"<|im_start|>system\n你是一个古诗词理解与推理助手。<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>system\n你是一个古诗词理解与推理助手。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    # 如果有输出，添加到提示中
    if output_text:
        prompt += f"{output_text}<|im_end|>"

    return prompt

def train_instruction_tuning(config):
    """进行多任务指令微调"""
    # 加载数据
    train_data_paths = {
        "word_task": config["train_data_paths"]["word_task"],
        "sent_task": config["train_data_paths"]["sent_task"],
        "emotion_task": config["train_data_paths"]["emotion_task"]
    }

    val_data_paths = {
        "word_task": config["val_data_paths"]["word_task"],
        "sent_task": config["val_data_paths"]["sent_task"],
        "emotion_task": config["val_data_paths"]["emotion_task"]
    }

    train_datasets = load_instruction_data(train_data_paths)
    val_datasets = load_instruction_data(val_data_paths)

    # 合并所有训练数据集
    train_dataset = concatenate_datasets(list(train_datasets.values()))
    val_dataset = concatenate_datasets(list(val_datasets.values()))

    logger.info(f"加载了 {len(train_dataset)} 条训练数据和 {len(val_dataset)} 条验证数据")

    # 加载领域预训练模型
    logger.info(f"加载领域预训练模型: {config['base_model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model_path"],
        trust_remote_code=True
    )

    # 确保分词器有正确的padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备模型进行量化训练
    model = prepare_model_for_kbit_training(model)

    # 配置LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 定义数据处理函数
    def preprocess_function(examples):
        # 格式化指令数据
        inputs = [format_instruction(instr, inp, out) for instr, inp, out in
                 zip(examples["instruction"], examples["input"], examples["output"])]

        # 分词
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=config["max_length"])

        # 创建标签
        labels = tokenizer(inputs, padding="max_length", truncation=True, max_length=config["max_length"])["input_ids"].copy()

        # 将非助手部分的标签设置为-100（忽略）
        for i, input_text in enumerate(inputs):
            assistant_start = input_text.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            if assistant_start >= len("<|im_start|>assistant\n"):
                # 找到助手部分在分词后的位置
                assistant_start_token = tokenizer(input_text[:assistant_start], return_tensors="pt")["input_ids"].shape[1] - 1
                # 将非助手部分的标签设置为-100
                labels[i][:assistant_start_token] = -100

        # 保留任务类型
        model_inputs["task_type"] = examples["task_type"]
        model_inputs["labels"] = labels

        return model_inputs

    # 处理数据集
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "input", "output"])
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["instruction", "input", "output"])

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=10,
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard"
    )

    # 创建多任务训练器
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        task_weights=config["task_weights"]
    )

    # 开始训练
    logger.info("开始多任务指令微调...")
    trainer.train()

    # 保存模型
    logger.info(f"保存模型到 {config['output_dir']}")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    return model, tokenizer

def main():
    # 训练配置
    config = {
        "base_model_path": "models/domain_pretrain",
        "train_data_paths": {
            "word_task": "data/final/instruction_tune/word_task_train.json",
            "sent_task": "data/final/instruction_tune/sent_task_train.json",
            "emotion_task": "data/final/instruction_tune/emotion_task_train.json"
        },
        "val_data_paths": {
            "word_task": "data/final/instruction_tune/word_task_val.json",
            "sent_task": "data/final/instruction_tune/sent_task_val.json",
            "emotion_task": "data/final/instruction_tune/emotion_task_val.json"
        },
        "output_dir": "models/instruction_tune",
        "num_epochs": 5,
        "batch_size": 4,
        "max_length": 1024,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gradient_accumulation_steps": 4,
        "save_steps": 500,
        "eval_steps": 500,
        "task_weights": {
            "word_explanation": 0.35,
            "sentence_translation": 0.35,
            "emotion_classification": 0.3
        }
    }

    # 进行多任务指令微调
    train_instruction_tuning(config)

if __name__ == "__main__":
    main()
```

运行多任务指令微调脚本：

```bash
python -m src.training.instruction_tune
```

### 3.3 人类偏好对齐

第三阶段，我们将使用DPO（Direct Preference Optimization）技术进行人类偏好对齐，使模型输出与人类专家评判保持一致。

```python
# src/training/preference_align.py
import os
import json
import torch
import logging
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_preference_data(data_path):
    """加载偏好对齐数据"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 将数据转换为数据集格式
    return Dataset.from_dict({
        "prompt": [item["prompt"] for item in data],
        "chosen": [item["chosen"] for item in data],
        "rejected": [item["rejected"] for item in data]
    })

def train_preference_alignment(config):
    """进行人类偏好对齐训练"""
    # 加载数据
    train_dataset = load_preference_data(config["train_data_path"])
    val_dataset = load_preference_data(config["val_data_path"])

    logger.info(f"加载了 {len(train_dataset)} 条训练数据和 {len(val_dataset)} 条验证数据")

    # 加载指令微调模型
    logger.info(f"加载指令微调模型: {config['base_model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model_path"],
        trust_remote_code=True
    )

    # 确保分词器有正确的padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 准备模型进行量化训练
    model = prepare_model_for_kbit_training(model)

    # 配置LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        logging_dir=f"{config['output_dir']}/logs",
        logging_steps=10,
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard"
    )

    # 创建DPO训练器
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        beta=config["beta"],
        max_length=config["max_length"],
        max_prompt_length=config["max_prompt_length"]
    )

    # 开始训练
    logger.info("开始人类偏好对齐训练...")
    dpo_trainer.train()

    # 保存模型
    logger.info(f"保存模型到 {config['output_dir']}")
    dpo_trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    return model, tokenizer

def main():
    # 训练配置
    config = {
        "base_model_path": "models/instruction_tune",
        "train_data_path": "data/final/preference_align/train.json",
        "val_data_path": "data/final/preference_align/val.json",
        "output_dir": "models/preference_align",
        "num_epochs": 3,
        "batch_size": 2,
        "max_length": 1024,
        "max_prompt_length": 512,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gradient_accumulation_steps": 4,
        "save_steps": 200,
        "eval_steps": 200,
        "beta": 0.1  # DPO特有参数
    }

    # 进行人类偏好对齐训练
    train_preference_alignment(config)

if __name__ == "__main__":
    main()
```

运行人类偏好对齐脚本：

```bash
python -m src.training.preference_align
```

完成所有训练后，我们可以创建一个一键训练脚本：

```bash
# scripts/train_all.sh
#!/bin/bash

echo "开始多阶段训练流程..."

# 1. 领域预训练
echo "步骤1: 领域预训练"
python -m src.training.domain_pretrain

# 2. 多任务指令微调
echo "步骤2: 多任务指令微调"
python -m src.training.instruction_tune

# 3. 人类偏好对齐
echo "步骤3: 人类偏好对齐"
python -m src.training.preference_align

echo "训练完成!"
```

运行一键训练脚本：

```bash
chmod +x scripts/train_all.sh
./scripts/train_all.sh
```

## 4. 模型集成与推理

在这一部分，我们将实现模型集成和推理流程，包括集成多个模型的输出和高级后处理。

### 4.1 模型集成

首先，我们实现模型集成类，用于组合多个模型的输出：

```python
# src/models/ensemble.py
import os
import json
import torch
import logging
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ModelEnsemble:
    """模型集成类，用于组合多个模型的输出"""

    def __init__(self, model_paths, weights=None, device="cuda"):
        """
        初始化模型集成

        Args:
            model_paths: 模型路径列表
            weights: 模型权重列表，如果为None，则所有模型权重相同
            device: 运行设备
        """
        self.models = []
        self.tokenizers = []
        self.weights = weights or [1.0] * len(model_paths)

        # 归一化权重
        self.weights = [w / sum(self.weights) for w in self.weights]

        # 加载所有模型
        for i, path in enumerate(model_paths):
            logger.info(f"加载模型 {i+1}/{len(model_paths)}: {path}")
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=True
            )

            self.models.append(model)
            self.tokenizers.append(tokenizer)

    def generate(self, prompt, max_length=1024, temperature=0.7, top_p=0.9):
        """
        生成文本并集成结果

        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: 核采样参数

        Returns:
            集成后的结果
        """
        all_outputs = []

        # 从每个模型获取输出
        for i, (model, tokenizer, weight) in enumerate(zip(self.models, self.tokenizers, self.weights)):
            logger.info(f"使用模型 {i+1}/{len(self.models)} 生成输出...")

            # 格式化提示
            formatted_prompt = f"<|im_start|>system\n你是一个古诗词理解与推理助手。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            # 分词
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            # 生成输出
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )

            # 解码输出
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

            # 提取助手部分
            assistant_start = output_text.rfind("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            assistant_end = output_text.rfind("<|im_end|>") if "<|im_end|>" in output_text else len(output_text)
            assistant_response = output_text[assistant_start:assistant_end].strip()

            # 尝试解析JSON
            try:
                parsed_output = json.loads(assistant_response)
                all_outputs.append((parsed_output, weight))
            except json.JSONDecodeError:
                logger.warning(f"模型 {i+1} 输出不是有效的JSON，尝试提取...")
                # 尝试提取JSON部分
                try:
                    json_start = assistant_response.find("{")
                    json_end = assistant_response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = assistant_response[json_start:json_end]
                        parsed_output = json.loads(json_str)
                        all_outputs.append((parsed_output, weight))
                    else:
                        logger.warning(f"无法从模型 {i+1} 输出中提取JSON")
                except Exception as e:
                    logger.error(f"处理模型 {i+1} 输出时出错: {e}")

        # 集成结果
        if not all_outputs:
            logger.error("所有模型都未能生成有效输出")
            return {}

        # 初始化集成结果
        ensemble_result = {
            "ans_qa_words": {},
            "ans_qa_sents": {},
            "choose_id": ""
        }

        # 集成词语解释
        word_explanations = {}
        for output, weight in all_outputs:
            if "ans_qa_words" in output:
                for word, explanation in output["ans_qa_words"].items():
                    if word not in word_explanations:
                        word_explanations[word] = {}

                    if explanation not in word_explanations[word]:
                        word_explanations[word][explanation] = 0

                    word_explanations[word][explanation] += weight

        # 选择最终词语解释
        for word, explanations in word_explanations.items():
            ensemble_result["ans_qa_words"][word] = max(explanations.items(), key=lambda x: x[1])[0]

        # 集成句子翻译
        sent_translations = {}
        for output, weight in all_outputs:
            if "ans_qa_sents" in output:
                for sent, translation in output["ans_qa_sents"].items():
                    if sent not in sent_translations:
                        sent_translations[sent] = {}

                    if translation not in sent_translations[sent]:
                        sent_translations[sent][translation] = 0

                    sent_translations[sent][translation] += weight

        # 选择最终句子翻译
        for sent, translations in sent_translations.items():
            ensemble_result["ans_qa_sents"][sent] = max(translations.items(), key=lambda x: x[1])[0]

        # 集成情感分类
        emotion_votes = {}
        for output, weight in all_outputs:
            if "choose_id" in output and output["choose_id"]:
                emotion = output["choose_id"]
                if emotion not in emotion_votes:
                    emotion_votes[emotion] = 0

                emotion_votes[emotion] += weight

        # 选择最终情感分类
        if emotion_votes:
            ensemble_result["choose_id"] = max(emotion_votes.items(), key=lambda x: x[1])[0]

        return ensemble_result

def load_ensemble_model(config):
    """加载集成模型"""
    model_paths = config["model_paths"]
    weights = config.get("weights", None)

    return ModelEnsemble(model_paths, weights)

def main():
    # 测试集成模型
    config = {
        "model_paths": [
            "models/preference_align",  # 主模型（Qwen2.5-7B）
            "models/chatglm_preference_align"  # 辅助模型（ChatGLM4-9B）
        ],
        "weights": [0.6, 0.4]  # 模型权重
    }

    ensemble = load_ensemble_model(config)

    # 测试提示
    test_prompt = """请解释以下古诗词中的词语：春风、江南
请翻译以下句子：春风又绿江南岸
请选择这首诗表达的主要情感：{"A": "思乡", "B": "写景"}"""

    # 生成结果
    result = ensemble.generate(test_prompt)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

运行模型集成测试：

```bash
python -m src.models.ensemble
```

### 4.2 推理流程

接下来，我们实现完整的推理流程，包括高级后处理：

```python
# src/inference/pipeline.py
import os
import json
import logging
import argparse
from tqdm import tqdm
from src.models.ensemble import load_ensemble_model

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_test_data(file_path):
    """加载测试数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_prompt(item):
    """格式化提示"""
    prompt = f"标题：{item.get('title', '')}\n"
    prompt += f"作者：{item.get('author', '')}\n"
    prompt += f"内容：{item.get('content', '')}\n\n"

    # 添加词语解释任务
    if 'qa_words' in item and item['qa_words']:
        prompt += f"请解释以下词语：{', '.join(item['qa_words'])}\n\n"

    # 添加句子翻译任务
    if 'qa_sents' in item and item['qa_sents']:
        prompt += f"请翻译以下句子：{', '.join(item['qa_sents'])}\n\n"

    # 添加情感分类任务
    if 'choose' in item and item['choose']:
        prompt += f"请选择这首诗表达的主要情感：{json.dumps(item['choose'], ensure_ascii=False)}"

    return prompt

def advanced_postprocessing(raw_result, item):
    """高级后处理，确保输出格式正确并优化质量"""
    try:
        # 初始化结果
        processed_result = {
            "ans_qa_words": {},
            "ans_qa_sents": {},
            "choose_id": ""
        }

        # 处理词语解释
        if 'qa_words' in item and item['qa_words']:
            for word in item['qa_words']:
                if word in raw_result.get("ans_qa_words", {}):
                    explanation = raw_result["ans_qa_words"][word]
                    # 质量检查：确保解释不是空字符串且长度合理
                    if explanation and len(explanation) >= 5:
                        processed_result["ans_qa_words"][word] = explanation
                    else:
                        processed_result["ans_qa_words"][word] = generate_fallback_explanation(word)
                else:
                    processed_result["ans_qa_words"][word] = generate_fallback_explanation(word)

        # 处理句子翻译
        if 'qa_sents' in item and item['qa_sents']:
            for sent in item['qa_sents']:
                if sent in raw_result.get("ans_qa_sents", {}):
                    translation = raw_result["ans_qa_sents"][sent]
                    # 质量检查：确保翻译不是空字符串且长度合理
                    if translation and len(translation) >= len(sent) * 0.5:
                        processed_result["ans_qa_sents"][sent] = translation
                    else:
                        processed_result["ans_qa_sents"][sent] = generate_fallback_translation(sent)
                else:
                    processed_result["ans_qa_sents"][sent] = generate_fallback_translation(sent)

        # 处理情感分类
        if 'choose' in item and item['choose']:
            choose_options = list(item['choose'].keys())
            if "choose_id" in raw_result and raw_result["choose_id"] in choose_options:
                processed_result["choose_id"] = raw_result["choose_id"]
            else:
                # 如果没有有效的情感分类，尝试从解释和翻译中推断
                processed_result["choose_id"] = infer_emotion_from_content(
                    processed_result["ans_qa_words"],
                    processed_result["ans_qa_sents"],
                    item['choose'],
                    choose_options
                )

        return processed_result

    except Exception as e:
        logger.error(f"后处理时出错: {e}")
        # 出现异常时返回备用结果
        return generate_complete_fallback(item)

def generate_fallback_explanation(word):
    """生成备用词语解释"""
    # 简单的备用解释字典
    fallbacks = {
        "春风": "指春天的风，常用来表示暖意和生机",
        "秋月": "指秋天的月亮，常用来表示清凉、明朗或思乡",
        "江南": "指长江以南的地区，常指现在的江苏、浙江一带",
        "山水": "指山和水，常用来指代自然风光或国家江山",
    }

    # 如果有备用解释，返回它
    if word in fallbacks:
        return fallbacks[word]

    # 否则返回通用解释
    return f"{word}是古诗词中常见的词语，具体含义与诗词上下文相关"

def generate_fallback_translation(sent):
    """生成备用句子翻译"""
    # 简单的备用翻译字典
    fallbacks = {
        "春风又绿江南岸": "春天的风再次使江南的河岸变绿了",
        "白日依山尽": "太阳落山，天色将暗",
        "黄河入海流": "黄河水流入海，永不停息",
    }

    # 如果有备用翻译，返回它
    if sent in fallbacks:
        return fallbacks[sent]

    # 否则返回简单的直译
    return f"这句诗的字面意思是：{sent}"

def infer_emotion_from_content(word_explanations, sent_translations, choose_options, choose_ids):
    """从内容推断情感类别"""
    # 情感关键词字典
    emotion_keywords = {
        "思乡": ["乡", "思", "家", "远", "归", "游子", "故园", "归期"],
        "送别": ["别", "送", "分别", "离别", "一壶", "酒", "道别"],
        "爱国": ["国", "山河", "江山", "忠", "报国", "山河", "江山"],
        "写景": ["景", "山", "水", "春", "秋", "风", "月", "花", "雪", "绿"],
        "哀思": ["哀", "思", "怨", "怨天", "怨命", "悲", "悲伤", "悲秋"]
    }

    # 计算每个情感类别的匹配分数
    scores = {emotion: 0 for emotion in choose_options.values()}

    # 从词语解释中推断
    for word, explanation in word_explanations.items():
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in explanation:
                    scores[emotion] = scores.get(emotion, 0) + 1

    # 从句子翻译中推断
    for sent, translation in sent_translations.items():
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in translation:
                    scores[emotion] = scores.get(emotion, 0) + 1

    # 选择得分最高的情感
    if scores:
        best_emotion = max(scores.items(), key=lambda x: x[1])[0]
        # 找到对应的ID
        for id, emotion in choose_options.items():
            if emotion == best_emotion:
                return id

    # 如果无法推断，随机选择一个
    return choose_ids[0]

def generate_complete_fallback(item):
    """生成完整的备用结果"""
    result = {
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": ""
    }

    # 生成备用词语解释
    if 'qa_words' in item and item['qa_words']:
        for word in item['qa_words']:
            result["ans_qa_words"][word] = generate_fallback_explanation(word)

    # 生成备用句子翻译
    if 'qa_sents' in item and item['qa_sents']:
        for sent in item['qa_sents']:
            result["ans_qa_sents"][sent] = generate_fallback_translation(sent)

    # 生成备用情感分类
    if 'choose' in item and item['choose']:
        result["choose_id"] = list(item['choose'].keys())[0]  # 选择第一个选项

    return result

def run_inference(model, test_data, output_path):
    """运行推理并保存结果"""
    results = []

    for i, item in enumerate(tqdm(test_data, desc="运行推理")):
        logger.info(f"处理样本 {i+1}/{len(test_data)}")

        # 格式化提示
        prompt = format_prompt(item)

        # 生成结果
        raw_result = model.generate(prompt)

        # 后处理
        processed_result = advanced_postprocessing(raw_result, item)

        # 添加原始数据的ID
        if 'id' in item:
            processed_result['id'] = item['id']

        results.append(processed_result)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"推理完成，结果已保存到 {output_path}")

    return results

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行古诗词理解与推理模型")
    parser.add_argument("--test_file", type=str, default="data/raw/ccl_poetry_test.json", help="测试数据文件路径")
    parser.add_argument("--output_file", type=str, default="results/predictions.json", help="输出文件路径")
    parser.add_argument("--model_config", type=str, default="configs/ensemble_config.json", help="模型配置文件路径")
    args = parser.parse_args()

    # 加载模型配置
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    # 加载集成模型
    model = load_ensemble_model(model_config)

    # 加载测试数据
    test_data = load_test_data(args.test_file)

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 运行推理
    run_inference(model, test_data, args.output_file)

if __name__ == "__main__":
    main()
```

创建模型配置文件：

```json
# configs/ensemble_config.json
{
  "model_paths": [
    "models/preference_align",
    "models/chatglm_preference_align"
  ],
  "weights": [0.6, 0.4]
}
```

运行推理脚本：

```bash
python -m src.inference.pipeline --test_file data/raw/ccl_poetry_test.json --output_file results/predictions.json
```

## 5. 结果优化与后处理

在这一部分，我们将实现结果优化和评估工具，以确保我们的模型输出质量。

### 5.1 结果评估

首先，我们实现一个评估脚本，用于评估模型输出的质量：

```python
# src/utils/metrics.py
import json
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

def load_predictions(pred_file):
    """加载预测结果"""
    with open(pred_file, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ground_truth(gt_file):
    """加载真实标签"""
    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_word_explanation_accuracy(predictions, ground_truth):
    """计算词语解释准确率"""
    total_words = 0
    correct_words = 0

    for pred, gt in zip(predictions, ground_truth):
        if 'ans_qa_words' in pred and 'ans_qa_words' in gt:
            for word in gt['ans_qa_words']:
                if word in pred['ans_qa_words']:
                    total_words += 1
                    # 计算BLEU分数作为相似度度量
                    reference = word_tokenize(gt['ans_qa_words'][word])
                    candidate = word_tokenize(pred['ans_qa_words'][word])

                    try:
                        bleu_score = sentence_bleu([reference], candidate)
                        if bleu_score > 0.5:  # 如果BLEU分数超过阈值，认为是正确的
                            correct_words += 1
                    except:
                        pass  # 如果计算BLEU失败，忽略该词

    return correct_words / total_words if total_words > 0 else 0

def calculate_sentence_translation_accuracy(predictions, ground_truth):
    """计算句子翻译准确率"""
    total_sents = 0
    correct_sents = 0
    bleu_scores = []

    for pred, gt in zip(predictions, ground_truth):
        if 'ans_qa_sents' in pred and 'ans_qa_sents' in gt:
            for sent in gt['ans_qa_sents']:
                if sent in pred['ans_qa_sents']:
                    total_sents += 1
                    # 计算BLEU分数
                    reference = word_tokenize(gt['ans_qa_sents'][sent])
                    candidate = word_tokenize(pred['ans_qa_sents'][sent])

                    try:
                        bleu_score = sentence_bleu([reference], candidate)
                        bleu_scores.append(bleu_score)
                        if bleu_score > 0.5:  # 如果BLEU分数超过阈值，认为是正确的
                            correct_sents += 1
                    except:
                        pass  # 如果计算BLEU失败，忽略该句

    accuracy = correct_sents / total_sents if total_sents > 0 else 0
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0

    return accuracy, avg_bleu

def calculate_emotion_classification_accuracy(predictions, ground_truth):
    """计算情感分类准确率"""
    total = 0
    correct = 0

    for pred, gt in zip(predictions, ground_truth):
        if 'choose_id' in pred and 'choose_id' in gt:
            total += 1
            if pred['choose_id'] == gt['choose_id']:
                correct += 1

    return correct / total if total > 0 else 0

def evaluate_predictions(predictions, ground_truth):
    """评估预测结果"""
    # 词语解释准确率
    word_accuracy = calculate_word_explanation_accuracy(predictions, ground_truth)

    # 句子翻译准确率和BLEU分数
    sent_accuracy, avg_bleu = calculate_sentence_translation_accuracy(predictions, ground_truth)

    # 情感分类准确率
    emotion_accuracy = calculate_emotion_classification_accuracy(predictions, ground_truth)

    # 计算加权平均分数
    weighted_score = 0.35 * word_accuracy + 0.35 * sent_accuracy + 0.3 * emotion_accuracy

    return {
        "word_explanation_accuracy": word_accuracy,
        "sentence_translation_accuracy": sent_accuracy,
        "sentence_translation_bleu": avg_bleu,
        "emotion_classification_accuracy": emotion_accuracy,
        "weighted_score": weighted_score
    }

def main():
    # 加载预测结果和真实标签
    predictions = load_predictions("results/predictions.json")
    ground_truth = load_ground_truth("data/raw/ccl_poetry_val.json")

    # 评估预测结果
    metrics = evaluate_predictions(predictions, ground_truth)

    # 打印评估结果
    print("\n评估结果:")
    print(f"\u8bcd语解释准确率: {metrics['word_explanation_accuracy']:.4f}")
    print(f"句子翻译准确率: {metrics['sentence_translation_accuracy']:.4f}")
    print(f"句子翻译BLEU分数: {metrics['sentence_translation_bleu']:.4f}")
    print(f"情感分类准确率: {metrics['emotion_classification_accuracy']:.4f}")
    print(f"加权平均分数: {metrics['weighted_score']:.4f}")

    # 保存评估结果
    with open("results/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
```

运行评估脚本：

```bash
python -m src.utils.metrics
```

### 5.2 结果可视化

我们还可以实现一个结果可视化脚本，用于分析模型输出：

```python
# src/utils/visualize.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import jieba

def load_predictions(pred_file):
    """加载预测结果"""
    with open(pred_file, "r", encoding="utf-8") as f:
        return json.load(f)

def visualize_word_explanations(predictions, output_dir):
    """可视化词语解释结果"""
    # 收集所有词语
    all_words = []
    for pred in predictions:
        if 'ans_qa_words' in pred:
            all_words.extend(list(pred['ans_qa_words'].keys()))

    # 统计词语频率
    word_counter = Counter(all_words)

    # 绘制前20个最常见的词语
    top_words = word_counter.most_common(20)
    words = [word for word, _ in top_words]
    counts = [count for _, count in top_words]

    plt.figure(figsize=(12, 8))
    plt.bar(words, counts)
    plt.title('最常见的词语')
    plt.xlabel('词语')
    plt.ylabel('频率')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_words.png'))
    plt.close()

    # 创建词云
    all_explanations = []
    for pred in predictions:
        if 'ans_qa_words' in pred:
            all_explanations.extend(list(pred['ans_qa_words'].values()))

    # 将所有解释连接成一个大文本
    text = ' '.join(all_explanations)

    # 分词
    words = ' '.join(jieba.cut(text))

    # 创建词云
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='SimHei.ttf').generate(words)

    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('词语解释词云')
    plt.savefig(os.path.join(output_dir, 'word_explanations_wordcloud.png'))
    plt.close()

def visualize_sentence_translations(predictions, output_dir):
    """可视化句子翻译结果"""
    # 统计翻译长度
    translation_lengths = []
    for pred in predictions:
        if 'ans_qa_sents' in pred:
            for translation in pred['ans_qa_sents'].values():
                translation_lengths.append(len(translation))

    # 绘制翻译长度分布
    plt.figure(figsize=(12, 8))
    plt.hist(translation_lengths, bins=20)
    plt.title('翻译长度分布')
    plt.xlabel('长度')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'translation_length_distribution.png'))
    plt.close()

    # 创建翻译词云
    all_translations = []
    for pred in predictions:
        if 'ans_qa_sents' in pred:
            all_translations.extend(list(pred['ans_qa_sents'].values()))

    # 将所有翻译连接成一个大文本
    text = ' '.join(all_translations)

    # 分词
    words = ' '.join(jieba.cut(text))

    # 创建词云
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='SimHei.ttf').generate(words)

    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('句子翻译词云')
    plt.savefig(os.path.join(output_dir, 'sentence_translations_wordcloud.png'))
    plt.close()

def visualize_emotion_classifications(predictions, output_dir):
    """可视化情感分类结果"""
    # 统计情感分类
    emotion_counter = Counter([pred.get('choose_id', '') for pred in predictions if 'choose_id' in pred])

    # 绘制情感分类分布
    emotions = list(emotion_counter.keys())
    counts = list(emotion_counter.values())

    plt.figure(figsize=(12, 8))
    plt.bar(emotions, counts)
    plt.title('情感分类分布')
    plt.xlabel('情感类别')
    plt.ylabel('频率')
    plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
    plt.close()

    # 绘制饼图
    plt.figure(figsize=(10, 10))
    plt.pie(counts, labels=emotions, autopct='%1.1f%%')
    plt.title('情感分类占比')
    plt.savefig(os.path.join(output_dir, 'emotion_pie_chart.png'))
    plt.close()

def main():
    # 加载预测结果
    predictions = load_predictions("results/predictions.json")

    # 创建输出目录
    output_dir = "results/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # 可视化词语解释结果
    visualize_word_explanations(predictions, output_dir)

    # 可视化句子翻译结果
    visualize_sentence_translations(predictions, output_dir)

    # 可视化情感分类结果
    visualize_emotion_classifications(predictions, output_dir)

    print(f"可视化结果已保存到 {output_dir}")

if __name__ == "__main__":
    main()
```

运行可视化脚本：

```bash
python -m src.utils.visualize
```

## 6. 提交与验证
