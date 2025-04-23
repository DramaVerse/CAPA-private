#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
古诗词理解与推理评测任务数据集特征分析脚本

本脚本对CCL 2025古诗词理解与推理评测任务的数据集进行详细分析，
提取有助于训练和模型选择的关键特征。

作者: Augment Agent
日期: 2024
"""

import json
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from tqdm import tqdm
import jieba
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 获取数据目录下的所有JSON文件
def get_all_json_files(data_dir="@data"):
    json_files = []
    # 如果路径以@开头，则将其解析为相对于项目根目录的路径
    if data_dir.startswith('@'):
        data_dir = data_dir[1:]
        # 假设当前脚本在项目根目录的try2子目录下
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_dir)

    # 递归遍历目录下的所有JSON文件
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    logger.info("找到%d个JSON文件" % len(json_files))
    return json_files

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='古诗词数据集特征分析工具')
    parser.add_argument('--input', type=str, help='输入数据集JSON文件路径，如果不指定则处理@data目录下所有JSON文件')
    parser.add_argument('--output_dir', type=str, default='.', help='分析结果输出目录')
    args = parser.parse_args()

    # 设置默认值
    args.save_plots = True
    args.detailed = True

    return args

# 创建输出目录
def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("创建输出目录: %s" % output_dir)

    # 创建子目录
    plots_dir = os.path.join(output_dir, 'plots')
    stats_dir = os.path.join(output_dir, 'stats')

    for dir_path in [plots_dir, stats_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    return plots_dir, stats_dir

# 加载数据集
def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("成功加载数据集: %s, 样本数量: %d" % (file_path, len(data)))
        return data
    except Exception as e:
        logger.error("加载数据集失败: %s" % e)
        sys.exit(1)

# 基本统计分析
def basic_statistics(data):
    """计算数据集的基本统计信息"""
    stats = {}

    # 样本数量
    stats['sample_count'] = len(data)

    # 诗词长度统计
    content_lengths = [len(item['content']) for item in data]
    stats['content_length'] = {
        'min': min(content_lengths),
        'max': max(content_lengths),
        'mean': np.mean(content_lengths),
        'median': np.median(content_lengths),
        'std': np.std(content_lengths)
    }

    # 词语解释任务统计
    word_counts = [len(item.get('keywords', {})) for item in data]
    stats['word_explanation'] = {
        'min': min(word_counts) if word_counts else 0,
        'max': max(word_counts) if word_counts else 0,
        'mean': np.mean(word_counts) if word_counts else 0,
        'median': np.median(word_counts) if word_counts else 0,
        'total_words': sum(word_counts),
        'unique_words': len(set([word for item in data for word in item.get('keywords', {})]))
    }

    # 句子翻译任务统计
    # 对于'trans'，它可能是字符串而不是列表，所以我们将其计为1个句子
    sent_counts = [1 if isinstance(item.get('trans', ''), str) else 0 for item in data]
    stats['sentence_translation'] = {
        'min': min(sent_counts) if sent_counts else 0,
        'max': max(sent_counts) if sent_counts else 0,
        'mean': np.mean(sent_counts) if sent_counts else 0,
        'median': np.median(sent_counts) if sent_counts else 0,
        'total_sentences': sum(sent_counts)
    }

    # 情感分类任务统计
    emotion_options = []
    for item in data:
        if isinstance(item.get('emotion', ''), str):
            # 如果是字符串，可能是用逗号分隔的多个情感
            emotions = [e.strip() for e in item.get('emotion', '').split('\u3001') if e.strip()]
            if not emotions:  # 如果没有用逗号分隔，尝试用逗号分隔
                emotions = [e.strip() for e in item.get('emotion', '').split(',') if e.strip()]
            emotion_options.extend(emotions)

    emotion_counter = Counter(emotion_options)
    stats['emotion_classification'] = {
        'unique_emotions': len(emotion_counter),
        'emotion_distribution': dict(emotion_counter)
    }

    # 作者统计
    author_counter = Counter([item.get('author', '佚名') for item in data])
    stats['authors'] = {
        'unique_authors': len(author_counter),
        'top_authors': dict(author_counter.most_common(10))
    }

    return stats

# 诗词类型分析
def analyze_poetry_types(data):
    """分析诗词的类型（律诗、绝句等）"""
    poetry_types = []

    for item in data:
        content = item['content']
        # 按行分割，去除空行
        lines = [line for line in content.split('\n') if line.strip()]

        # 简单判断诗歌类型（基于行数和字数）
        if len(lines) == 4:
            if all(len(line) == 5 or len(line) == 6 for line in lines):
                poetry_types.append("五言绝句")
            elif all(len(line) == 7 or len(line) == 8 for line in lines):
                poetry_types.append("七言绝句")
            else:
                poetry_types.append("其他绝句")
        elif len(lines) == 8:
            if all(len(line) == 5 or len(line) == 6 for line in lines):
                poetry_types.append("五言律诗")
            elif all(len(line) == 7 or len(line) == 8 for line in lines):
                poetry_types.append("七言律诗")
            else:
                poetry_types.append("其他律诗")
        elif len(lines) > 8 and len(lines) % 2 == 0:
            poetry_types.append("长篇")
        else:
            poetry_types.append("其他")

    type_counter = Counter(poetry_types)
    return dict(type_counter)

# 词语分析
def analyze_words(data):
    """分析需要解释的词语的特点"""
    # 收集所有需要解释的词语
    all_words = [word for item in data for word in item.get('keywords', {}).keys()]
    word_counter = Counter(all_words)

    # 词语长度分布
    word_lengths = [len(word) for word in all_words]
    word_length_counter = Counter(word_lengths)

    # 分析词语的字符组成
    char_counter = Counter([char for word in all_words for char in word])

    # 分析词语在诗中的位置
    word_positions = defaultdict(list)
    for item in data:
        content = item.get('content', '')
        for word in item.get('keywords', {}).keys():
            positions = [m.start() for m in re.finditer(re.escape(word), content)]
            if positions:  # 可能有些词语在内容中找不到精确位置
                word_positions[word].extend(positions)

    # 计算每个词在诗中的平均位置百分比
    avg_positions = {}
    for word, positions in word_positions.items():
        if positions:  # 确保有位置信息
            samples = []
            for item in data:
                if word in item.get('keywords', {}).keys():
                    content_len = len(item.get('content', ''))
                    word_pos = content_len if content_len > 0 else 1  # 防止除零
                    for pos in [m.start() for m in re.finditer(re.escape(word), item.get('content', ''))]:
                        samples.append(pos / word_pos)
            if samples:
                avg_positions[word] = np.mean(samples)

    return {
        'word_frequency': dict(word_counter.most_common(50)),
        'word_length_distribution': dict(sorted(word_length_counter.items())),
        'top_characters': dict(char_counter.most_common(30)),
        'avg_position_percentage': dict(sorted(avg_positions.items(), key=lambda x: x[1])[:20])
    }

# 句子分析
def analyze_sentences(data):
    """分析需要翻译的句子的特点"""
    # 收集所有需要翻译的句子
    all_sentences = []
    for item in data:
        if isinstance(item.get('trans', ''), str) and item.get('trans', '').strip():
            all_sentences.append(item.get('trans', ''))

    # 句子长度分布
    sent_lengths = [len(sent) for sent in all_sentences]
    sent_length_counter = Counter(sent_lengths)

    # 分析句子的字符组成
    char_counter = Counter([char for sent in all_sentences for char in sent])

    # 分析句子在诗中的位置
    sent_positions = defaultdict(list)
    for item in data:
        content = item.get('content', '')
        trans = item.get('trans', '')
        if isinstance(trans, str) and trans.strip():
            positions = [m.start() for m in re.finditer(re.escape(trans[:20]), content) if trans[:20].strip()]
            if positions:  # 可能有些句子在内容中找不到精确位置
                sent_positions[trans].extend(positions)

    # 分析句子是否为完整诗句
    complete_lines = 0
    for item in data:
        content = item.get('content', '')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        trans = item.get('trans', '')
        if isinstance(trans, str) and trans.strip():
            # 简单检查翻译是否包含完整诗句
            for line in lines:
                if line in trans:
                    complete_lines += 1
                    break

    # 分析句子的句式结构
    sentence_patterns = []
    for sent in all_sentences:
        # 简化的句式分析，分析句子是否包含常见的句式标记
        pattern = ''
        if '，' in sent:
            pattern += '逗号'
        if '。' in sent:
            pattern += '句号'
        if '！' in sent:
            pattern += '叹号'
        if '？' in sent:
            pattern += '问号'
        if '；' in sent:
            pattern += '分号'
        if not pattern:  # 如果没有标点
            pattern = '无标点'
        sentence_patterns.append(pattern)

    pattern_counter = Counter(sentence_patterns)

    return {
        'sentence_length_distribution': dict(sorted(sent_length_counter.items())),
        'top_characters': dict(char_counter.most_common(30)),
        'complete_line_percentage': complete_lines / len(all_sentences) if all_sentences else 0,
        'sentence_pattern_distribution': dict(pattern_counter)
    }

# 情感分析
def analyze_emotions(data):
    """分析情感分类任务的特点"""
    # 收集所有情感选项
    emotion_options = []
    for item in data:
        if isinstance(item.get('emotion', ''), str):
            # 如果是字符串，可能是用逗号分隔的多个情感
            emotions = [e.strip() for e in item.get('emotion', '').split('、') if e.strip()]
            if not emotions:  # 如果没有用逗号分隔，尝试用逗号分隔
                emotions = [e.strip() for e in item.get('emotion', '').split(',') if e.strip()]
            emotion_options.extend(emotions)

    # 情感选项统计
    emotion_counter = Counter(emotion_options)

    # 分析不同情感类别的诗词特点
    emotion_word_stats = {}
    emotion_sent_stats = {}

    for emotion in set(emotion_options):
        # 找出包含该情感的样本
        emotion_samples = []
        for item in data:
            if isinstance(item.get('emotion', ''), str) and emotion in item.get('emotion', ''):
                emotion_samples.append(item)

        if emotion_samples:
            # 计算该情感类别的诗词平均长度
            avg_length = np.mean([len(item.get('content', '')) for item in emotion_samples])

            # 计算该情感类别的词语和句子平均数量
            avg_words = np.mean([len(item.get('keywords', {})) for item in emotion_samples])
            avg_sents = np.mean([1 if isinstance(item.get('trans', ''), str) and item.get('trans', '').strip() else 0 for item in emotion_samples])

            # 统计该情感类别的常见词语
            emotion_words = [word for item in emotion_samples for word in item.get('keywords', {}).keys()]
            top_words = dict(Counter(emotion_words).most_common(10))

            emotion_word_stats[emotion] = {
                'sample_count': len(emotion_samples),
                'avg_content_length': avg_length,
                'avg_word_count': avg_words,
                'avg_sent_count': avg_sents,
                'top_words': top_words
            }

            # 统计该情感类别的常见句子特点
            emotion_sents = [item.get('trans', '') for item in emotion_samples if isinstance(item.get('trans', ''), str) and item.get('trans', '').strip()]
            avg_sent_length = np.mean([len(sent) for sent in emotion_sents]) if emotion_sents else 0

            emotion_sent_stats[emotion] = {
                'avg_sent_length': avg_sent_length,
                'sent_count': len(emotion_sents)
            }

    return {
        'emotion_distribution': dict(emotion_counter),
        'emotion_id_distribution': {},  # 空字典，因为我们没有ID
        'emotion_word_stats': emotion_word_stats,
        'emotion_sent_stats': emotion_sent_stats
    }

# 可视化函数
def visualize_basic_stats(stats, plots_dir):
    """可视化基本统计信息"""
    # 诗词长度分布直方图
    plt.figure(figsize=(10, 6))
    content_lengths = [stats['content_length']['min'], stats['content_length']['max'],
                      stats['content_length']['mean'], stats['content_length']['median']]
    plt.bar(['Min', 'Max', 'Mean', 'Median'], content_lengths, color='skyblue')
    plt.title('诗词长度统计')
    plt.ylabel('字符数')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'content_length_stats.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 词语和句子数量统计
    plt.figure(figsize=(10, 6))
    task_counts = [
        stats['word_explanation']['total_words'],
        stats['word_explanation']['unique_words'],
        stats['sentence_translation']['total_sentences']
    ]
    plt.bar(['词语总数', '唯一词语数', '句子总数'], task_counts, color=['blue', 'green', 'orange'])
    plt.title('词语和句子统计')
    plt.ylabel('数量')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'word_sentence_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 情感分布饼图
    plt.figure(figsize=(12, 8))
    emotions = stats['emotion_classification']['emotion_distribution']
    plt.pie(emotions.values(), labels=emotions.keys(), autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')  # 确保饼图是圆的
    plt.title('情感分布')
    plt.savefig(os.path.join(plots_dir, 'emotion_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 作者分布条形图
    plt.figure(figsize=(12, 8))
    authors = stats['authors']['top_authors']
    plt.barh(list(authors.keys()), list(authors.values()), color='purple')
    plt.title('前10位作者分布')
    plt.xlabel('诗词数量')
    plt.gca().invert_yaxis()  # 使最高频的在顶部
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'top_authors.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_poetry_types(poetry_types, plots_dir):
    """可视化诗词类型分布"""
    plt.figure(figsize=(12, 8))
    types = dict(sorted(poetry_types.items(), key=lambda x: x[1], reverse=True))
    plt.bar(types.keys(), types.values(), color='teal')
    plt.title('诗词类型分布')
    plt.ylabel('数量')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'poetry_types.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_word_analysis(word_analysis, plots_dir):
    """可视化词语分析结果"""
    # 词语长度分布
    plt.figure(figsize=(10, 6))
    word_lengths = word_analysis['word_length_distribution']
    plt.bar(word_lengths.keys(), word_lengths.values(), color='coral')
    plt.title('词语长度分布')
    plt.xlabel('词语长度')
    plt.ylabel('数量')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'word_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 高频词语条形图
    plt.figure(figsize=(12, 10))
    top_words = dict(list(word_analysis['word_frequency'].items())[:20])  # 取前20个
    plt.barh(list(top_words.keys()), list(top_words.values()), color='darkblue')
    plt.title('高频词语前20位')
    plt.xlabel('出现频率')
    plt.gca().invert_yaxis()  # 使最高频的在顶部
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'top_words.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 尝试创建词云
    try:
        # 创建词云
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              font_path=fm.findfont(fm.FontProperties(family='SimHei')),
                              max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate_from_frequencies(word_analysis['word_frequency'])

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('需要解释的词语词云')
        plt.savefig(os.path.join(plots_dir, 'word_cloud.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning("创建词云失败: %s" % e)

def visualize_sentence_analysis(sentence_analysis, plots_dir):
    """可视化句子分析结果"""
    # 句子长度分布
    plt.figure(figsize=(12, 6))
    sent_lengths = sentence_analysis['sentence_length_distribution']
    plt.bar(sent_lengths.keys(), sent_lengths.values(), color='lightgreen')
    plt.title('句子长度分布')
    plt.xlabel('句子长度')
    plt.ylabel('数量')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plots_dir, 'sentence_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 句式结构分布
    plt.figure(figsize=(12, 8))
    patterns = sentence_analysis['sentence_pattern_distribution']
    plt.pie(patterns.values(), labels=patterns.keys(), autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('句式结构分布')
    plt.savefig(os.path.join(plots_dir, 'sentence_pattern_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 完整诗句百分比
    plt.figure(figsize=(8, 6))
    complete_percentage = sentence_analysis['complete_line_percentage'] * 100
    incomplete_percentage = 100 - complete_percentage
    plt.pie([complete_percentage, incomplete_percentage],
            labels=['完整诗句', '非完整诗句'],
            autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
    plt.axis('equal')
    plt.title('完整诗句与非完整诗句占比')
    plt.savefig(os.path.join(plots_dir, 'complete_line_percentage.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_emotion_analysis(emotion_analysis, plots_dir):
    """可视化情感分析结果"""
    # 情感分布条形图
    plt.figure(figsize=(12, 8))
    emotions = emotion_analysis['emotion_distribution']
    plt.bar(emotions.keys(), emotions.values(), color='plum')
    plt.title('情感分布')
    plt.ylabel('数量')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'emotion_bar_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 不同情感类别的诗词平均长度
    plt.figure(figsize=(12, 8))
    emotions = emotion_analysis['emotion_word_stats']
    avg_lengths = [stats_item['avg_content_length'] for _, stats_item in emotions.items()]
    plt.bar(emotions.keys(), avg_lengths, color='lightcoral')
    plt.title('不同情感类别的诗词平均长度')
    plt.ylabel('平均字符数')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'emotion_avg_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 不同情感类别的词语和句子平均数量
    plt.figure(figsize=(14, 8))
    emotions = emotion_analysis['emotion_word_stats']

    x = np.arange(len(emotions))
    width = 0.35

    avg_words = [stats_item['avg_word_count'] for _, stats_item in emotions.items()]
    avg_sents = [stats_item['avg_sent_count'] for _, stats_item in emotions.items()]

    plt.bar(x - width/2, avg_words, width, label='平均词语数', color='skyblue')
    plt.bar(x + width/2, avg_sents, width, label='平均句子数', color='salmon')

    plt.title('不同情感类别的词语和句子平均数量')
    plt.xticks(x, emotions.keys(), rotation=45, ha='right')
    plt.ylabel('平均数量')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'emotion_avg_word_sent_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 保存统计结果
def save_statistics(stats, poetry_types, word_analysis, sentence_analysis, emotion_analysis, stats_dir):
    """将统计结果保存为JSON文件"""
    # 保存基本统计信息
    with open(os.path.join(stats_dir, 'basic_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    # 保存诗词类型分析
    with open(os.path.join(stats_dir, 'poetry_types.json'), 'w', encoding='utf-8') as f:
        json.dump(poetry_types, f, ensure_ascii=False, indent=4)

    # 保存词语分析
    with open(os.path.join(stats_dir, 'word_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(word_analysis, f, ensure_ascii=False, indent=4)

    # 保存句子分析
    with open(os.path.join(stats_dir, 'sentence_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(sentence_analysis, f, ensure_ascii=False, indent=4)

    # 保存情感分析
    with open(os.path.join(stats_dir, 'emotion_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(emotion_analysis, f, ensure_ascii=False, indent=4)

    logger.info("统计结果已保存到 %s" % stats_dir)

# 生成数据集特征报告
def generate_report(stats, poetry_types, word_analysis, sentence_analysis, emotion_analysis, output_dir):
    """生成数据集特征报告文件"""
    report_path = os.path.join(output_dir, 'dataset_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 古诗词理解与推理评测数据集特征报告\n\n")

        # 基本统计信息
        f.write("## 1. 基本统计信息\n\n")
        f.write("- 样本数量: %d\n" % stats['sample_count'])
        f.write("- 诗词平均长度: %.2f 字符\n" % stats['content_length']['mean'])
        f.write("- 词语解释任务总数: %d 词语\n" % stats['word_explanation']['total_words'])
        f.write("- 唯一词语数量: %d 词语\n" % stats['word_explanation']['unique_words'])
        f.write("- 句子翻译任务总数: %d 句子\n" % stats['sentence_translation']['total_sentences'])
        f.write("- 情感分类类别数: %d\n" % stats['emotion_classification']['unique_emotions'])
        f.write("- 唯一作者数量: %d\n\n" % stats['authors']['unique_authors'])

        # 诗词类型分布
        f.write("## 2. 诗词类型分布\n\n")
        f.write("| 类型 | 数量 | 百分比 |\n")
        f.write("| --- | --- | --- |\n")
        total = sum(poetry_types.values())
        for type_name, count in sorted(poetry_types.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total * 100 if total > 0 else 0
            f.write("| %s | %d | %.2f%% |\n" % (type_name, count, percentage))
        f.write("\n")

        # 词语分析
        f.write("## 3. 词语分析\n\n")

        # 词语长度分布
        f.write("### 3.1 词语长度分布\n\n")
        f.write("| 词语长度 | 数量 | 百分比 |\n")
        f.write("| --- | --- | --- |\n")
        total_words = sum(word_analysis['word_length_distribution'].values())
        for length, count in sorted(word_analysis['word_length_distribution'].items()):
            percentage = count / total_words * 100 if total_words > 0 else 0
            f.write("| %d | %d | %.2f%% |\n" % (length, count, percentage))
        f.write("\n")

        # 高频词语
        f.write("### 3.2 高频词语（前20位）\n\n")
        f.write("| 词语 | 出现频率 |\n")
        f.write("| --- | --- |\n")
        for word, freq in list(word_analysis['word_frequency'].items())[:20]:
            f.write("| %s | %d |\n" % (word, freq))
        f.write("\n")

        # 句子分析
        f.write("## 4. 句子分析\n\n")

        # 句子长度分布
        f.write("### 4.1 句子长度分布\n\n")
        f.write("| 句子长度 | 数量 | 百分比 |\n")
        f.write("| --- | --- | --- |\n")
        total_sents = sum(sentence_analysis['sentence_length_distribution'].values())
        for length, count in sorted(sentence_analysis['sentence_length_distribution'].items()):
            percentage = count / total_sents * 100 if total_sents > 0 else 0
            f.write("| %d | %d | %.2f%% |\n" % (length, count, percentage))
        f.write("\n")

        # 句式结构分布
        f.write("### 4.2 句式结构分布\n\n")
        f.write("| 句式结构 | 数量 | 百分比 |\n")
        f.write("| --- | --- | --- |\n")
        total_patterns = sum(sentence_analysis['sentence_pattern_distribution'].values())
        for pattern, count in sorted(sentence_analysis['sentence_pattern_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_patterns * 100 if total_patterns > 0 else 0
            f.write("| %s | %d | %.2f%% |\n" % (pattern, count, percentage))
        f.write("\n")

        # 完整诗句百分比
        complete_percentage = sentence_analysis['complete_line_percentage'] * 100
        f.write("### 4.3 完整诗句占比\n\n")
        f.write("- 完整诗句百分比: %.2f%%\n" % complete_percentage)
        f.write("- 非完整诗句百分比: %.2f%%\n\n" % (100-complete_percentage))

        # 情感分析
        f.write("## 5. 情感分析\n\n")

        # 情感分布
        f.write("### 5.1 情感分布\n\n")
        f.write("| 情感类别 | 数量 | 百分比 |\n")
        f.write("| --- | --- | --- |\n")
        total_emotions = sum(emotion_analysis['emotion_distribution'].values())
        for emotion, count in sorted(emotion_analysis['emotion_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_emotions * 100 if total_emotions > 0 else 0
            f.write("| %s | %d | %.2f%% |\n" % (emotion, count, percentage))
        f.write("\n")

        # 不同情感类别的特点
        f.write("### 5.2 不同情感类别的特点\n\n")
        f.write("| 情感类别 | 样本数 | 平均诗词长度 | 平均词语数 | 平均句子数 |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for emotion, stats in emotion_analysis['emotion_word_stats'].items():
            f.write("| %s | %d | %.2f | %.2f | %.2f |\n" % (emotion, stats['sample_count'], stats['avg_content_length'], stats['avg_word_count'], stats['avg_sent_count']))

        # 添加结论和建议
        f.write("\n## 6. 结论与建议\n\n")
        f.write("基于以上分析，我们可以得出以下结论和建议：\n\n")

        # 这里可以根据实际分析结果生成一些结论和建议
        f.write("1. **数据集特点**: 数据集中包含多种类型的古诗词，以五言、七言绝句和律诗为主。\n")
        f.write("2. **词语特点**: 需要解释的词语以双字词为主，这可能是因为双字词在古汉语中往往具有特殊的文化含义。\n")
        f.write("3. **句子特点**: 大部分需要翻译的句子是完整的诗句，这表明模型需要理解完整的诗句结构。\n")
        f.write("4. **情感分析**: 不同情感类别的诗词在长度、词语数量和句子数量上存在差异，这可能是情感分类的重要特征。\n\n")

        f.write("### 对模型选择和训练的建议\n\n")
        f.write("1. **模型选择**: 建议选择具有强大中文理解能力的模型，如Qwen2-7B或ChatGLM4-9B，这些模型在中文文化背景理解上有优势。\n")
        f.write("2. **数据增强**: 考虑使用全唐诗、全宋词等公开数据集进行数据增强，以提高模型对古汉语的理解能力。\n")
        f.write("3. **微调策略**: 建议采用QLoRA等参数高效微调方法，并根据不同类型的诗词进行分层微调。\n")
        f.write("4. **评估指标**: 除了标准的BLEU和BertScore外，建议开发专门针对古汉语的评估指标，以更准确地评估模型性能。\n")

    logger.info("数据集特征报告已生成并保存到 %s" % report_path)
    return report_path

# 分析单个数据集
def analyze_single_dataset(input_file, output_dir):
    # 从路径中提取数据集名称（用于输出目录）
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    # 如果数据集在子目录中，将子目录名也包含在数据集名称中
    parent_dir = os.path.basename(os.path.dirname(input_file))
    if parent_dir != 'data':
        dataset_name = "%s_%s" % (parent_dir, dataset_name)

    # 创建该数据集的输出目录
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    plots_dir, stats_dir = create_output_dir(dataset_output_dir)

    # 加载数据集
    try:
        data = load_dataset(input_file)
    except Exception as e:
        logger.error("加载数据集 %s 失败: %s" % (input_file, e))
        return None

    # 计算基本统计信息
    logger.info("计算 %s 的基本统计信息..." % dataset_name)
    stats = basic_statistics(data)

    # 分析诗词类型
    logger.info("分析 %s 的诗词类型..." % dataset_name)
    poetry_types = analyze_poetry_types(data)

    # 分析词语
    logger.info("分析 %s 的词语特点..." % dataset_name)
    word_analysis = analyze_words(data)

    # 分析句子
    logger.info("分析 %s 的句子特点..." % dataset_name)
    sentence_analysis = analyze_sentences(data)

    # 分析情感
    logger.info("分析 %s 的情感类别..." % dataset_name)
    emotion_analysis = analyze_emotions(data)

    # 保存统计结果
    logger.info("保存 %s 的统计结果..." % dataset_name)
    save_statistics(stats, poetry_types, word_analysis, sentence_analysis, emotion_analysis, stats_dir)

    # 生成报告
    logger.info("生成 %s 的数据集特征报告..." % dataset_name)
    report_path = generate_report(stats, poetry_types, word_analysis, sentence_analysis, emotion_analysis, dataset_output_dir)

    # 生成可视化图表
    logger.info("生成 %s 的可视化图表..." % dataset_name)
    visualize_basic_stats(stats, plots_dir)
    visualize_poetry_types(poetry_types, plots_dir)
    visualize_word_analysis(word_analysis, plots_dir)
    visualize_sentence_analysis(sentence_analysis, plots_dir)
    visualize_emotion_analysis(emotion_analysis, plots_dir)

    logger.info("%s 分析完成！结果已保存到 %s" % (dataset_name, dataset_output_dir))
    return {
        'dataset_name': dataset_name,
        'stats': stats,
        'poetry_types': poetry_types,
        'word_analysis': word_analysis,
        'sentence_analysis': sentence_analysis,
        'emotion_analysis': emotion_analysis,
        'report_path': report_path
    }

# 生成汇总报告
def generate_summary_report(all_results, output_dir):
    summary_path = os.path.join(output_dir, 'summary_report.md')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 古诗词数据集分析汇总报告\n\n")

        # 数据集概述
        f.write("## 1. 数据集概述\n\n")
        f.write("| 数据集 | 样本数量 | 词语总数 | 句子总数 | 情感类别数 |\n")
        f.write("| --- | --- | --- | --- | --- |\n")

        for result in all_results:
            if not result:  # 跳过失败的分析
                continue
            dataset_name = result['dataset_name']
            stats = result['stats']
            f.write("| %s | %d | %d | %d | %d |\n" % (dataset_name, stats['sample_count'], stats['word_explanation']['total_words'], stats['sentence_translation']['total_sentences'], stats['emotion_classification']['unique_emotions']))

        # 诗词类型分布比较
        f.write("\n## 2. 诗词类型分布比较\n\n")

        # 收集所有数据集的诗词类型
        all_types = set()
        for result in all_results:
            if not result:  # 跳过失败的分析
                continue
            all_types.update(result['poetry_types'].keys())

        # 创建表格头
        f.write("| 数据集 | " + " | ".join(all_types) + " |\n")
        f.write("| --- | " + " | ".join(["---" for _ in all_types]) + " |\n")

        # 填充每个数据集的诗词类型分布
        for result in all_results:
            if not result:  # 跳过失败的分析
                continue
            dataset_name = result['dataset_name']
            poetry_types = result['poetry_types']
            total = sum(poetry_types.values())

            row = [dataset_name]
            for type_name in all_types:
                count = poetry_types.get(type_name, 0)
                percentage = count / total * 100 if total > 0 else 0
                row.append("%.1f%%" % percentage)

            f.write("| %s |\n" % " | ".join(row))

        # 情感分布比较
        f.write("\n## 3. 情感分布比较\n\n")

        # 收集所有数据集的情感类别
        all_emotions = set()
        for result in all_results:
            if not result:  # 跳过失败的分析
                continue
            all_emotions.update(result['emotion_analysis']['emotion_distribution'].keys())

        # 创建表格头
        f.write("| 数据集 | " + " | ".join(all_emotions) + " |\n")
        f.write("| --- | " + " | ".join(["---" for _ in all_emotions]) + " |\n")

        # 填充每个数据集的情感分布
        for result in all_results:
            if not result:  # 跳过失败的分析
                continue
            dataset_name = result['dataset_name']
            emotion_dist = result['emotion_analysis']['emotion_distribution']
            total = sum(emotion_dist.values())

            row = [dataset_name]
            for emotion in all_emotions:
                count = emotion_dist.get(emotion, 0)
                percentage = count / total * 100 if total > 0 else 0
                row.append("%.1f%%" % percentage)

            f.write("| %s |\n" % " | ".join(row))

        # 添加结论和建议
        f.write("\n## 4. 结论与建议\n\n")
        f.write("基于对多个数据集的分析，我们可以得出以下结论和建议：\n\n")
        f.write("1. **数据集差异**: 不同数据集之间存在显著的差异，这可能影响模型的泛化能力。\n")
        f.write("2. **训练策略**: 建议采用多数据集联合训练策略，以提高模型对不同类型诗词的理解能力。\n")
        f.write("3. **数据平衡**: 在训练时应注意平衡不同类型的诗词和情感类别，避免模型偏向主要类别。\n")
        f.write("4. **评估方法**: 建议在多个数据集上进行交叉验证，以全面评估模型性能。\n")

    logger.info("汇总报告已生成并保存到 %s" % summary_path)
    return summary_path

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()

    # 确定要分析的数据集
    if args.input:
        # 如果指定了输入文件，则只分析该文件
        input_files = [args.input]
    else:
        # 否则分析data目录下的所有JSON文件
        input_files = get_all_json_files("@data")

    # 创建主输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 分析每个数据集
    all_results = []
    for input_file in input_files:
        logger.info("\n开始分析数据集: %s" % input_file)
        result = analyze_single_dataset(input_file, args.output_dir)
        all_results.append(result)

    # 生成汇总报告
    if len(all_results) > 1:
        logger.info("\n生成汇总报告...")
        summary_path = generate_summary_report(all_results, args.output_dir)
        logger.info("\n分析完成！汇总报告: %s" % summary_path)
    else:
        logger.info("\n分析完成！")

if __name__ == "__main__":
    main()
