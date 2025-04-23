#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 对于macOS
plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('seaborn')  # 使用seaborn样式美化图表

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poetry_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PoetryAnalyzer:
    def __init__(self, data_dir: str = "../data"):
        """初始化诗词分析器
        
        Args:
            data_dir: 数据目录的路径
        """
        self.data_dir = Path(data_dir)
        self.poetry_data = defaultdict(list)
        self.analysis_results = {}
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> None:
        """加载所有诗词数据"""
        logger.info("开始加载数据...")
        
        # 加载唐诗
        tang_poetry_dir = self.data_dir / "唐诗"
        for category in ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]:
            category_dir = tang_poetry_dir / category
            if category_dir.exists():
                for file in category_dir.glob("*.json"):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for poem in data:
                                poem['category'] = category
                                poem['type'] = '唐诗'
                                self.poetry_data['唐诗'].append(poem)
                    except Exception as e:
                        logger.error(f"加载文件 {file} 时出错: {e}")
        
        # 加载宋词
        song_ci_dir = self.data_dir / "宋词"
        for period in ["北宋词", "南宋词", "唐五代词"]:
            period_dir = song_ci_dir / period
            if period_dir.exists():
                for file in period_dir.glob("*.json"):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for poem in data:
                                poem['period'] = period
                                poem['type'] = '宋词'
                                self.poetry_data['宋词'].append(poem)
                    except Exception as e:
                        logger.error(f"加载文件 {file} 时出错: {e}")
        
        logger.info(f"数据加载完成。唐诗: {len(self.poetry_data['唐诗'])}首，宋词: {len(self.poetry_data['宋词'])}首")

    def analyze_basic_stats(self) -> None:
        """分析基本统计信息"""
        logger.info("开始分析基本统计信息...")
        
        stats = {
            '总数据量': {
                '唐诗': len(self.poetry_data['唐诗']),
                '宋词': len(self.poetry_data['宋词'])
            },
            '唐诗分类统计': Counter(poem['category'] for poem in self.poetry_data['唐诗']),
            '宋词时期统计': Counter(poem['period'] for poem in self.poetry_data['宋词']),
            '作者统计': {
                '唐诗': Counter(poem.get('author', '佚名') for poem in self.poetry_data['唐诗']),
                '宋词': Counter(poem.get('author', '佚名') for poem in self.poetry_data['宋词'])
            }
        }
        
        # 分析词语和句子数量
        word_counts = {'唐诗': [], '宋词': []}
        sent_counts = {'唐诗': [], '宋词': []}
        emotion_options = {'唐诗': [], '宋词': []}
        
        # 处理唐诗
        for poem in self.poetry_data['唐诗']:
            # 处理词语解释 - 可能是keywords或qa_words字段
            if 'keywords' in poem:
                word_counts['唐诗'].append(len(poem['keywords']))
            elif 'qa_words' in poem:
                word_counts['唐诗'].append(len(poem['qa_words']))
                
            # 处理句子翻译 - 可能是trans或qa_sents字段
            if 'trans' in poem:
                # 如果trans是字符串，计为1句，否则按列表长度计算
                if isinstance(poem['trans'], str):
                    sent_counts['唐诗'].append(1)
                else:
                    sent_counts['唐诗'].append(len(poem['trans']))
            elif 'qa_sents' in poem:
                sent_counts['唐诗'].append(len(poem['qa_sents']))
                
            # 处理情感分类 - 可能是emotion或choose字段
            if 'emotion' in poem:
                # 处理emotion字段，可能是字符串或列表或字典
                if isinstance(poem['emotion'], str):
                    emotions = [e.strip() for e in poem['emotion'].split('、')]
                    emotion_options['唐诗'].extend(emotions)
                elif isinstance(poem['emotion'], list):
                    emotion_options['唐诗'].extend(poem['emotion'])
                elif isinstance(poem['emotion'], dict):
                    emotion_options['唐诗'].extend(poem['emotion'].values())
            elif 'choose' in poem:
                if isinstance(poem['choose'], dict):
                    emotion_options['唐诗'].extend(poem['choose'].values())
        
        # 处理宋词 - 与处理唐诗类似
        for poem in self.poetry_data['宋词']:
            # 处理词语解释
            if 'keywords' in poem:
                word_counts['宋词'].append(len(poem['keywords']))
            elif 'qa_words' in poem:
                word_counts['宋词'].append(len(poem['qa_words']))
                
            # 处理句子翻译
            if 'trans' in poem:
                if isinstance(poem['trans'], str):
                    sent_counts['宋词'].append(1)
                else:
                    sent_counts['宋词'].append(len(poem['trans']))
            elif 'qa_sents' in poem:
                sent_counts['宋词'].append(len(poem['qa_sents']))
                
            # 处理情感分类
            if 'emotion' in poem:
                if isinstance(poem['emotion'], str):
                    emotions = [e.strip() for e in poem['emotion'].split('、')]
                    emotion_options['宋词'].extend(emotions)
                elif isinstance(poem['emotion'], list):
                    emotion_options['宋词'].extend(poem['emotion'])
                elif isinstance(poem['emotion'], dict):
                    emotion_options['宋词'].extend(poem['emotion'].values())
            elif 'choose' in poem:
                if isinstance(poem['choose'], dict):
                    emotion_options['宋词'].extend(poem['choose'].values())
        
        # 统计平均数
        stats['词语和句子统计'] = {
            '唐诗': {
                '平均解释词语数': np.mean(word_counts['唐诗']) if word_counts['唐诗'] else 0,
                '平均翻译句子数': np.mean(sent_counts['唐诗']) if sent_counts['唐诗'] else 0,
            },
            '宋词': {
                '平均解释词语数': np.mean(word_counts['宋词']) if word_counts['宋词'] else 0,
                '平均翻译句子数': np.mean(sent_counts['宋词']) if sent_counts['宋词'] else 0,
            },
            '总体': {
                '平均解释词语数': np.mean(word_counts['唐诗'] + word_counts['宋词']) if (word_counts['唐诗'] + word_counts['宋词']) else 0,
                '平均翻译句子数': np.mean(sent_counts['唐诗'] + sent_counts['宋词']) if (sent_counts['唐诗'] + sent_counts['宋词']) else 0,
            }
        }
        
        # 情感分类统计
        stats['情感分类统计'] = {
            '唐诗': Counter(emotion_options['唐诗']),
            '宋词': Counter(emotion_options['宋词']),
            '总体': Counter(emotion_options['唐诗'] + emotion_options['宋词'])
        }
        
        self.analysis_results['basic_stats'] = stats
        logger.info("基本统计信息分析完成")

    def analyze_text_features(self) -> None:
        """分析文本特征"""
        logger.info("开始分析文本特征...")
        
        def get_text_content(poem):
            """获取诗词的文本内容"""
            content = poem.get('content', '')
            # 移除标点符号和空白字符
            content = ''.join(char for char in content if char.strip() and not char in '，。！？；：、（）《》""''')
            return content
        
        text_features = {
            '唐诗': {
                '平均字数': np.mean([len(get_text_content(poem)) for poem in self.poetry_data['唐诗']]),
                '字数分布': Counter(len(get_text_content(poem)) for poem in self.poetry_data['唐诗']),
                '常用字': Counter(''.join(get_text_content(poem) for poem in self.poetry_data['唐诗']))
            },
            '宋词': {
                '平均字数': np.mean([len(get_text_content(poem)) for poem in self.poetry_data['宋词']]),
                '字数分布': Counter(len(get_text_content(poem)) for poem in self.poetry_data['宋词']),
                '常用字': Counter(''.join(get_text_content(poem) for poem in self.poetry_data['宋词']))
            }
        }
        
        self.analysis_results['text_features'] = text_features
        logger.info("文本特征分析完成")

    def generate_visualizations(self) -> None:
        """生成可视化图表"""
        logger.info("开始生成可视化图表...")
        
        # 1. 诗词数量分布饼图
        plt.figure(figsize=(10, 6))
        plt.pie(
            [len(self.poetry_data['唐诗']), len(self.poetry_data['宋词'])],
            labels=['唐诗', '宋词'],
            autopct='%1.1f%%'
        )
        plt.title('唐诗宋词数量分布')
        plt.savefig(self.output_dir / 'poetry_distribution_pie.png')
        plt.close()
        
        # 2. 唐诗分类统计条形图
        plt.figure(figsize=(12, 6))
        tang_categories = self.analysis_results['basic_stats']['唐诗分类统计']
        plt.bar(tang_categories.keys(), tang_categories.values())
        plt.title('唐诗分类统计')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tang_poetry_categories.png')
        plt.close()
        
        # 3. 宋词时期统计条形图
        plt.figure(figsize=(12, 6))
        song_periods = self.analysis_results['basic_stats']['宋词时期统计']
        plt.bar(song_periods.keys(), song_periods.values())
        plt.title('宋词时期统计')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'song_ci_periods.png')
        plt.close()
        
        # 4. 字数分布直方图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 唐诗字数分布
        tang_lengths = list(self.analysis_results['text_features']['唐诗']['字数分布'].keys())
        tang_counts = list(self.analysis_results['text_features']['唐诗']['字数分布'].values())
        ax1.hist(np.repeat(tang_lengths, tang_counts), bins=30)
        ax1.set_title('唐诗字数分布')
        ax1.set_xlabel('字数')
        ax1.set_ylabel('频数')
        
        # 宋词字数分布
        song_lengths = list(self.analysis_results['text_features']['宋词']['字数分布'].keys())
        song_counts = list(self.analysis_results['text_features']['宋词']['字数分布'].values())
        ax2.hist(np.repeat(song_lengths, song_counts), bins=30)
        ax2.set_title('宋词字数分布')
        ax2.set_xlabel('字数')
        ax2.set_ylabel('频数')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'length_distribution.png')
        plt.close()
        
        # 5. 情感分类选项分布图
        if self.analysis_results['basic_stats'].get('情感分类统计', {}).get('总体'):
            # 转换为DataFrame并排序，只取前15个最常见的情感
            total_emotions = pd.DataFrame.from_dict(
                self.analysis_results['basic_stats']['情感分类统计']['总体'], 
                orient='index', 
                columns=['count']
            ).sort_values('count', ascending=False).head(15)
            
            if not total_emotions.empty:
                plt.figure(figsize=(16, 8))
                ax = total_emotions.plot(kind='bar', figsize=(16, 8))
                plt.title('前15种情感分类分布', fontsize=16)
                plt.xlabel('情感类别', fontsize=14)
                plt.ylabel('出现次数', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                
                # 为每个柱添加数值标签
                for i, v in enumerate(total_emotions['count']):
                    ax.text(i, v + 0.1, str(v), ha='center', fontsize=10)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'emotion_distribution.png', dpi=300)
                plt.close()
                
                # 分别绘制唐诗和宋词的情感分布（前15种）
                tang_emotions = pd.DataFrame.from_dict(
                    self.analysis_results['basic_stats']['情感分类统计']['唐诗'], 
                    orient='index', 
                    columns=['count']
                ).sort_values('count', ascending=False).head(15)
                
                song_emotions = pd.DataFrame.from_dict(
                    self.analysis_results['basic_stats']['情感分类统计']['宋词'], 
                    orient='index', 
                    columns=['count']
                ).sort_values('count', ascending=False).head(15)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
                
                if not tang_emotions.empty:
                    tang_emotions.plot(kind='bar', ax=ax1)
                    ax1.set_title('唐诗前15种情感分类分布', fontsize=16)
                    ax1.set_xlabel('情感类别', fontsize=14)
                    ax1.set_ylabel('出现次数', fontsize=14)
                    ax1.tick_params(axis='x', rotation=45, labelsize=12)
                    ax1.tick_params(axis='y', labelsize=12)
                    
                    # 为每个柱添加数值标签
                    for i, v in enumerate(tang_emotions['count']):
                        ax1.text(i, v + 0.1, str(v), ha='center', fontsize=10)
                
                if not song_emotions.empty:
                    song_emotions.plot(kind='bar', ax=ax2)
                    ax2.set_title('宋词前15种情感分类分布', fontsize=16)
                    ax2.set_xlabel('情感类别', fontsize=14)
                    ax2.set_ylabel('出现次数', fontsize=14)
                    ax2.tick_params(axis='x', rotation=45, labelsize=12)
                    ax2.tick_params(axis='y', labelsize=12)
                    
                    # 为每个柱添加数值标签
                    for i, v in enumerate(song_emotions['count']):
                        ax2.text(i, v + 0.1, str(v), ha='center', fontsize=10)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'emotion_distribution_by_type.png', dpi=300)
                plt.close()
        
        logger.info("可视化图表生成完成")

    def generate_report(self) -> None:
        """生成分析报告"""
        logger.info("开始生成分析报告...")
        
        # 将Counter对象转换为DataFrame以便排序
        tang_chars = pd.DataFrame.from_dict(self.analysis_results['text_features']['唐诗']['常用字'], 
                                          orient='index', 
                                          columns=['count']).sort_values('count', ascending=False)
        song_chars = pd.DataFrame.from_dict(self.analysis_results['text_features']['宋词']['常用字'], 
                                          orient='index', 
                                          columns=['count']).sort_values('count', ascending=False)
        
        # 格式化情感分类数据，并按出现频率降序排序
        emotion_stats = ""
        if '情感分类统计' in self.analysis_results['basic_stats']:
            # 将情感计数转换为DataFrame并排序
            total_emotions_df = pd.DataFrame.from_dict(
                self.analysis_results['basic_stats']['情感分类统计']['总体'], 
                orient='index', 
                columns=['count']
            ).sort_values('count', ascending=False)
            
            tang_emotions_df = pd.DataFrame.from_dict(
                self.analysis_results['basic_stats']['情感分类统计']['唐诗'], 
                orient='index', 
                columns=['count']
            ).sort_values('count', ascending=False)
            
            song_emotions_df = pd.DataFrame.from_dict(
                self.analysis_results['basic_stats']['情感分类统计']['宋词'], 
                orient='index', 
                columns=['count']
            ).sort_values('count', ascending=False)
            
            emotion_stats = f"""
### 1.4 情感分类统计

#### 1.4.1 总体前20种情感分布:
{total_emotions_df.head(20).to_string()}

#### 1.4.2 唐诗前15种情感分布:
{tang_emotions_df.head(15).to_string()}

#### 1.4.3 宋词前15种情感分布:
{song_emotions_df.head(15).to_string()}
"""
        
        # 词语和句子统计
        qa_stats = ""
        if '词语和句子统计' in self.analysis_results['basic_stats']:
            qa_stats = f"""
### 1.5 词语和句子统计
- 总体平均需要解释的词语数量: {self.analysis_results['basic_stats']['词语和句子统计']['总体']['平均解释词语数']:.2f}
- 总体平均需要翻译的句子数量: {self.analysis_results['basic_stats']['词语和句子统计']['总体']['平均翻译句子数']:.2f}

- 唐诗平均需要解释的词语数量: {self.analysis_results['basic_stats']['词语和句子统计']['唐诗']['平均解释词语数']:.2f}
- 唐诗平均需要翻译的句子数量: {self.analysis_results['basic_stats']['词语和句子统计']['唐诗']['平均翻译句子数']:.2f}

- 宋词平均需要解释的词语数量: {self.analysis_results['basic_stats']['词语和句子统计']['宋词']['平均解释词语数']:.2f}
- 宋词平均需要翻译的句子数量: {self.analysis_results['basic_stats']['词语和句子统计']['宋词']['平均翻译句子数']:.2f}
"""
        
        report = f"""
# 古诗词数据分析报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 基本统计信息

### 1.1 数据量
- 总数据量: {sum(self.analysis_results['basic_stats']['总数据量'].values())}首
  - 唐诗: {self.analysis_results['basic_stats']['总数据量']['唐诗']}首
  - 宋词: {self.analysis_results['basic_stats']['总数据量']['宋词']}首

### 1.2 唐诗分类统计
{pd.Series(self.analysis_results['basic_stats']['唐诗分类统计']).to_string()}

### 1.3 宋词时期统计
{pd.Series(self.analysis_results['basic_stats']['宋词时期统计']).to_string()}
{emotion_stats}
{qa_stats}

## 2. 文本特征分析

### 2.1 字数统计
- 唐诗平均字数: {self.analysis_results['text_features']['唐诗']['平均字数']:.2f}字
- 宋词平均字数: {self.analysis_results['text_features']['宋词']['平均字数']:.2f}字

### 2.2 常用字统计（top 20）
唐诗常用字:
{tang_chars.head(20).to_string()}

宋词常用字:
{song_chars.head(20).to_string()}

## 3. 可视化图表
所有可视化图表已保存在 analysis_output 目录下：
- poetry_distribution_pie.png: 唐诗宋词数量分布饼图
- tang_poetry_categories.png: 唐诗分类统计条形图
- song_ci_periods.png: 宋词时期统计条形图
- length_distribution.png: 字数分布直方图
- emotion_distribution.png: 情感分类选项分布图（前15种情感）
- emotion_distribution_by_type.png: 唐诗宋词情感分布对比图（前15种情感）
"""
        
        with open(self.output_dir / 'analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("分析报告生成完成")

    def run_analysis(self) -> None:
        """运行完整的分析流程"""
        self.load_data()
        self.analyze_basic_stats()
        self.analyze_text_features()
        self.generate_visualizations()
        self.generate_report()
        logger.info("分析完成！报告和可视化结果已保存到 analysis_output 目录")

if __name__ == "__main__":
    analyzer = PoetryAnalyzer()
    analyzer.run_analysis() 