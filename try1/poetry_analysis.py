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