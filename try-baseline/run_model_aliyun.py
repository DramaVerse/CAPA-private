import json
import torch
import re
import argparse
import traceback
import os
import psutil
import sys
from tqdm import tqdm
from modelscope.models.nlp import SbertModel
from modelscope.pipelines import pipeline
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer

def load_json(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def check_system_resources():
    """检查系统资源使用情况"""
    try:
        # 获取内存信息
        memory = psutil.virtual_memory()
        print(f"系统内存情况:")
        print(f"  总内存: {memory.total / (1024**3):.2f} GB")
        print(f"  可用内存: {memory.available / (1024**3):.2f} GB")
        print(f"  内存使用率: {memory.percent}%")

        # 获取CPU信息
        print(f"CPU使用率: {psutil.cpu_percent()}%")

        # 获取当前进程信息
        process = psutil.Process(os.getpid())
        print(f"当前进程内存使用: {process.memory_info().rss / (1024**3):.2f} GB")

        # 尝试获取GPU信息（如果有torch且有GPU）
        if torch.cuda.is_available():
            print(f"GPU信息:")
            print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            print(f"  当前分配的GPU内存: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
            print(f"  当前缓存的GPU内存: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    except Exception as e:
        print(f"获取系统资源信息失败: {e}")

def process_poetry_data(data, pipe, max_samples=5):
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

        # 使用ModelScope pipeline生成回答
        result = pipe(prompt)
        response = result['response']

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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行Qwen模型进行古诗词理解")
    parser.add_argument("--input", type=str, default="test_data.json", help="输入数据文件路径")
    parser.add_argument("--output", type=str, default="baseline_output.json", help="输出结果文件路径")
    parser.add_argument("--samples", type=int, default=5, help="处理的样本数量")
    parser.add_argument("--use_8bit", action="store_true", help="使用8位量化加载模型，减少内存使用")
    parser.add_argument("--debug", action="store_true", help="启用详细调试信息")
    args = parser.parse_args()

    # 检查系统资源
    print("检查系统资源...")
    check_system_resources()

    # 使用ModelScope的模型
    print("正在加载模型和分词器...")
    print("这可能需要几分钟时间...")

    # 使用ModelScope的pipeline API
    try:
        # 准备模型参数
        model_kwargs = {}

        # 如果启用8位量化
        if args.use_8bit:
            print("启用8位量化，这将减少内存使用但可能略微降低模型性能")
            model_kwargs['load_in_8bit'] = True
            model_kwargs['device_map'] = 'auto'

        # 尝试使用Qwen-7B-Chat
        print(f"正在加载Qwen-7B-Chat模型...")
        if args.debug:
            print(f"模型参数: {model_kwargs}")

        try:
            pipe = pipeline(
                task='chat',
                model='qwen/Qwen-7B-Chat',
                model_revision='v1.0.5',
                model_kwargs=model_kwargs
            )
            print("Qwen-7B-Chat模型加载成功!")
        except Exception as e:
            print(f"加载Qwen-7B-Chat失败，详细错误信息:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            if args.debug:
                print("完整错误堆栈:")
                traceback.print_exc()

            print("\n尝试加载Qwen-1.8B-Chat...")
            # 如果失败，尝试使用更小的模型
            pipe = pipeline(
                task='chat',
                model='qwen/Qwen-1.8B-Chat',
                model_revision='v1.0.0',
                model_kwargs=model_kwargs
            )
            print("Qwen-1.8B-Chat模型加载成功!")

        # 再次检查资源使用情况
        print("\n模型加载后的系统资源使用情况:")
        check_system_resources()

        # 加载测试数据
        data = load_json(args.input)
        print(f"成功加载数据，共{len(data)}条")

        # 处理数据
        result_data = process_poetry_data(data, pipe, max_samples=args.samples)

        # 保存结果
        save_json(result_data, args.output)
        print(f"处理完成。结果已保存到 {args.output}")

    except Exception as e:
        print(f"\n程序执行过程中发生错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n详细错误堆栈:")
        traceback.print_exc()

        print("\n系统资源状态:")
        check_system_resources()
        sys.exit(1)

if __name__ == "__main__":
    main()
