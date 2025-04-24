import json
import torch
import re
import argparse
import os
import traceback
import psutil
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer

def load_json(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def print_memory_usage():
    """打印内存使用情况"""
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    process = psutil.Process(os.getpid())
    print(f"进程内存使用: {process.memory_info().rss / 1024**3:.2f} GB")

def prepare_competition_data(data, max_samples=5):
    """准备符合竞赛要求的数据"""
    competition_data = []
    
    # 只处理指定数量的样本
    data = data[:max_samples]
    
    for idx, item in enumerate(data):
        # 创建竞赛格式的数据项
        comp_item = {
            "idx": idx,
            "title": item.get("title", ""),
            "author": item.get("author", ""),
            "content": item.get("content", ""),
            "qa_words": list(item.get("keywords", {}).keys()),  # 只取关键词的键，不包含释义
            "qa_sents": [item.get("content", "")],  # 使用整首诗作为需要翻译的句子
            "choose": ["A:思乡", "B:离别", "C:怀旧", "D:爱国", "E:悲伤"]  # 示例情感选项
        }
        competition_data.append(comp_item)
    
    return competition_data

def process_competition_data(data, model, tokenizer, max_new_tokens=512):
    """处理竞赛数据并使用模型生成回答"""
    results = []
    
    for item in tqdm(data, desc="处理诗词"):
        # 构建提示
        prompt = f"""
        你是一个古诗词专家，现在有一些古诗词需要你的帮助。
        
        我会给你提供一个古诗词的标题、作者和内容，以及需要你解释的词语和句子。
        请你根据你的知识，解释这些词语，翻译这些句子，并判断诗词表达的情感。
        
        标题: {item.get('title', '无标题')}
        作者: {item.get('author', '佚名')}
        内容: {item.get('content', '')}
        
        需要解释的词语: {', '.join(item.get('qa_words', []))}
        需要翻译的句子: {item.get('qa_sents', [''])[0]}
        
        情感选项: {', '.join(item.get('choose', []))}
        
        请你生成如下JSON格式的结果:
        {{
            "idx": {item.get('idx', 0)},
            "ans_qa_words": {{
                "词语1": "词语1的解释",
                "词语2": "词语2的解释",
                ...
            }},
            "ans_qa_sents": {{
                "句子1": "句子1的白话文翻译",
                "句子2": "句子2的白话文翻译",
                ...
            }},
            "choose_id": "最符合该诗词情感的选项ID（如A、B、C、D或E）"
        }}
        
        请确保输出是有效的JSON格式，不要包含任何额外的解释或注释。
        """

        # 构建消息格式
        messages = [
            {"role": "system", "content": "你是一个古诗词专家，擅长解释古诗词中的词语、翻译古诗词，并分析诗词情感。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用chat模板格式化输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 解码生成的文本
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
        
        # 打印内存使用情况
        print_memory_usage()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="使用预量化模型进行古诗词理解（竞赛版）")
    parser.add_argument("--input", type=str, default="test_data.json", help="输入数据文件路径")
    parser.add_argument("--output", type=str, default="competition_output.json", help="输出结果文件路径")
    parser.add_argument("--samples", type=int, default=5, help="处理的样本数量")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8", 
                        help="模型名称，可选 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8' 或 'lmstudio-community/Qwen2.5-7B-Instruct-MLX-8bit'")
    parser.add_argument("--max_tokens", type=int, default=512, help="生成的最大token数")
    args = parser.parse_args()
    
    print(f"正在加载预量化模型: {args.model}")
    print("检查系统资源...")
    print_memory_usage()
    
    try:
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("模型加载成功!")
        print("模型加载后的系统资源使用情况:")
        print_memory_usage()
        
        # 加载测试数据
        raw_data = load_json(args.input)
        print(f"成功加载原始数据，共{len(raw_data)}条")
        
        # 准备竞赛格式的数据
        competition_data = prepare_competition_data(raw_data, max_samples=args.samples)
        print(f"已准备竞赛格式数据，共{len(competition_data)}条")
        
        # 处理数据
        result_data = process_competition_data(
            competition_data, 
            model, 
            tokenizer, 
            max_new_tokens=args.max_tokens
        )
        
        # 保存结果
        save_json(result_data, args.output)
        print(f"处理完成。结果已保存到 {args.output}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("详细错误堆栈:")
        traceback.print_exc()
        print("\n系统资源状态:")
        print_memory_usage()

if __name__ == "__main__":
    main()
