import json
import torch
import re
import argparse
import os
import traceback
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    import psutil
    process = psutil.Process(os.getpid())
    print(f"进程内存使用: {process.memory_info().rss / 1024**3:.2f} GB")

def process_poetry_data(data, model, tokenizer, max_samples=5, max_new_tokens=512):
    """处理诗词数据并使用模型生成回答"""
    results = []
    
    # 只处理指定数量的样本
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

        # 使用Transformers直接生成
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 将输入移动到模型所在的设备
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 解码生成的文本
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]  # 移除输入提示
        
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
    parser = argparse.ArgumentParser(description="使用量化模型进行古诗词理解")
    parser.add_argument("--input", type=str, default="test_data.json", help="输入数据文件路径")
    parser.add_argument("--output", type=str, default="quantized_output.json", help="输出结果文件路径")
    parser.add_argument("--samples", type=int, default=5, help="处理的样本数量")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="量化位数 (4 或 8)")
    parser.add_argument("--cpu", action="store_true", help="使用CPU运行模型")
    parser.add_argument("--max_tokens", type=int, default=512, help="生成的最大token数")
    args = parser.parse_args()
    
    print(f"正在加载量化模型 (使用 {args.bits} 位量化)...")
    
    try:
        # 安装必要的库
        try:
            import bitsandbytes as bnb
            print("bitsandbytes 已安装")
        except ImportError:
            print("安装 bitsandbytes...")
            os.system("pip install bitsandbytes")
            import bitsandbytes as bnb
        
        # 设置量化参数
        quantization_config = {
            "load_in_4bit": args.bits == 4,
            "load_in_8bit": args.bits == 8,
        }
        
        # 设置设备
        device_map = "cpu" if args.cpu else "auto"
        print(f"使用设备: {device_map}")
        
        # 加载模型和分词器
        model_name = "qwen/Qwen-7B-Chat"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            **quantization_config
        )
        
        print("模型加载成功!")
        print_memory_usage()
        
        # 加载测试数据
        data = load_json(args.input)
        print(f"成功加载数据，共{len(data)}条")
        
        # 处理数据
        result_data = process_poetry_data(
            data, 
            model, 
            tokenizer, 
            max_samples=args.samples,
            max_new_tokens=args.max_tokens
        )
        
        # 保存结果
        save_json(result_data, args.output)
        print(f"处理完成。结果已保存到 {args.output}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
