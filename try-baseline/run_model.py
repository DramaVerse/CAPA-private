import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re
import argparse

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
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行Qwen2.5-7B模型进行古诗词理解")
    parser.add_argument("--input", type=str, default="test_data.json", help="输入数据文件路径")
    parser.add_argument("--output", type=str, default="baseline_output.json", help="输出结果文件路径")
    parser.add_argument("--samples", type=int, default=5, help="处理的样本数量")
    args = parser.parse_args()
    
    # 模型路径
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 加载模型和分词器
    print("正在加载模型和分词器...")
    print("这可能需要几分钟时间，模型大小约为14GB...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # 使用bfloat16以节省显存
    )
    print("模型加载成功!")
    
    # 加载测试数据
    data = load_json(args.input)
    print(f"成功加载数据，共{len(data)}条")
    
    # 处理数据
    result_data = process_poetry_data(data, model, tokenizer, max_samples=args.samples)
    
    # 保存结果
    save_json(result_data, args.output)
    print(f"处理完成。结果已保存到 {args.output}")

if __name__ == "__main__":
    main()
