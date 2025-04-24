import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

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
        # 构建优化后的提示
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
    # 模型路径
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 加载模型和分词器
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # 使用bfloat16以节省显存
    )
    print("模型加载成功!")
    
    # 加载测试数据
    input_file = "test_data.json"
    output_file = "optimized_output.json"
    
    data = load_json(input_file)
    print(f"成功加载数据，共{len(data)}条")
    
    # 处理数据
    result_data = process_poetry_data(data, model, tokenizer)
    
    # 保存结果
    save_json(result_data, output_file)
    print(f"处理完成。结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
