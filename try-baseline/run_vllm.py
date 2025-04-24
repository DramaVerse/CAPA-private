import json
import os
from vllm import LLM, SamplingParams
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

def process_poetry_data_vllm(data, llm, max_samples=5):
    """使用vLLM处理诗词数据"""
    results = []
    prompts = []
    
    # 只处理指定数量的样本（用于测试）
    data = data[:max_samples]
    
    # 准备所有提示
    for item in data:
        prompt = f"""
        <|im_start|>system
        你是一个古诗词专家，擅长解释古诗词中的词语、翻译句子和分析情感。
        <|im_end|>
        <|im_start|>user
        我会给你提供一个 JSON 数据，格式如下：
        - **"title"**：古诗词的标题  
        - **"author"**：古诗词的作者（如果有）  
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
        <|im_end|>
        <|im_start|>assistant
        """
        prompts.append(prompt)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024
    )
    
    # 批量生成
    print("正在生成回答...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 处理输出
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        
        # 尝试提取JSON
        try:
            # 使用正则表达式提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                results.append(result)
                print(f"成功处理: {data[i].get('title', '无标题')}")
            else:
                print(f"无法提取JSON: {data[i].get('title', '无标题')}")
                print(f"原始响应: {response}")
        except json.JSONDecodeError:
            print(f"JSON解析错误: {data[i].get('title', '无标题')}")
            print(f"原始响应: {response}")
    
    return results

def main():
    # 模型路径
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 加载vLLM模型
    print("正在加载vLLM模型...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,  # 使用1个GPU
        trust_remote_code=True,
        dtype="bfloat16"  # 使用bfloat16以节省显存
    )
    print("模型加载成功!")
    
    # 加载测试数据
    input_file = "test_data.json"
    output_file = "vllm_output.json"
    
    data = load_json(input_file)
    print(f"成功加载数据，共{len(data)}条")
    
    # 处理数据
    result_data = process_poetry_data_vllm(data, llm)
    
    # 保存结果
    save_json(result_data, output_file)
    print(f"处理完成。结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
