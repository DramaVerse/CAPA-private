import re
import json

def fix_json_string(json_str):
    """尝试修复常见的JSON格式错误"""
    # 移除可能的前缀和后缀文本
    json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    
    # 修复未转义的引号
    json_str = re.sub(r'(?<!\\)"(?=.*":)', r'\"', json_str)
    
    # 修复缺少逗号的情况
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    
    # 修复多余逗号
    json_str = re.sub(r',\s*}', '}', json_str)
    
    try:
        # 尝试解析修复后的JSON
        parsed = json.loads(json_str)
        return parsed
    except:
        # 如果仍然失败，返回None
        return None
