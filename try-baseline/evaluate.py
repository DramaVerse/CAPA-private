import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_results(results, ground_truth):
    """简单评估模型输出与参考答案的匹配程度"""
    total_words = 0
    correct_words = 0
    total_sents = 0
    correct_sents = 0
    
    for result, gt in zip(results, ground_truth):
        # 评估词语解释
        if 'ans_qa_words' in result:
            for word, explanation in result['ans_qa_words'].items():
                total_words += 1
                if word in gt['keywords'] and explanation.strip() == gt['keywords'][word].strip():
                    correct_words += 1
        
        # 评估句子翻译
        if 'ans_qa_sents' in result:
            for sent, translation in result['ans_qa_sents'].items():
                total_sents += 1
                if translation.strip() == gt['trans'].strip():
                    correct_sents += 1
    
    word_accuracy = correct_words / total_words if total_words > 0 else 0
    sent_accuracy = correct_sents / total_sents if total_sents > 0 else 0
    
    return {
        'word_accuracy': word_accuracy,
        'sent_accuracy': sent_accuracy
    }

def main():
    results = load_json('baseline_output.json')
    ground_truth = load_json('test_data.json')[:len(results)]
    
    metrics = evaluate_results(results, ground_truth)
    
    print(f"词语解释准确率: {metrics['word_accuracy']:.4f}")
    print(f"句子翻译准确率: {metrics['sent_accuracy']:.4f}")

if __name__ == "__main__":
    main()
