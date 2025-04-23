任务介绍
古诗词理解和推理评测任务是一个few-shot的任务，包含了古诗、唐诗和宋词，拥有五言绝句、七言绝句、五言律诗和七言律诗等形式的古诗词。评测提供了200条数据用于训练，400条数据用来验证、测试。所有数据均以 JSON 格式提供。每条数据包括以下字段：

title：古诗词题目
author：古诗词作者
content：古诗词内容
keywords：古诗词的关键词及其释义
trans：古诗词的白话文译文
emotion：古诗词的情感表达
qa_words：需要回答的关键词
qa_sents：需要回答的句子
choose：情感选项
ans_qa_words：回答关键词的结果
ans_qa_sents：回答句子的结果
choose_id：选择的情感选项的下标
示例数据格式
训练数据集：
{
    "title": "泊秦淮",
    "author": "杜牧",
    "content": "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "keywords": {
        "泊": "停泊",
        "商女": "歌女",
        "后庭花": "歌曲《玉树后庭花》的简称"
    },
    "trans": "迷离的月色下，轻烟笼罩寒水、白沙，夜晚船只停泊在秦淮边靠近岸上的酒家。卖唱的歌女好似不懂什么叫亡国之恨，隔着江水仍然高唱着《玉树后庭花》。",
    "emotion": "爱国"
}
测试数据集：
{
    "title": "泊秦淮",
    "author": "杜牧",
    "content": "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "qa_words": ["泊", "商女", "后庭花"],
    "qa_sents": ["烟笼寒水月笼沙", "夜泊秦淮近酒家"],
    "choose": ["A":"爱国", "B":"庆祝", "C":"闲适", "D":"赞美"],
}
提交结果格式
提交的结果应为 submit.json 文件，格式如下：

[
    {
        "idx": 0,
        "ans_qa_words": {
            "衰草": "",
            "故关": "",
            "风尘": ""
        },
        "ans_qa_sents": {
            "故关衰草遍，离别自堪悲": "",
            "掩泪空相向，风尘何处期。": ""
        },
        "choose_id": ""
    },
    {
        "idx": 1,
        "ans_qa_words": {
            "窥": "",
            "牧马": "",
            "临洮": ""
        },
        "ans_qa_sents": {
            "至今窥牧马": "",
            "不敢过临洮": ""
        },
        "choose_id": ""
    }
]
每日提交次数为3次，评测失败不扣除次数。

评价指标
对于古诗词理解任务，评测任务采用BLEU值、 中文BertScore分数作为评估指标。对于古诗词推理任务，根据选择题准确率计算得分。

系统排名
所有评测任务均采用百分制分数显示，小数点后保留3位。

系统排名取各项任务得分的加权和（0.5，0.5）

task_score = 0.5taskA + 0.5 taskB

评测数据
数据由json格式给出，数据集包含以下内容：

train-data.zip 训练+验证数据

eval-data.zip 赛事评估数据

Baseline
Qwen2.5-7b zeroshot

任务评估结果
总分数 taskA emo-acc taskB bleu_words bleu_sents sim_words sim_sents
0.667 0.771 0.771 0.564 0.230 0.241 0.873 0.911
数据集信息
数据集提供方：

数据集协议
该数据集遵循协议：CC BY-NC 4.0协议

由于版权保护问题，CFN数据集只免费提供给用户用于非盈利性科学研究使用，参赛人员不得将数据用于任何商业用途。如果用于商业产品，

请联系：<23S151077@stu.hit.edu.cn>

参考文献
如果你对我们的工作感兴趣，欢迎查看我们的工作

@article{chen2024benchmarking,
  title={Benchmarking llms for translating classical chinese poetry: Evaluating adequacy, fluency, and elegance},
  author={Chen, Andong and Lou, Lianzhang and Chen, Kehai and Bai, Xuefeng and Xiang, Yang and Yang, Muyun and Zhao, Tiejun and Zhang, Min},
  journal={arXiv preprint arXiv:2408.09945},
  year={2024}
}
