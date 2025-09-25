import os
from dotenv import load_dotenv
load_dotenv() # 加载.env文件中的环境变量
import numpy as np
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
import json
import yaml

# 准备评估用的LLM（使用阿里通义千问）
# 使用Ragas的LangchainLLMWrapper包装器来包装LangChain的Tongyi模型
llm = LangchainLLMWrapper(Tongyi(model="qwen-turbo"))

# 准备数据集
# 从data/sakila/q2sql_pairs.json加载数据集
with open('data/sakila/q2sql_pairs.json', 'r', encoding='utf-8') as f:
    q2sql_data = json.load(f)

# 从db_description.yaml加载数据库表描述
with open('data/sakila/db_description.yaml', 'r', encoding='utf-8') as f:
    db_descriptions = yaml.safe_load(f)

# 从ddl_statements.yaml加载数据库表结构定义
with open('data/sakila/ddl_statements.yaml', 'r', encoding='utf-8') as f:
    ddl_statements = yaml.safe_load(f)

# 根据问题内容确定相关的表
def get_relevant_tables(question):
    # 简单的关键词匹配来确定相关表
    tables = []
    question_lower = question.lower()
    
    # 检查问题中提到的表名
    for table in db_descriptions.keys():
        if table in question_lower:
            tables.append(table)
    
    # 根据关键词映射到相关表
    if not tables:
        if any(word in question_lower for word in ['actor', '演员']):
            tables.extend(['actor'])
        if any(word in question_lower for word in ['film', 'movie', '影片']):
            tables.extend(['film'])
        if any(word in question_lower for word in ['customer', '客户']):
            tables.extend(['customer'])
        if any(word in question_lower for word in ['category', '类型', '种类']):
            tables.extend(['category'])
        if any(word in question_lower for word in ['address', '地址']):
            tables.extend(['address'])
        if any(word in question_lower for word in ['city', '城市']):
            tables.extend(['city'])
        if any(word in question_lower for word in ['country', '国家']):
            tables.extend(['country'])
        if any(word in question_lower for word in ['inventory', '库存']):
            tables.extend(['inventory'])
        if any(word in question_lower for word in ['payment', '支付']):
            tables.extend(['payment'])
        if any(word in question_lower for word in ['rental', '租赁']):
            tables.extend(['rental'])
        if any(word in question_lower for word in ['staff', '员工']):
            tables.extend(['staff'])
        if any(word in question_lower for word in ['store', '商店']):
            tables.extend(['store'])
    
    # 如果没有匹配到任何表，则提供所有表的信息
    if not tables:
        tables = list(db_descriptions.keys())
    
    return list(set(tables))  # 去重

# 为每个问题构建上下文
def build_context_for_question(question):
    relevant_tables = get_relevant_tables(question)
    
    context_parts = []
    
    # 添加数据库总体描述
    context_parts.append("这是Sakila数据库，一个影片租赁数据库")
    
    # 为每个相关表添加描述和DDL语句
    for table in relevant_tables:
        if table in db_descriptions:
            # 添加表描述
            table_desc = f"表 {table} 的描述：\n"
            for field, desc in db_descriptions[table].items():
                table_desc += f"  {field}: {desc}\n"
            context_parts.append(table_desc)
            
        if table in ddl_statements:
            # 添加表结构定义
            table_ddl = f"表 {table} 的结构定义：\n{ddl_statements[table]}"
            context_parts.append(table_ddl)
    
    return context_parts

# 构造评估数据
questions = [item["question"] for item in q2sql_data]
answers = [item["sql"] for item in q2sql_data]

# 为每个问题构建上下文
contexts = [build_context_for_question(question) for question in questions]

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts
}
print(data)
# 将字典转换为Hugging Face的Dataset对象，方便Ragas处理
dataset = Dataset.from_dict(data)

print("\n=== Ragas评估指标说明 ===")
print("\n1. Faithfulness（忠实度）")
print("- 评估生成的答案是否忠实于上下文内容")
print("- 通过将答案分解为简单陈述，然后验证每个陈述是否可以从上下文中推断得出")
print("- 该指标仅依赖LLM，不需要embedding模型")

# 评估Faithfulness
# 创建Faithfulness评估指标，它只需要一个LLM来进行评估
faithfulness_metric = [Faithfulness(llm=llm)] # 只需要提供生成模型
print("\n正在评估忠实度...")
# 使用evaluate函数对数据集进行评估
faithfulness_result = evaluate(dataset, faithfulness_metric)
# 提取忠实度分数
scores = faithfulness_result['faithfulness']
print(scores)
# 计算平均分
mean_score = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"忠实度评分: {mean_score:.4f}")

print("\n2. AnswerRelevancy（答案相关性）")
print("- 评估生成的答案与问题的相关程度")
print("- 使用embedding模型计算语义相似度")
print("- 我们将比较开源embedding模型和阿里云embedding模型")

# 设置两种embedding模型
# 使用Ragas的LangchainEmbeddingsWrapper来包装LangChain的嵌入模型
# 1. 开源的 all-MiniLM-L6-v2 模型
# 本地模型路径（确保路径正确）
LOCAL_MODEL_PATH = "./data/all-MiniLM-L6-v2"

# 初始化嵌入模型
hf_embeddings = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
)

opensource_embedding = LangchainEmbeddingsWrapper(
    hf_embeddings
)
# 2. 阿里云的 text-embedding-v4 模型
aliyun_embedding = LangchainEmbeddingsWrapper(
    DashScopeEmbeddings(model="text-embedding-v4")
)

# 创建答案相关性评估指标
# 分别为两种embedding模型创建AnswerRelevancy评估指标
opensource_relevancy = [AnswerRelevancy(llm=llm, embeddings=opensource_embedding)]
aliyun_relevancy = [AnswerRelevancy(llm=llm, embeddings=aliyun_embedding)]

print("\n正在评估答案相关性...")
print("\n使用开源Embedding模型评估:")
# 使用开源embedding模型进行评估
opensource_result = evaluate(dataset, opensource_relevancy)
scores = opensource_result['answer_relevancy']
opensource_mean = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {opensource_mean:.4f}")

print("\n使用阿里云Embedding模型评估:")
# 使用阿里云embedding模型进行评估
aliyun_result = evaluate(dataset, aliyun_relevancy)
scores = aliyun_result['answer_relevancy']
aliyun_mean = np.mean(scores) if isinstance(scores, (list, np.ndarray)) else scores
print(f"相关性评分: {aliyun_mean:.4f}")

# 比较两种embedding模型的结果
print("\n=== Embedding模型比较 ===")
diff = aliyun_mean - opensource_mean
print(f"开源模型评分: {opensource_mean:.4f}")
print(f"阿里云模型评分: {aliyun_mean:.4f}")
print(f"差异: {diff:.4f} ({'阿里云更好' if diff > 0 else '开源模型更好' if diff < 0 else '相当'})")