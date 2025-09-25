import os
import sys
import json
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 修改导入部分，使用通义千问
from langchain_community.llms import Tongyi
from ragas import Dataset, experiment
from ragas.llms import LangchainLLMWrapper  # 使用 Ragas 的 LangchainLLMWrapper
from ragas.metrics import DiscreteMetric

# 使用绝对导入替换相对导入
from rag import default_rag_client

# 初始化通义千问客户端
tongyi_client = Tongyi(dashscope_api_key='sk-ceab0c4dbd814619bfa3cad7d9e9f32a', model="qwen-turbo")
rag_client = default_rag_client(llm_client=tongyi_client)

# 使用 Ragas 的 LangchainLLMWrapper 包装 Tongyi 模型
llm = LangchainLLMWrapper(tongyi_client)


def load_dataset():
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )

    data_samples = [
        {
            "question": "What is ragas 0.3",
            "grading_notes": "- experimentation as the central pillar - provides abstraction for datasets, experiments and metrics - supports evals for RAG, LLM workflows and Agents",
        },
        {
            "question": "how are experiment results stored in ragas 0.3?",
            "grading_notes": "- configured using different backends like local, gdrive, etc - stored under experiments/ folder in the backend storage",
        },
        {
            "question": "What metrics are supported in ragas 0.3?",
            "grading_notes": "- provides abstraction for discrete, numerical and ranking metrics",
        },
    ]

    for sample in data_samples:
        row = {"question": sample["question"], "grading_notes": sample["grading_notes"]}
        dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)


@experiment()
async def run_experiment(row):
    response = rag_client.query(row["question"])

    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"],
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view


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
    context_parts.append(
        "这是Sakila数据库，一个影片租赁数据库，包含以下表：actor, address, category, city, country, customer, film, film_actor, film_category, inventory, language, payment, rental, staff, store")

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

async def main():
    dataset = load_dataset()
    dataset = data
    print("dataset loaded successfully", dataset)
    experiment_results = await run_experiment.arun(dataset)
    print("Experiment completed successfully!")
    print("Experiment results:", experiment_results)
    
    # 更直观地显示结果
    print("\n=== 详细评估结果 ===")
    for i, result in enumerate(experiment_results):
        print(f"\n问题 {i+1}: {result['question']}")
        print(f"回答: {result['response']}")
        print(f"评分: {result['score']}")
        print(f"评分标准: {result['grading_notes']}")
        print(f"日志文件: {result['log_file']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())