import pandas as pd
import json
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import SystemMessage, HumanMessage

# 读取 CSV 文件（动态加载）
def load_csv(file_path, input_str=None):
    global df
    df = pd.read_csv(file_path)
    return json.dumps({"message": f"CSV file '{file_path}' loaded successfully."})


load_csv('titanic.csv')

# 初始化 LLM
model = ChatOllama(model="llama3.2:latest", base_url="http://localhost:11434", temperature=0, verbose=True)

# 定义 Pandas 相关工具
def check_df():
    if 'df' not in globals() or df is None:
        return json.dumps({"error": "No dataset loaded. Please load a CSV file first."})
    return None

def describe_dataset(input_str=None):
    error = check_df()
    if error:
        return error
    return json.dumps(df.describe(include='all').to_dict())

def count_rows(input_str=None):
    error = check_df()
    if error:
        return error
    return json.dumps({"row_count": df.shape[0]})

def list_columns(input_str=None):
    error = check_df()
    if error:
        return error
    return json.dumps({"columns": df.columns.tolist()})

def count_unique_values(input_str=None):
    error = check_df()
    if error:
        return error
    return json.dumps({"unique_values": df.nunique().to_dict()})

def show_head(input_str=None):
    error = check_df()
    if error:
        return error
    return json.dumps({"head": df.head().to_dict(orient='records')})

def plot_column_distribution(column_name, input_str=None):
    import matplotlib.pyplot as plt
    error = check_df()
    if error:
        return error
    if column_name in df.columns:
        df[column_name].value_counts().plot(kind="bar")
        plt.xlabel(column_name)
        plt.ylabel("Count")
        plt.title(f"Distribution of {column_name}")
        plt.show()
        return json.dumps({"message": f"Bar chart for {column_name} plotted."})
    return json.dumps({"error": f"Column '{column_name}' not found in dataset."})

# 创建工具
tools = [
    Tool(name="Load CSV", func=load_csv, description="Load a new CSV file dynamically.", return_direct=True),
    Tool(name="Describe Dataset", func=describe_dataset, description="Get basic statistical description of the dataset.", return_direct=True),
    Tool(name="Count Rows", func=count_rows, description="Get the number of rows in the dataset.", return_direct=True),
    Tool(name="List Columns", func=list_columns, description="Get the list of column names in the dataset.", return_direct=True),
    Tool(name="Count Unique Values", func=count_unique_values, description="Get the count of unique values for each column.", return_direct=True),
    Tool(name="Show Head", func=show_head, description="Show the first few rows of the dataset.", return_direct=True),
    Tool(name="Plot Column Distribution", func=plot_column_distribution, description="Plot the distribution of a specified column.", return_direct=True)
]

# 设置系统提示词
system_prompt = """
    You are a data analysis assistant working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    Use Python pandas methods to answer the following question concisely.
    Always verify if a dataset is loaded before proceeding.
    Always call the relevant tool instead of providing a textual response.
    Do not provide instructions—always execute the appropriate function and return the result.
"""

system_message = SystemMessage(content=system_prompt)

# 创建 Agent
agent_executor = chat_agent_executor.create_tool_calling_executor(
    model=model,
    tools=tools,
    prompt=system_message
)

# 测试问题
questions = [
    "Load CSV titanic.csv",
    "What are the columns in this dataset?",
    "How many rows are in this dataset?",
    "Can you describe the basic information about this dataset?",
    "How many unique values are in each column?",
    "Can you show me the first five rows?",
    "Can you plot the distribution of a specific column, for example 'Age'?"
]

for q in questions:
    print(f"\nQuestion: {q}")
    response = agent_executor.invoke({'messages': [HumanMessage(content=q)]})
    result = response['messages']
    print("Response:", result[-1])
    print("*" * 100)
