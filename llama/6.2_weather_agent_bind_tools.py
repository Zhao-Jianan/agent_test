import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import chat_agent_executor
from langchain_ollama import ChatOllama


os.environ["TAVILY_API_KEY"] = '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'

# 聊天机器人案例
# 创建模型
model = ChatOllama(model="llama3.2:latest", base_url="http://localhost:11434",temperature=0, verbose=True)


search = TavilySearchResults(max_results=2)

# 让模型绑定工具
tools = [search]

#  创建代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)

quentions = [
    "What is the capital city of New Zealand?",
    "What is the weather today in Auckland?"
]

for q in quentions:
    print(f"\nQuestion: {q}")
    result = agent_executor.invoke({'messages': [HumanMessage(content=q)]})
    print("Answer:", result['messages'][2].content)