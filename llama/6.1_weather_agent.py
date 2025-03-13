from langchain_community.tools import TavilySearchResults
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, initialize_agent, AgentType
import os

os.environ['TAVILY_API_KEY'] = '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'

# Initialize LLM and search tool
llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")
search = TavilySearchResults(max_results=1)

# Create agent with custom prompt
agent_executor = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    agent_kwargs={
        "prefix": """You are an AI that follows instructions precisely.

CRITICAL INSTRUCTION:
For questions about basic facts (capitals, history, geography), you MUST answer IMMEDIATELY with "Final Answer:" followed by your knowledge. DO NOT use any tools or take any actions.

The ONLY time you should use a tool is for current information like weather or news.
To use the tool, format exactly like this:
Action: tavily_search_results_json
Action Input: your search query

If the question is about basic facts, respond with your knowledge directly without using the tool.

Example responses:
1. "What is the capital of France?"
Final Answer: Paris

2. "What is the weather in Paris today?"
Action: tavily_search_results_json
Action Input: current weather in Paris today

Question: {input}
"""
    }
)

# Test questions
quentions = [
    "What is the capital city of New Zealand?",
    "What is the weather today in Auckland?"
]

for q in quentions:
    print(f"\nQuestion: {q}")
    result = agent_executor.invoke({"input": q})
    print("Answer:", result["output"])


