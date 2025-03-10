from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

msg = [
    SystemMessage(content="Please translate the message to English"),
    HumanMessage(content= "你好吗")
    ]

parser = StrOutputParser()

chain = llm | parser

result = chain.invoke(msg)
print(result)