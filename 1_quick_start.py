from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are a expert of technology all over the world'),
    ('user', '{input}')
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({'input': '用中文介绍一下agent的科研和应用前景'})
print(result)