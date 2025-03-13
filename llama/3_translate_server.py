from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")


prompt = ChatPromptTemplate.from_messages([
    ('system','Please translate the message to {language}'),
    ('user', '{text}')
])

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({'language': 'Maori', 'text': 'University of auckland'})
print(result)

app = FastAPI(title='My LangChain Service', version='V1.0', description='LangChain Translate')

add_routes(
    app,
    chain,
    path="/chainDemo",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)