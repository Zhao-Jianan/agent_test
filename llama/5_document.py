from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

embeddings = OllamaEmbeddings(
    model="llama3.2:latest",
    base_url="http://localhost:11434"
)


# 准备测试数据 ，假设我们提供的文档数据如下：
documents = [
    Document(
        page_content="Dogs make great companions, known for their loyalty and friendliness.",
        metadata={"source": "Mammal Pet Documentation"},
    ),
    Document(
        page_content="Cats are independent pets and generally enjoy their own space.",
        metadata={"source": "Mammal Pet Documentation"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners and require relatively simple care.",
        metadata={"source": "Fish Pet Documentation"},
    ),
    Document(
        page_content="Parrots are intelligent birds that are able to imitate human speech.",
        metadata={"source": "Bird Pet Documentation"},
    ),
    Document(
        page_content="Rabbits are social animals and need plenty of space to jump.",
        metadata={"source": "Mammal Pet Documentation"},
    ),
]

vector_store = Chroma.from_documents(documents, embedding=embeddings)
# 相似度的查询: 返回相似的分数， 分数越低相似度越高
# print(vector_store.similarity_search_with_score('British Bobtail Cat'))

# 检索器: bind(k=1) 返回相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search)

# print(retriever.batch(['shark', 'dog', 'goat']))

message = """
Answer the question according the context.
This is the question: {question}.
This is the context: {context}.
"""

prompt = ChatPromptTemplate.from_messages([('human', message)])

chain = {'question': RunnablePassthrough(), 'context': retriever} | prompt | llm

response = chain.invoke('Introduce the dog')

print(response)

