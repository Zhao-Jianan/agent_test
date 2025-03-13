from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory

# Initialize LLM and search tool
model = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

embeddings = OllamaEmbeddings(
    model="llama3.2:latest",
    base_url="http://localhost:11434"
)

loader = WebBaseLoader(
    web_path = ["https://arxiv.org/html/2401.02020v1"],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('ltx_page_main', 'ltx_page_content', 'ltx_document ltx_authors_1line'))
    )
)

docs = loader.load()

# print(len(docs))
# print(docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = splitter.split_documents(docs)

vector_store = Chroma.from_documents(docs, embedding=embeddings)

retriever = vector_store.as_retriever()


system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer concise.\n

{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

create_doc_chain = create_stuff_documents_chain(model, prompt)


contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

retriever_history_temp = ChatPromptTemplate.from_messages(
    [
        ('system', contextualize_q_system_prompt),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)

history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)

history_records = {}



def get_session_history(session_id: str):
    if session_id not in history_records:
        history_records[session_id] = ChatMessageHistory()
    return history_records[session_id]


chain = create_retrieval_chain(history_chain, create_doc_chain)

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)


# round 1
resp1 = result_chain.invoke(
    {'input': 'Please briefly introduce Spiking Self-Attention (SSA) in 100 words?'},
    config={'configurable': {'session_id': 'spikingformer_01'}}
)

print(resp1['answer'])
print("-------------------------------------")

# round 2
resp2 = result_chain.invoke(
    {'input': 'What is the difference between it and Spiking Convolutional Stem (SCS)?'},
    config={'configurable': {'session_id': 'spikingformer_01'}}
)

print(resp2['answer'])