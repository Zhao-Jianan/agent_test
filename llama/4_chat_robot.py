from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

prompt = ChatPromptTemplate.from_messages([
    ('system','You are an accommodating assistant, '
              'please do your best to use {language} to answer all the questions'),
    MessagesPlaceholder('msg')
])

chain = prompt | llm

chat_histories = {}


def get_session_chat_histories(session_id : str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_chat_histories,
    input_messages_key='msg'  # the key of messages sent each round
)

config = {'configurable': {'session_id': 'session_01'}}

#round 1
response = do_message.invoke(
    {
        'msg': [HumanMessage(content='Hello, I am Junn')],
        'language': 'English'
    },
    config=config
)

print(response)
print('-------------------------------')

#round 2
response = do_message.invoke(
    {
        'msg': [HumanMessage(content='Do you know what is my name?')],
        'language': 'English'
    },
    config=config
)

print(response)
print('-------------------------------')
config2 = {'configurable': {'session_id': 'session_02'}}
#round stream
for response in do_message.stream(
        {
            'msg': [HumanMessage(content='讲一个笑话')],
            'language': '中文'
        },
        config=config2):
    print(response, end='-')

