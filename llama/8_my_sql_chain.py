from operator import itemgetter

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

# Initialize LLM and search tool
model = OllamaLLM(model="llama3.2:latest", base_url="http://localhost:11434")

HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'csc8019group8'
USERNAME = 'root'
PASSWORD = '123456'

MYSQL_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)
# print(db.get_usable_table_names())
# print(db.run('select count(*) from storage;'))

# Define a more direct prompt for generating SQL with examples
sql_query_prompt = PromptTemplate.from_template(
    """You are an SQL query generator. Given an input question, return only the SQL query without any additional text.
        Here are some examples:
        - Question: How many records are in the storage table?
          SELECT COUNT(*) FROM storage;
        - Question: What are the names of all users?
          SELECT name FROM users;
        - Question: What is the total amount in the sales table?
          SELECT SUM(amount) FROM sales;
        Only use the following tables: {table_info}
        top_k: {top_k}
        Question: {input}
        SQLQuery: 
    """
)

# Create the require_chain with the specific answer prompt
require_chain = create_sql_query_chain(model, db, prompt=sql_query_prompt)

# Define the general answer prompt for the final output
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, SQL statement, and the result after SQL execution, answer the user question.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
)

sql_executor_tool = QuerySQLDataBaseTool(db=db)

# Create the SQL run chain
sql_run_chain = RunnablePassthrough.assign(query=require_chain).assign(result=itemgetter('query') | sql_executor_tool)

# Combine the chains
chain = sql_run_chain | answer_prompt | model | StrOutputParser()

# Invoke the chain and print the response
response = chain.invoke(input={'question': 'How many records are in the storage table'})
print(response)



