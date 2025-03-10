from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
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

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()


agent_executor = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=20,
    agent_kwargs={
        "prefix": """You are an intelligent agent designed to interact with SQL databases.
Your task is to answer questions about the database by generating and executing SQL queries.
Note: In the database, you can see the data table. You need to enter the corresponding data table to query the specific data records to complete the task.
When given a question, you should:
1. First check and remember what tables are in the database.
2. In the table in the database you remember, select the appropriate data table to query
3. Create a syntactically correct SQL statement based on the input question.
4. Execute the SQL statement to retrieve the results.
5. Return the results to the user, ensuring that you limit SQL queries to a maximum of 10 results unless specified otherwise.
6. Always check the schema of the relevant table before querying it to ensure accuracy.
7. If you encounter any errors while executing the query, rewrite the SQL query and try again.
Do not perform any DML statements (insert, update, delete, etc.) on the database.

Make sure to provide clear and accurate answers based on the data in the database.
"""
    }
)


# Test questions
quentions = [
    "How many records are in the storage table?",
    "In the product data table, which of the first 10 products is the most expensive??",
    "List all the users' emails."
]

for q in quentions:
    print(f"\nQuestion: {q}")
    result = agent_executor.invoke({"input": q})
    print("Answer:", result["output"])
    print("---------------------------------------------------")



