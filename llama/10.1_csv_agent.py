from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
import pandas as pd
import requests
from typing import Optional

# Define prompt template
PREFIX = PromptTemplate.from_template(
    """You are a data analysis assistant working with a pandas dataframe in Python. 
    The name of the dataframe is `df`.
    Use Python pandas methods to answer the following question concisely.
    Always verify the dataframe structure with 'df.shape' before proceeding.
    Question: {input}""")


class CSVAgent:
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize CSV Agent
        Args:
            model_name: Ollama model name
        """
        self.llm = OllamaLLM(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0
        )
        self.df = None
        self.agent = None

    def load_csv(self, file_path: str, encoding: str = 'utf-8') -> None:
        """
        Load CSV file
        Args:
            file_path: Path to CSV file
            encoding: File encoding, defaults to utf-8
        """
        try:
            self.df = pd.read_csv(file_path, encoding=encoding).fillna(value=0)
            self.agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                verbose=True,
                max_iterations=20
            )
            print(f"Successfully loaded CSV file: {file_path}")
            print(f"Data shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")

    def query(self, question: str) -> Optional[str]:
        """
        Query the agent
        Args:
            question: Question about the data
        Returns:
            Agent's response
        """
        if self.agent is None:
            return "Please load a CSV file first!"

        try:
            # If asking about row count, return direct calculation
            if "how many rows" in question.lower():
                return f"The dataset has {len(self.df)} rows."

            # If asking about basic information, return direct analysis
            if "basic information" in question.lower() or "describe" in question.lower():
                num_rows, num_cols = self.df.shape
                columns = list(self.df.columns)
                basic_stats = self.df.describe()

                return f"""Dataset Overview:
- Number of rows: {num_rows}
- Number of columns: {num_cols}
- Column names: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''} 
- Numeric columns statistics:
{basic_stats.to_string()}"""

            # If asking to plot a bar chart of a specific column
            if "plot the bar chart" in question.lower():
                column_name = question.split("plot the bar chart of ")[-1].strip()
                import matplotlib.pyplot as plt

                # Normalize column names for matching
                normalized_columns = {col.lower(): col for col in self.df.columns}
                column_name_lower = column_name.lower()

                # Check if the specified column exists
                if column_name_lower in normalized_columns:
                    actual_column_name = normalized_columns[column_name_lower]
                    plt.figure(figsize=(10, 6))
                    self.df[actual_column_name].value_counts().plot(kind='bar', color='skyblue')
                    plt.title(f"Bar Chart of {actual_column_name.capitalize()}")
                    plt.xlabel(actual_column_name.capitalize())
                    plt.ylabel("Frequency")
                    plt.grid(axis='y')
                    plt.show()
                    return f"Bar chart of {actual_column_name} has been plotted."
                else:
                    return f"The '{column_name}' column is not available in the dataset."

            # Build query with explicit instructions
            prompt = f"""Analyze the pandas dataframe and provide a direct answer.
Important: The dataframe has {self.df.shape[0]} rows and {self.df.shape[1]} columns.
1. First, examine the data using appropriate pandas commands.
2. Identify the relevant columns based on the question.
3. Provide a clear and concise answer based on the analysis.
4. Do not show the code, just give the results.
5. If showing numbers or statistics, format them clearly.
6. Always verify your numbers against df.shape before responding.
Question: {question}"""

            response = self.agent.invoke({"input": prompt})
            return response["output"] if isinstance(response, dict) else response
        except Exception as e:
            if "Could not parse LLM output" in str(e):
                start = str(e).find('Could not parse LLM output: `') + len('Could not parse LLM output: `')
                end = str(e).find('`', start)
                if start > -1 and end > -1:
                    return str(e)[start:end]
            return f"Error processing query: {str(e)}"


def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            if not any(model.get('name') == 'llama3.2:latest' for model in models.get('models', [])):
                print("\nWarning: llama3.2:latest model not found!")
                return False
            return True
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to Ollama service!")
        return False
    return False


def main():
    # Check Ollama service is running
    if not check_ollama_service():
        print("Ollama service is not running")
        return

    # Create CSVAgent
    agent = CSVAgent(model_name="llama3.2:latest")

    # Load CSV
    csv_path = "titanic.csv"
    agent.load_csv(csv_path)

    # Query
    questions = [
        "How many rows are in this dataset?",
        "Can you describe the basic information about this dataset?",
        "What is the average age of passengers in this dataset?",
        "How many male and female passengers in this dataset?",
        "How many survived passengers (survived column: 1 for survived)?",
        "What are the most important features that affect survived?",
        "Can you plot the bar chart of embarked"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = agent.query(question)
        print(f"Answer: {response}")
        print("-" * 80)


if __name__ == "__main__":
    main()
