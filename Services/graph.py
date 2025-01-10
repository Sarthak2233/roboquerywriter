import os
import re
import json
from typing import TypedDict, Annotated, Sequence
import operator
import mysql.connector
from mysql.connector import Error
from utils import split_and_store

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, HarmCategory
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import matplotlib.pyplot as plt
from PIL import Image
from schema import Configuration_Database, Understandind_question
import io

load_dotenv()


def get_resized_image(app, width, height):
    """
    Gets the graph from the app, draws it as a PNG,
    and resizes the resulting image to the specified width and height.

    Args:
        app: The application object.
        width: The desired width of the resized image.
        height: The desired height of the resized image.

    Returns:
        A resized PIL Image object.
    """

    png_bytes = app.get_graph().draw_png()
    img = Image.open(io.BytesIO(png_bytes))
    resized_img = img.resize((width, height))
    return resized_img


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    config: Configuration_Database
    path: str


parser_0 = PydanticOutputParser(pydantic_object=Understandind_question)

llm = ChatGoogleGenerativeAI(
    model='models/gemini-1.5-pro',
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.environ['GEMMMINI_API_KEY'],  # AIzaSyCdmxCdnZbRtV90BCaBcUa2u3PqW8HkRwA
    safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                         HarmBlockThreshold.BLOCK_NONE
                     },
)


@tool
def read_files_for_context(path) -> str:
    """Describes the an image given its URL.

      Args:
        path: The path to a folder containing files .

      Returns:
        A string with the metadata and schema.
      """
    print('------Reading files for context------')

    # Read the metadata and schema.er files from the given path
    path = os.path.normpath(path.strip())
    meta_datapath = os.path.join(path, 'metadata.json')
    er_diagram_path = os.path.join(path, 'schema.er')

    with open(meta_datapath, 'r') as f:
        metadata = json.load(f)

    with open(er_diagram_path, 'r') as f:
        er_diagram = f.read()

    return json.dumps(metadata, indent=4) + '\n' + er_diagram + '\n'


def make_er(state):
    print('----------------Generating ER diagram--------------------')
    import subprocess
    path = state['path']
    config = state['config']
    user = config['user']
    password = config['password']
    host = config['host']
    port = config['port']
    database = config['database']
    #f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

    # Generate the output file path
    output_file = os.path.join(path, "schema.er")

    # Command to run eralchemy
    command = [
        "eralchemy",
        "-i",
        connection_string,
        "-o",
        output_file
    ]

    # Run the command
    subprocess.run(command, check=True)

    return None

import psycopg2
import json
# def make_connection_get(state):
#     print('-----------Getting database connection and writing metadata to file-----------')
#     connection = None
#     cursor = None
#     try:
#         # Extract configuration and file path from the state
#         config = state['config']
#         output_path = state['path'] + '/metadata.json'
#
#         # Connect to the PostgreSQL database
#         connection = psycopg2.connect(
#             dbname=config['database'],
#             user=config['user'],
#             password=config['password'],
#             host=config['host'],
#             port=config.get('port', 5432)  # Default port for PostgreSQL
#         )
#         cursor = connection.cursor()
#
#         # Query to get all tables in the public schema
#         cursor.execute("""
#             SELECT table_name
#             FROM information_schema.tables
#             WHERE table_schema = 'public'
#         """)
#         tables = cursor.fetchall()
#
#         metadata = {}
#
#         for (table,) in tables:
#             # Query to describe the table structure (columns)
#             cursor.execute(f"""
#                 SELECT column_name, data_type, is_nullable, column_default
#                 FROM information_schema.columns
#                 WHERE table_schema = 'public' AND table_name = '{table}'
#             """)
#             columns = cursor.fetchall()
#
#             # Query to get relationships (foreign keys)
#             cursor.execute(f"""
#                 SELECT
#                     tc.table_name,
#                     kcu.column_name,
#                     ccu.table_name AS referenced_table,
#                     ccu.column_name AS referenced_column
#                 FROM
#                     information_schema.table_constraints AS tc
#                 JOIN information_schema.key_column_usage AS kcu
#                     ON tc.constraint_name = kcu.constraint_name
#                     AND tc.table_schema = kcu.table_schema
#                 JOIN information_schema.constraint_column_usage AS ccu
#                     ON ccu.constraint_name = tc.constraint_name
#                     AND ccu.table_schema = tc.table_schema
#                 WHERE tc.constraint_type = 'FOREIGN KEY'
#                   AND tc.table_schema = 'public'
#                   AND tc.table_name = '{table}'
#             """)
#             relationships = cursor.fetchall()
#
#             metadata[table] = {
#                 "columns": [
#                     {
#                         "column_name": column[0],
#                         "data_type": column[1],
#                         "is_nullable": column[2],
#                         "default_value": column[3]
#                     }
#                     for column in columns
#                 ],
#                 "relationships": [
#                     {
#                         "table": row[0],
#                         "column": row[1],
#                         "referenced_table": row[2],
#                         "referenced_column": row[3],
#                     }
#                     for row in relationships
#                 ],
#             }
#
#         # Write metadata to the output file
#         with open(output_path, 'w') as f:
#             json.dump(metadata, f, indent=4)
#
#         print(f"Metadata written to {output_path}")
#
#     except psycopg2.Error as err:
#         return f"Database error: {err}"
#
#     except Exception as ex:
#         return f"Error: {ex}"
#
#     finally:
#         if cursor:
#             cursor.close()
#         if connection:
#             connection.close()
#
#     return None
def make_connection_get(state):
    print('-----------Getting database connection ans writing metadata to file-----------')
    global connection, cursor
    try:
        # Extract configuration and file path from the state
        config = state['config']
        output_path = state['path'] + '/metadata.json'
        # Connect to the MySQL database
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()

        # Query to get all tables in the database
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        metadata = {}

        for (table,) in tables:
            # Query to describe the table structure
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()

            # Query to get relationships (foreign keys)
            cursor.execute(f"""
                SELECT
                    TABLE_NAME, COLUMN_NAME,
                    REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM
                    INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE
                    TABLE_SCHEMA = '{config['database']}' AND
                    TABLE_NAME = '{table}' AND
                    REFERENCED_TABLE_NAME IS NOT NULL
            """)
            relationships = cursor.fetchall()

            metadata[table] = {
                "columns": [dict(zip(["Field", "Type", "Null", "Key", "Default", "Extra"], column)) for column in
                            columns],
                "relationships": [
                    {
                        "table": row[0],
                        "column": row[1],
                        "referenced_table": row[2],
                        "referenced_column": row[3],
                    }
                    for row in relationships
                ],
            }

        # Write metadata to the output file
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata written to {output_path}")

    except mysql.connector.Error as err:
        return f"Database error: {err}"

    except Exception as ex:
        return f"Error: {ex}"

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

    return None
def get_database_context(state):
    print("------Getting database context------")

    query = "Have you understood about the given database?"

    print('From the first message, the user is asking about the database.', state['messages'][-1])

    # Make feed the context to the model about the databse by reading the
    # metadata to get the relationships and columns of the tables
    metadata_path = state['path'] + '/metadata.json'
    er_diagram_path = state['path'] + '/schema.er'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    with open(er_diagram_path, 'r') as f:
        er_diagram = f.read()

    print('The metadata of the database is:', metadata)
    print('The ER diagram of the database is:', er_diagram)

    # Generate the prompt
    prompt = '''
        You are an AI assistant. You are responsible to understand the schemas and the meta-datas and schema.er 
        of the database provides to you. Your response should be in either Yes or NO.
        
        The provided metadata is {metadata}.
        The provided ER diagram is {er_diagram}
        
        Answer your query: {query}
        Your response must be a strict boolean value (Yes or No)
        
        {format_instructions}
    
    '''
    prompt_template = PromptTemplate(
        input_variables=['metadata', 'query', 'er_diagram'],
        template=prompt,
        partial_variables={'format_instructions': parser_0.get_format_instructions()},
    )

    chain = prompt_template | llm | parser_0

    response = chain.invoke(
        {
            'metadata': metadata,
            'query': query,
            'er_diagram': er_diagram,
        },
        verbose=True
    )

    print(f"The response from the model is: {response}")

    return {'messages': [response.answer]}


def router_1(state):
    message = state['messages'][-1]
    if message == "Yes":
        return {
            'route': ['Yes']
        }
    else:
        return {
            'route': ['No']
        }


# Make a React agent that takes the user query and generates the Sql query and runs
# from the given configuration dictionary.

def sql_writing_agent(state):
    print(len(state['messages']))
    user_query = state['messages'][0]

    print('-----------Generating SQL query from user query-----------')
    print(f'Your query is: {user_query}')
    if user_query.lower() == 'exit':
        print('Exiting.....')
        exit("BYE!!! Have a nice day.")

    path = state['path']
    tools = [read_files_for_context]

    prompt = PromptTemplate(
        input_variables=['agent_scratchpad', 'input', 'path', 'tools', 'tool_names'],
        template="""Answer the following questions as best as you can. You have access to the following tools:\n
        {tools}\n
        Use the following format:\n
        Question: The input question you must answer about the database.
        Thought: Analyze the question, determine the steps to take, and Use the provided path {path} is necessary to write the SQL query stydying all the table field names from er schema using the tools.
        Action: The action to take, which should be one of [{tool_names}].
        Action Input: The exact input to the action.
        Observation: The result of the action.
        ... (this Thought/Action/Action Input/Observation can repeat N times as necessary)
        Thought: I now know the final answer.
        Final Answer: The final answer to the original input question must be a single SQL query that directly addresses the question.

        Important: 
        - Your response must follow the format strictly.
        - The Final Answer must be only the MySQL query, without any additional text, characters, or commentary.
        - Do not include retry messages or error explanations in your response.

        Example:
        Question: How many users are in the database?
        Thought: I need to count the total number of users in the 'users' table. I will write a MySQL query to achieve this studying the metadata and schema.
        Action: Write SQL query
        Action Input: SELECT COUNT(*) FROM users;
        Observation: The query is correct.
        Thought: I now know the final answer.
        Final Answer: SELECT COUNT(*) FROM users;

        Begin!

        Question: {input}

        Thought:{agent_scratchpad}
        """
    )

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        parsers=[StrOutputParser()],
        verbose=True,
        handle_parsing_errors=True,
    )

    response = agent_executor.invoke(
        {'input': user_query,
         'path': path,
         },
        verbose=True
    )

    answer = response['output']

    return {'messages': [answer]}


# def execute_sql_query(state):
#     global connection
#     global cursor
#     to_run = input('Do you want SQL query to execute: y/n')
#     if to_run.lower() == 'y':
#         print('-----------Executing SQL query-------------')
#         config = state['config']
#         query = state['messages'][-1]
#
#         print(f'The SQL query to execute is: {query}')
#         fetch_results = True
#
#         # Connect to the database
#         try:
#             # Connect to the database
#             connection = mysql.connector.connect(**config)
#             cursor = connection.cursor()
#
#             # Execute the query
#             cursor.execute(query)
#
#             print("Query executed successfully.")
#             # Commit changes for queries that modify data
#             if connection.is_connected() and not fetch_results:
#                 connection.commit()
#                 print('Changes committed successfully.')
#
#             # Fetch results for SELECT queries
#             if fetch_results:
#                 results = cursor.fetchall()
#                 print("Result fetched executed successfully.")
#                 return {'message': [results]}  # Change in the state
#
#         except Error as e:
#             print(f"Error: {e}")
#         finally:
#             if connection.is_connected():
#                 cursor.close()
#                 connection.close()
#                 print("Database connection closed.")
#     else:
#         exit("BYE!!! Have a nice day.")
#
#------------Postgresql- config__________________
import psycopg2
from psycopg2 import sql, OperationalError, Error
# def execute_sql_query(state):
#     to_run = input('Do you want to execute the SQL query? (y/n): ').strip().lower()
#     if to_run not in {'y', 'n'}:
#         print("Invalid input. Please enter 'y' or 'n'.")
#         return 'Invalid input. BYE!!! Have a nice day.'
#     if to_run == 'n':
#         exit("BYE!!! Have a nice day.")
#
#     print('-----------Executing SQL query-------------')
#     config = state['config']
#     query = state['messages'][-1].strip()
#     print(f"The query to execute: {query}")
#
#     fetch_results = query.lower().startswith("select")  # Determine query type
#
#     try:
#         # Connect to PostgreSQL database
#         with psycopg2.connect(
#                 dbname=config['database'],
#                 user=config['user'],
#                 password=config['password'],
#                 host=config['host'],
#                 port=config.get('port', 5432)  # Default port for PostgreSQL
#         ) as connection:
#             with connection.cursor() as cursor:
#                 # Execute the query
#                 cursor.execute(query)
#
#                 if fetch_results:
#                     # Fetch results for SELECT queries
#                     results = cursor.fetchall()
#                     print("Query executed successfully. Results fetched:")
#                     print(results)
#                     return {'messages': results}
#                 else:
#                     # Commit changes for INSERT/UPDATE/DELETE queries
#                     connection.commit()
#                     print("Query executed successfully. Changes committed.")
#     except OperationalError as oe:
#         print(f"Database connection error: {oe}")
#     except Error as e:
#         print(f"Database error: {e}")
#     except Exception as ex:
#         print(f"An unexpected error occurred: {ex}")


#----------------------For my sql query execution------------------
def execute_sql_query(state):
    to_run = input('Do you want to execute the SQL query? (y/n): ').strip().lower()
    if to_run not in {'y', 'n'}:
        print("Invalid input. Please enter 'y' or 'n'.")
        return 'Invalid input.' + exit("BYE!!! Have a nice day.")
    if to_run == 'n':
        exit("BYE!!! Have a nice day.")

    print('-----------Executing SQL query-------------')
    config = state['config']
    query = state['messages'][-1].strip()
    print(query)

    # List of keywords that indicate the query should fetch results
    fetch_keywords = ["select", "show", "describe", "explain"]
    fetch_results = any(query.lower().startswith(keyword) for keyword in fetch_keywords)  # Determine query type

    try:
        with mysql.connector.connect(**config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                if fetch_results:
                    results = cursor.fetchall()
                    print(results)
                    print("Query executed successfully. Results fetched.")
                    return {'messages': [results]}
                else:
                    connection.commit()
                    print("Query executed successfully. Changes committed.")
    except mysql.connector.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def show_query_results(state):
    print('-----------Showing query results------------')
    message = state['messages'][-1]
    print(type(message))
    print(message)

    prompt = '''
        You are an AI assistant. You will analyze the prompt {initial_prompt} and the result of the query {query_result}.
        
        Answer the following questions as best as you can.
        
        Question: What is the result of the query?
        Thought: The result of the query is {query_result}.
        
        Question: What is the initial prompt?
        Thought: The initial prompt is {initial_prompt}.
        
        Question: Describe the result of the query analyzing the user query?
        
        Your answer must be  descriptive and short.
        
        
    '''

    prompt_template = PromptTemplate(
        input_variables=['initial_prompt', 'query_result'],
        template=prompt,
    )

    chain = prompt_template | llm | StrOutputParser()

    response = chain.invoke(
        {
            'initial_prompt': state['messages'][0],
            'query_result': message,
        },
        verbose=True
    )
    print(f"{response}")
    return {'messages': [message]}


def router_2(state):
    ask_question = input('Do you want to talk about the results or query? Type query or result')
    if ask_question.lower() == 'query':
        return {
            'route': ['ask_about_query']
        }
    if ask_question.lower() == 'result':
        return {
            'route': ['ask_about_results']
        }
    else:
        return {
            'route': ['NO']
        }


def talk_about_results(state):
    print('-----------Talking about results--------------')
    ask_about_results = input('Ask questions about the results?')
    result = state['messages'][-1]

    prompt = '''
        You are an AI assistant. You are responsible to talk about the results of the question 
        asked by the user. 
        
        Question: {ask_about_results}
        
        Thought: The results of the query are {result}.
        
        YOUR RESPONSE MUST BE CONCISE AND CLEAR.
    
    '''
    prompt_template = PromptTemplate(
        input_variables=['ask_about_results', 'result'],
        template=prompt,
    )

    chain = prompt_template | llm | StrOutputParser()

    response = chain.invoke(
        {
            'ask_about_results': ask_about_results,
            'result': result,
        },
        verbose=True
    )
    print(f"The response from the model is: {response}")

    return {'messages': [response]}


def talk_about_query(state):
    print('-----------Talking about query--------------')
    ask_about_query = input('Ask questions about the query?')
    result = state['messages'][-3]

    prompt = '''
            You are an AI assistant. 
            You are responsible to talk about the query {result} of the question 
            asked by the user. 

            Question: {ask_about_query}

            YOUR RESPONSE MUST BE CONCISE AND CLEAR.
        '''

    prompt_template = PromptTemplate(
        input_variables=['ask_about_query', 'result'],
        template=prompt,
    )

    chain = prompt_template | llm | StrOutputParser()

    response = chain.invoke(
        {
            'ask_about_query': ask_about_query,
            'result': result,
        },
        verbose=True
    )
    print(f"The response from the model is: {response}")

    return {'messages': [response]}


# Desigining the state graph

workflow = StateGraph(AgentState)

workflow.add_node('make_er', make_er)
workflow.add_node('make_connection_get', make_connection_get)
workflow.add_node('get_database_context', get_database_context)
workflow.add_node('router_1', router_1)
workflow.add_node('sql_writing_agent', sql_writing_agent)
workflow.add_node('execute_sql_query', execute_sql_query)
workflow.add_node('show_query_results', show_query_results)
workflow.add_node('router_2', router_2)
workflow.add_node('talk_about_results', talk_about_results)
workflow.add_node('talk_about_query', talk_about_query)

# Add the node entry point

workflow.set_entry_point('make_er')

# Add the edges

workflow.add_edge('make_er', 'make_connection_get')
workflow.add_edge('make_connection_get', 'get_database_context')
workflow.add_edge('get_database_context', 'router_1')

# Add the edges for the router_1
workflow.add_conditional_edges(
    source="router_1",  # After node_2, determine next step
    path=lambda state: state['route'],  # Function to determine routing
    path_map={
        'Yes': 'sql_writing_agent',
        'No': 'get_database_context'
    }
)

workflow.add_edge('sql_writing_agent', 'execute_sql_query')
workflow.add_edge('execute_sql_query', 'show_query_results')
workflow.add_edge('show_query_results', 'router_2')

# Add the edges for the router_2
workflow.add_conditional_edges(
    source="router_2",  # After node_2, determine next step
    path=lambda state: state['route'],
    path_map={
        'ask_about_query': 'talk_about_query',
        'ask_about_results': 'talk_about_results',
        'NO': END
    }
)

workflow.add_edge('talk_about_query', END)
workflow.add_edge('talk_about_results', END)

# Compiling the state graph

graph = workflow.compile()

# Visualizing the state graph
# Example usage with matplotlib
# resized_image = get_resized_image(graph, 800, 800)
# plt.imshow(resized_image)
# plt.axis('off')
# plt.show()
#DATABASE_URL=postgresql://nepalnowdevuser:9pydzz6OeLa7lfmS12@94.136.187.86:5433/nepalnow-dev
#"mysql+mysqlconnector://root:admin12345@localhost:3306/mywebapp"
# inputs = {
#     'messages': [input('Enter the question that you want to know from the database?')],
#     'config': {
#         'user': 'nepalnowdevuser',
#         'password': '9pydzz6OeLa7lfmS12',
#         'host': '94.136.187.86',
#         'port': '5433',
#         'database': 'nepalnow-dev'
#     },
#     'path': r'E:/Codes/roboql/Contexts',
# }

inputs = {
    'messages': [input('Enter the question that you want to know from the database?')],
    'config': {
        'user': 'root',
        'password': 'admin12345',
        'host': 'localhost',
        'port': '3306',
        'database':'mywebapp'
    },
    'path': r'E:/Codes/roboql/Contexts',
}

# Running the   state graph
for output in graph.stream(inputs):
    for node, value in output.items():
        if node != "router_1" and node != "router_2":
            print(f'Output from node {node}')
            print("--" * 8)
            print(type(value))
            print(value)
