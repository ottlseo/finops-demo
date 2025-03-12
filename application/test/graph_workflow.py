from typing import TypedDict, List, Optional, Any
import boto3
from botocore.config import Config
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from src.opensearch import OpenSearchVectorRetriever, OpenSearchClient
from src.common_utils import SQLDatabase
import streamlit as st
import json
import copy

# class TraverseState(TypedDict, total=False):
#     parent_id: Optional[int]
#     parent_name: Optional[str]
#     child_level: int
#     selected_child_ids: List[int]
#     child_names: List[str]
#     next_action: str

class GraphState(TypedDict, total=False):
    llm_model: str
    # support_model: str
    region_name: str
    question: str
    intent: str
    sample_queries: list
    readiness: str
    tables_summaries: list
    table_names: list
    table_details: list
    query_state: dict
    next_action: str
    answer: str
    dialect: str
    # subgraph: str
    # target_node: List[int]
    # next_step: str  # next_action
    # parent_id: int
    # parent_name: str
    # content: str
    # contents_length: int
    # search_type: str
    # traverse_state: List[TraverseState]
    # searching_scheme: str
    # language: str
    # status: str
    # k: int

global_object = {
    "graph": None,
    "boto3_client": None,
    "sql_search_client": None,
    "table_search_client": None,
    "sql_retriever": None,
    "table_retriever": None,
    "engine": None,
    "db": None,
    "Session": None,
    # "graph_url": None,
    # "graph_username": None, 
    # "graph_password": None,
    "csv_list_response_format": "Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`",
    "initial_query_state": {
        "status": "success",
        "query": "",
        "result": "",
        "error": {
            "code": "",
            "message": "",
            "failed_step": "",
            "hint": ""
        }
    }
}

# boto_session = boto3.Session()
# region_name = boto_session.region_name

# llm_model = "anthropic.claude-3-5-haiku-20241022-v1:0" # TODO: change to Nova
# llm_model = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# csv_list_response_format = "Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`"
# json_response_format = """'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"setup": {"title": "Setup", "description": "question to set up a joke", "type": "string"}, "punchline": {"title": "Punchline", "description": "answer to resolve the joke", "type": "string"}}, "required": ["setup", "punchline"]}\n```'"""

# engine = create_engine("sqlite:///Chinook.db")
# db = SQLDatabase(engine)
# DIALECT = "sqlite"
# Session = sessionmaker(bind=engine)


def update_global_object(**kwargs):
    global_object.update(kwargs)


def init_boto3_client(region: str):
    retry_config = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "standard"}
    )
    return boto3.client("bedrock-runtime", region_name=region, config=retry_config)

def init_search_resources():  
    
    sql_search_client = OpenSearchClient(region_name=global_object['region_name'], index_name='example_queries', mapping_name='mappings-sql', vector="input_v", text="input", output=["input", "query"])
    table_search_client = OpenSearchClient(region_name=global_object['region_name'], index_name='schema_descriptions', mapping_name='mappings-detailed-schema', vector="table_summary_v", text="table_summary", output=["table_name", "table_summary"])

    sql_retriever = OpenSearchVectorRetriever(sql_search_client, region_name=global_object['region_name'], k=20)
    table_retriever = OpenSearchVectorRetriever(table_search_client, region_name=global_object['region_name'], k=10)
    return sql_search_client, table_search_client, sql_retriever, table_retriever

def converse_with_bedrock(model_client, sys_prompt, usr_prompt, model_id):
    temperature = 0
    top_p = 0.1
    #top_k = 1
    inference_config = {"temperature": temperature, "topP": top_p}
    #additional_model_fields = {"top_k": top_k}
    response = model_client.converse(
        modelId=model_id, 
        messages=usr_prompt, 
        system=sys_prompt,
        inferenceConfig=inference_config,
    #    additionalModelRequestFields=additional_model_fields
    )
    return response['output']['message']['content'][0]['text']

def create_prompt(sys_template, user_template, **kwargs):
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt

def get_column_description(table_search_client, table_name):
    query = {
        "query": {
            "match": {
                "table_name": table_name
            }
        }
    }
    response = table_search_client.conn.search(index=table_search_client.index_name, body=query)

    if response['hits']['total']['value'] > 0:
        source = response['hits']['hits'][0]['_source']
        columns = source.get('columns', [])
        if columns:
            return {col['col_name']: col['col_desc'] for col in columns}
        else:
            return {}
    else:
        return {}


def search_by_keywords(table_search_client, keyword):
    query = {
        "size": 10, 
        "query": {
            "nested": {
                "path": "columns",
                "query": {
                    "match": {
                        "columns.col_desc": f"{keyword}"
                    }
                },
                "inner_hits": {
                    "size": 1, 
                    "_source": ["columns.col_name", "columns.col_desc"]
                }
            }
        },
        "_source": ["table_name"]
    }
    response = table_search_client.conn.search(
        index=table_search_client.index_name,
        body=query
    )
    
    search_result = ""
    try:
        results = []
        table_names = set()  
        if 'hits' in response and 'hits' in response['hits']:
            for hit in response['hits']['hits']:
                table_name = hit['_source']['table_name']
                table_names.add(table_name)  
                for inner_hit in hit['inner_hits']['columns']['hits']['hits']:
                    column_name = inner_hit['_source']['col_name']
                    column_description = inner_hit['_source']['col_desc']
                    results.append({
                        "table_name": table_name,
                        "column_name": column_name,
                        "column_description": column_description
                    })
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break
        search_result += json.dumps(results, ensure_ascii=False)
    except:
        search_result += f"{keyword} not found"
    return search_result    

################## SubGraph 1) Schema Linking ##################

def analyze_intent(state: GraphState) -> GraphState:
    question = state["question"]
    sys_prompt_template = "You are an assistant who understands the intent of user questions. Your task is to classify each user question into one category."
    usr_prompt_template = f"If a database query is needed to answer the user's question, respond with 'database'. Otherwise, respond with 'general'. Skip any preamble. \n\n #Question: {question}"    
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question)
    intent = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)

    return GraphState(intent=intent)

def get_sample_queries(state: GraphState) -> GraphState:
    question = state["question"]
    samples = global_object['sql_retriever'].vector_search(question)
    page_contents = [json.loads(doc.page_content) for doc in samples]

    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=global_object['region_name'])
    rerank_model_id = "cohere.rerank-v3-5:0"
    model_package_arn = f"arn:aws:bedrock:{global_object['region_name']}::foundation-model/{rerank_model_id}"

    text_sources = [
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {
                    "text": content['input'],
                }
            }
        } for content in page_contents
    ]

    response = bedrock_agent_runtime.rerank(
                queries=[
                    {
                        "type": "TEXT",
                        "textQuery": {
                            "text": question
                        }
                    }
                ],
                sources=text_sources,
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfResults": 5,
                        "modelConfiguration": {
                            "modelArn": model_package_arn,
                        }
                    }
                }
            )
    
    reranked_samples = []
    for result in response['results']:
        index = result['index']
        sample_input = page_contents[index]['input']
        sample_query = page_contents[index]['query']
        reranked_samples.append({
            'input': sample_input,
            'query': sample_query,
        })

    return GraphState(sample_queries=reranked_samples)
   
def check_readiness(state: GraphState) -> GraphState:
    print(state)
    question = state["question"]
    sample_queries = state["sample_queries"]
    table_details = state.get("table_details", "")

    sys_prompt_template = "You are a skilled database engineer who writes SQL queries for user questions. Your task is to determine whether it's possible to write an SQL query for the user's question based on the given database information."
    usr_prompt_template = "Determine if sufficient information has been provided to generate an SQL query for the question. Respond with 'Ready' if there's enough information, or 'Not Ready' if the information is insufficient. \n\n #Question: {question}\n\n #Sample queries:\n {sample_queries}\n\n #Available tables:\n {table_details} \n\n Skip the preamble or explaination. Only provide 'Ready' or 'Not Ready'"    
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, sample_queries=sample_queries, table_details=table_details)
    readiness = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)
    
    return GraphState(readiness=readiness)

def get_relevant_tables(state: GraphState) -> GraphState:
    question = state["question"]
    tables = global_object['table_retriever'].vector_search(question)
    page_contents = [doc.page_content for doc in tables if doc is not None]
    table_inputs = [json.loads(content)['table_summary'] for content in page_contents]

    sys_prompt_template = "You are a skilled database engineer who writes SQL queries to match user requests. Your task is to select the tables needed to write the SQL query. Select the tables needed to generate an SQL query that matches the user's request, sort them by importance. \n\n - Response Format: {csv_list_response_format}"
    usr_prompt_template = "Table information:\n {table_inputs}\n\n #Question: {question}\n\n Skip the preamble and only provide the valid csv format."
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, table_inputs=table_inputs, csv_list_response_format=csv_list_response_format)
    selected_tables = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)

    try:
        if selected_tables == '""':
            return GraphState(tables=[], table_names=[])
        else:
            table_names_list = [name.strip() for name in selected_tables.split(',') if name.strip()]
            tables = [json.loads(content) for content in page_contents if json.loads(content)['table_name'] in table_names_list]
            table_names = [table['table_name'] for table in tables]
            return GraphState(tables=tables, table_names=table_names)
    except:
        return GraphState(tables=[], table_names=[])

def describe_schema(state: GraphState) -> GraphState:
    table_names = state["table_names"]
    table_details = []
    inspector = inspect(global_object['engine'])
    
    for table_name in table_names:
        columns = inspector.get_columns(table_name)

        create_table_sql = f"CREATE TABLE {table_name} (\n"
        create_table_sql += ",\n".join([f"    {col['name']} {col['type']}" for col in columns])
        create_table_sql += "\n);"

        with global_object['engine'].connect() as connection:
            sample_query = text(f"SELECT * FROM {table_name} LIMIT 5")
            result = connection.execute(sample_query)
            sample_data = [dict(zip(result.keys(), row)) for row in result]
            
        table_desc = get_column_description(global_object['table_search_client'], table_name) if 'table_search_client' in globals() else {}

        table_detail = {
            "table": table_name,
            "cols": table_desc if table_desc else {col['name']: str(col['type']) for col in columns},
            "create_table_sql": create_table_sql,
            "sample_data": str(sample_data) if sample_data else "No sample data available"
        }

        if not table_detail["cols"]:
            print(f"No columns found for table {table_name}")
        table_details.append(table_detail) 
                    
    return GraphState(table_details=table_details)

def next_step_by_intent(state: GraphState) -> GraphState:
    return state["intent"]

def next_step_by_readiness(state: GraphState) -> GraphState:
    return state["readiness"]

################## SubGraph 2) Text2SQL ##################

def generate_query(state: GraphState) -> GraphState:
    dialect = global_object['dialect'] #DIALECT
    new_query_state = copy.deepcopy(global_object['initial_query_state'])
    question = state["question"]
    sample_queries = state["sample_queries"]
    table_details = state["table_details"]

    query_state = state.get("query_state", {}) or {}
    error_info = query_state.get("error", {}) or {}
    hint = error_info.get("hint", "None")
    
    sys_prompt_template = "You are a skilled database engineer who writes {dialect} SQL queries in response to user questions. Your task is to create accurate SQL queries that match the user's question based on the given database information."
    usr_prompt_template = "Based on the following sample queries, schema information, and past failure history, create a query that matches the DB dialect. Skip the introduction and provide only the generated SQL query statement. \n\n #Question: {question}\n\n #Sample queries:\n {sample_queries}\n\n #Available tables:\n {table_details}\n\n #Additional information (past failure history, additional acquired information, etc.):\n {hint}"    
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, dialect=dialect, sample_queries=sample_queries, table_details=table_details, hint=hint)
    generated_query = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)

    new_query_state["query"] = generated_query

    return GraphState(query_state=new_query_state)

def validate_query(state: GraphState) -> GraphState:
    dialect = global_object['dialect'] #DIALECT
    question = state["question"]
    query_state = copy.deepcopy(state["query_state"])
    query = query_state["query"]
    
    explain_statements = {
        'mysql': "EXPLAIN {query}",
        'mariadb': "EXPLAIN {query}",
        'sqlite': "EXPLAIN QUERY PLAN {query}",
        'oracle': "EXPLAIN PLAN FOR\n{query}\n\nSELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);",
        'postgresql': "EXPLAIN ANALYZE {query}",
        'postgres': "EXPLAIN ANALYZE {query}",
        'presto': "EXPLAIN ANALYZE {query}",
        'sqlserver': "SET STATISTICS PROFILE ON; {query}; SET STATISTICS PROFILE OFF;"
    }
    
    if dialect.lower() not in explain_statements:
        query_plan = " "
    else:
        try:
            explain_query = explain_statements[dialect.lower()].format(query=query)
            with global_object['Session'] as session:
                result = session.execute(text(explain_query))
                query_plan = "\n".join([str(row) for row in result])
        except Exception as e:
            query_state["status"] = "error"
            query_state["error"]["code"] = "E01"
            query_state["error"]["message"] = f"An error occurred while executing the EXPLAIN query: {str(e)}"
            query_state["error"]["failed_step"] = "validation"
            query_state["query"] = query
            return GraphState(query_state=query_state)

    sys_prompt_template = "You are a database expert who reviews existing {dialect} SQL queries in response to user questions and optimizes them when necessary. Your task is to examine the query's coherence and potential for optimization based on the given SQL query and additional information, and provide a final query based on this analysis." 
    usr_prompt_template = "Please add aliases to the query to match the user's question. It is not allowed to add tables or columns that were not used in the original SQL query. Skip the introduction and provide only the generated SQL query statement. \n\n #Question: {question}\n\n #Existing query:\n {query}\n\n #Query plan:\n {query_plan}"    
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, dialect=dialect, query=query, query_plan=query_plan)
    validated_query = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)
    query_state["query"] = validated_query

    return GraphState(query_state=query_state)

def execute_query(state: GraphState) -> GraphState:
    query_state = copy.deepcopy(state["query_state"])
    query = query_state["query"]
    try:
        with global_object['Session'] as session:
            result = session.execute(text(query))
            query_state["result"] = "\n".join([str(row) for row in result])
    except Exception as e:
        query_state["status"] = "error"
        query_state["error"]["code"] = "E02"
        query_state["error"]["message"] = f"An error occurred while executing the validated query: {str(e)}"
        query_state["error"]["failed_step"] = "execution"
        return GraphState(query_state=query_state)
    return GraphState(query_state=query_state)
    
def handle_failure(state: GraphState) -> GraphState:
    query_state = copy.deepcopy(state["query_state"])
    query = query_state['query']
    message = query_state['error']['message']
    sys_prompt_template = "You are a skilled database engineer who handles SQL query failures. Your task is to identify the cause of failure for the given SQL query and determine the next steps for problem resolution."
    usr_prompt_template = """Based on the failure message of the given SQL query, provide one of the following causes (`failure_type`) along with a clue for resolution (`hint`).
Here are examples of failure_type choices:
Inaccurate query syntax: `syntax_check`
Schema mismatch (no such table or column): `schema_check`
External DB factors (permissions, connection issues, etc.): `stop`
Temporary DB malfunction (query re-execution needed): `retry`

#Failed query: {query}

#Failure message: {message}

#Format: 
{{
    "failure_type": "<one of the failure types mentioned above>",
    "hint": "<brief explanation or suggestion for resolution>"
}}

Skip the preamble and only provide the valid JSON document."""

    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, query=query, message=message)
    result = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)

    try:
        json_result = json.loads(result)
        failure_type = json_result.get("failure_type", "unknown")
        hint = json_result.get("hint", "No hint provided")
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response: {result}")
        failure_type = "unknown"
        hint = "Failed to parse AI response"

    query_state["hint"] = hint
    
    return GraphState(next_action=failure_type, query_state=query_state)

def get_relevant_columns(state: GraphState) -> GraphState:
    query_state = copy.deepcopy(state["query_state"])
    question = state["question"]
    query = query_state["query"]
    message = query_state['error']['message']
    sys_prompt_template = "You are an expert SQL query troubleshooter. Your task is to analyze failed queries and suggest relevant keywords for schema exploration to resolve the issue."
    usr_prompt_template = """Given a user question, a failed SQL query, and an error message, provide 3-5 relevant keywords or phrases for database schema exploration. These should help in finding the correct table and column names to fix the query.\n\n#User question: {question}\n\n#Failed query:\n{query}\n\n#Error message:\n{message}\n\nRespond only with a comma-separated list of keywords or short phrases, without any additional text or explanation.\n\n#Format: {csv_list_response_format}"""
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query=query, message=message, csv_list_response_format=csv_list_response_format)
    keywords = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)
    return keywords

def next_step_by_query_state(state:GraphState) -> GraphState:
    return state["query_state"]["status"]

def next_step_by_next_action(state:GraphState) -> GraphState:
    return state["next_action"]

################## Answer generation nodes ##################
def get_general_answer(state: GraphState) -> GraphState:
    question = state["question"]
    sys_prompt_template = "You are a capable assistant who answers general questions from users. If you don't know the answer to a question, admit that you don't know."
    usr_prompt_template = "#Question: {question}"
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question)
    answer = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)

    return GraphState(answer=answer)

def get_database_answer(state: GraphState) -> GraphState:
    question = state["question"]
    query_state = state["query_state"]
    query = query_state["query"]
    data = query_state["result"]
    failed_step = query_state["error"]["failed_step"]
    message = query_state["error"]["message"]
    sys_prompt_template = "You are a competent assistant who answers user questions based on database information. Your task is to provide thorough answers to user questions, referencing the given information."
    
    if query_state["status"] == "success":
        usr_prompt_template = "The answer should include the used query, dataframe (as a Markdown Table), and a brief response to the question. \n\n#Question: {question}\n\n#Used query: {query}\n\n#Data: {data}\n\n"
        sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query=query, data=data)
    else:
        usr_prompt_template = "The following is a record of a failed query execution for a user question. Based on this, explain why the request processing failed.\n\n#Question: {question}\n\n#Used query: {query}\n\n#Failed step: {failed_step}\n\n#Error message: {message}\n\n"
        sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query=query, failed_step=failed_step, message=message)    
        
    answer = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt)
    return GraphState(answer=answer)


# ### legacy
# def select_subgraph(state: GraphState) -> GraphState:
#     question = state['question']  
#     query = """
#         MATCH (n:Title {level: "1"})
#         RETURN n.value, id(n) as node_id
#     """
#     results = global_object['graph'].run(query)
#     subgraph_list = [(record["n.value"], record["node_id"]) for record in results]
#     subgraph_list_with_number = [f"{i}. {subgraph[0]}" for i, subgraph in enumerate(subgraph_list)]

#     sys_prompt_template = """ 
#     You are an expert engineer well-versed in AWS manual documents. 
#     Your task is to select the most appropriate manual document name for the user's question. 
#     If there are no relevant documents, provide an empty list (""). """

#     usr_prompt_template = """ 
#     Please select the single most relevant document name for the given question.

#     #Question: {question}

#     #Document List: {subgraph_list_with_number}

#     #Response Format: Provide only the index number of the selected document (omit any preamble) """
#     sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, subgraph_list_with_number=subgraph_list_with_number)
    
#     model_id = state['support_model']
#     selected_id = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)

#     try:
#         if selected_id == "":
#             return GraphState(target_node=[], subgraph="Not Found", next_step="global_search")

#         else: 
#             selected_subgraph_id = subgraph_list[int(selected_id)][1]
#             print("Selected:", subgraph_list[int(selected_id)][0])
#             return GraphState(target_node=[selected_subgraph_id], subgraph=subgraph_list[int(selected_id)][0], next_step="traverse_subgraph")
#     except:
#         return GraphState(target_node=[], subgraph="Not Found", next_step="global_search")

# def traverse_subgraph(state: GraphState) -> GraphState:
#     question = state['question']
    
#     if not state['traverse_state']:
#         parent_id = state['target_node'][0]
#     else:
#         parent_id = state['traverse_state'][-1]['selected_child_ids'][0]
#     subgraph = state['subgraph']

#     query = """
#         MATCH (n)
#         WHERE id(n) = $parent_id
#         OPTIONAL MATCH (n)-[:HAS_CHILD]->(c)
#         RETURN n.value as parent_name, c.level as child_level, c.value as child_name, id(c) as child_id
#     """
#     params = {"parent_id": parent_id}
#     query_results = global_object['graph'].run(query, params)

#     parent_name = None
#     child_level = None
#     child_list = []
#     child_names = []

#     for record in query_results:
#         if parent_name is None:
#             parent_name = record["parent_name"]
#         if child_level is None:
#             child_level = record["child_level"]
#         if record["child_name"] is not None:
#             child_list.append((record["child_name"], record["child_id"]))
#             child_names.append(record["child_name"])

#     print(f"Traversing '{parent_name}'...")

#     if not child_list:
#         print("No child. Proceed to 'get_contents'...")
#         traverse_state: TraverseState = {
#             "parent_id": parent_id,
#             "parent_name": parent_name,
#             "child_level": -1,
#             "selected_child_ids": [],
#             "child_names": [],
#             "next_action": "get_contents"
#         }
#         return GraphState(traverse_state=state['traverse_state'] + [traverse_state])


#     full_path = " > ".join([state["parent_name"] for state in state['traverse_state']])
#     full_path = f"{full_path} > {parent_name}"

#     child_list_with_number = [f"{i}. {child}" for i, child in enumerate(child_list)]
# #    csv_list_response_format = "Your response should be a list of comma separated values, eg: `foo, bar` or `foo,bar`"
#     sys_prompt_template = """ You are an AI assistant to answer the user's question with the provided documentation.
#     Your task involves staying at the current menu level or selecting relevant sub-menu(s) when necessary 

#     Instructions:
#     1. Carefully analyze the current menu and sub-menus. .
#     2. If the current menu level adequately addresses the question, do not select any sub-menu.
#     3. If a specific sub-menu is expected to offer additional relevant information, select one.
#         - Only select when the question explicitly requires that level of detail from the sub-menus.
#         - Consider keywords from the question to gauge the required depth.
#     4. Respond with the index number of the selected menu item, starting from 0.

#     Depth guidance:
#     - For overview questions, prioritize higher-level menu items over sub-menu items.
#     - Consider going one level deeper if sub-menus directly relate to key aspects of the question. 
#     - Balance providing sufficient information without excessive detail.

#     Important: Respond ONLY with index number or an empty string("")."""

#     usr_prompt_template = """
#     Question: {question}

#     Current menu path: <{subgraph}> - {full_path}

#     SubMenu list: 
#     {child_list_with_number}"""
# #    Response format: {csv_list_response_format} 
#     sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, 
#                                            subgraph=subgraph, question=question, 
#                                            full_path=full_path, child_list_with_number=child_list_with_number) 
# #                                           csv_list_response_format=csv_list_response_format)
#     model_id = state['core_model']
#     selected_ids = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)
    
#     try:
#         selected_id_list = [int(id.strip()) for id in selected_ids.split(',') if id.strip().isdigit()]

#         if not selected_id_list:
#             traverse_state: TraverseState = {
#                 "parent_id": parent_id,
#                 "parent_name": parent_name,
#                 "child_level": -1,
#                 "selected_child_ids": [],
#                 "child_names": [],
#                 "next_action": "get_contents"
#             }
#             return GraphState(traverse_state=state['traverse_state'] + [traverse_state])

#         selected_child_ids = [child_list[id][1] for id in selected_id_list if id < len(child_list)]
#         selected_child_names = [child_list[id][0] for id in selected_id_list if id < len(child_list)]
#         traverse_state: TraverseState = {
#             "parent_id": parent_id,
#             "parent_name": parent_name,
#             "child_level": child_level,
#             "selected_child_ids": selected_child_ids,
#             "child_names": selected_child_names,
#             "next_action": "traverse_subgraph"
#         }
#         return GraphState(traverse_state=state['traverse_state'] + [traverse_state])

#     except Exception as e:
#         traverse_state: TraverseState = {
#             "parent_id": parent_id,
#             "parent_name": parent_name,
#             "child_level": -1,
#             "selected_child_ids": [],
#             "child_names": [],
#             "next_action": "get_contents"
#         }
#         return GraphState(traverse_state=state['traverse_state'] + [traverse_state])

# def get_contents(state: GraphState) -> GraphState:
#     parent_id = state['traverse_state'][-1]['parent_id']
#     k = state['k']

#     count_query = """
#         MATCH (n)-[:HAS_CONTENTS]->(c)
#         WHERE id(n) = $parent_id
#         RETURN count(c) as contents_length, n.value as parent_name
#     """
#     params = {"parent_id": parent_id}
#     count_result = global_object['graph'].run(count_query, params).data()[0]
#     contents_length = count_result['contents_length']
#     parent_name = count_result['parent_name']
#     print(f"Num Documents: {contents_length}")

#     if contents_length <= k * 2:
#         search_type = "get_short_documents"
#         content_query = """
#             MATCH (n)-[:HAS_CONTENTS]->(c)
#             WHERE id(n) = $parent_id
#             RETURN c.text
#             ORDER BY c.order
#             LIMIT $k
#         """
#         params = {"parent_id": parent_id, "k": k}
#         content_results = global_object['graph'].run(content_query, params)
#         contents = [record["c.text"] for record in content_results]
#         context = " ".join(contents)

#     else:
#         search_type = "node_level_search"
#         context = ""

#     new_state = state.copy()
#     new_state.update({
#         'parent_id': parent_id,
#         'parent_name': parent_name,
#         'content': context,
#         'contents_length': contents_length,
#         'search_type': search_type,
#         'k': k
#     })

#     return new_state

# def node_level_search(state: GraphState) -> GraphState:
#     question = state['question']  
#     subgraph = state['subgraph']
#     parent_id = state['parent_id']
#     parent_name = state['parent_name']
#     language = state['language']
#     searching_scheme = state['searching_scheme']
#     k = state['k']
    
#     csv_list_response_format = "Your response should be a list of comma separated values, eg: `foo, bar` or `foo,bar`"
#     if searching_scheme == "vector":
#         sys_prompt_template = """
#         You are an expert engineer well-versed in AWS. Based on the user's question, please create a question suitable for vector search to find documents in the manual. ({language})

#         Note:
#         - Generate the most relevant and characteristic question considering both the document name and the user's question.
#         - Prefer natural language questions that well represent specific content of the document.
#         """

#         usr_prompt_template = """
#         #Target document name:
#         {subgraph}

#         #Question:
#         {question}

#         #Response format:
#         {csv_list_response_format}
#         """

#         sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, language=language, subgraph=subgraph, question=question, csv_list_response_format=csv_list_response_format)
#         model_id = state['support_model'] 
#         keywords = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)

#         index_name = "content_embedding_index"
#         embeddings = BedrockEmbeddings(model_id="cohere.embed-multilingual-v3", region_name=state['region_name'])
#         vector_store = Neo4jVector.from_existing_index(
#             embedding=embeddings,
#             url=global_object['graph_url'],
#             username=global_object['graph_username'],
#             password=global_object['graph_password'],
#             index_name=index_name,
#             node_label="Content",
#             text_node_property="text", 
#             embedding_node_property="embedding"
#         )

#         question_embedding = embeddings.embed_query(keywords)

#         vector_search_query = """
#             MATCH (parent)-[:HAS_CONTENTS]->(child:Content)
#             WHERE id(parent) = $parent_id
#             WITH child
#             CALL db.index.vector.queryNodes($index_name, $k, $question_embedding) YIELD node, score
#             WHERE node = child
#             RETURN id(node) AS node_id, node.text AS text, score
#             ORDER BY score DESC
#             LIMIT $k
#         """

#         params = {
#             "parent_id": parent_id,
#             "question_embedding": question_embedding,
#             "k": k,
#             "index_name": index_name
#         }

#         search_results = vector_store.query(vector_search_query, params=params)
        
#     else:
#         sys_prompt_template = """
#         You are an expert engineer well-versed in AWS. Based on the user's question, extract one core keyword from the manual.
#         The keyword must meet the following conditions:
#         1. No special characters like '_', '-' in the keyword (e.g., respond with 'custom model' instead of 'custom_model')
#         2. Choose the most appropriate word within the given document name that fits the context of the question (no need to include the document name in the keyword)
#         3. Select specific words that represent the particular function or concept you're searching for, rather than content already included in the document name
#         4. Provide the keyword in {language}

#         Note:
#         - Consider both the document name and the question to select the most relevant and characteristic word within that document.
#         - Prefer words that represent specific content of the document rather than overly general terms.
#         """

#         usr_prompt_template = """
#         #Target document name:
#         {parent_name}

#         #Question:
#         {question}

#         #Response format:
#         {csv_list_response_format}
#         """
#         sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, language=language, parent_name=parent_name, question=question, csv_list_response_format=csv_list_response_format)
#         model_id = state['support_model'] 
#         keywords = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)
    

#         if searching_scheme == "full_text":
#             search_query = """MATCH (parent)-[:HAS_CONTENTS]->(child)
#                 WHERE id(parent) = $parent_id
#                 WITH child
#                 CALL db.index.fulltext.queryNodes("Search_Content_by_FullText", $keywords) YIELD node, score
#                 WHERE node = child
#                 RETURN node.text as text, score
#                 ORDER BY score DESC
#                 LIMIT $k
#             """
#         elif searching_scheme == "keyword":
#             search_query = """MATCH (parent)-[:HAS_CONTENTS]->(child)
#                 WHERE id(parent) = $parent_id
#                 WITH child, $keywords AS keyword
#                 WHERE child.text CONTAINS keyword
#                 RETURN child.text AS text, 
#                     size(split(toLower(child.text), toLower(keyword))) - 1 AS score
#                 ORDER BY score DESC
#                 LIMIT $k
#             """
        
#         params = {"parent_id": parent_id, "keywords": keywords, "k": k}
#         search_results = global_object['graph'].run(search_query, params)
        
#     content = "\n\n\n".join(f"{record['text']} (Score: {record['score']})" for record in search_results)
#     return GraphState(content=content)

# def check_relevance(state: GraphState) -> GraphState:
#     question = state['question']
#     content = state['content']
#     contents_length = state['contents_length']
#     search_type = state['search_type']
#     subgraph = state['subgraph']
#     parent_name = state['parent_name']
#     k = state['k']
#     optional_prompt1 = ""
#     optional_prompt2 = ""
    
#     if search_type == "get_short_documents" and contents_length > k:
#         optional_prompt1 = "- Preliminary information matches the question's intent but further content needs to be checked: `Partial`"
#         optional_prompt2 = "or `Partial`"

#     sys_prompt_template = """
#     You are an information analysis expert. Your task is to determine if the provided preliminary information is sufficient to answer the given question.

#     Judgment Criteria:
#     1. Check if key keywords from the question appear in the document name or preliminary information.
#     2. Evaluate if the preliminary information directly addresses the core aspects of the question.

#     Response:
#     - If the document name and preliminary information are not relevant to the question, respond with 'None'.
#     - If the question can be fully answered with the preliminary information alone, respond with 'Complete'.
#     {partial1}

#     Skip any preamble and respond only with `None` or `Complete`{partial2}.
#     """

#     usr_prompt_template = """
#     #Preliminary information (Document name: {subgraph} / Menu: {parent_name})
#     {context}

#     #Question: {question}
#     """

#     sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, partial1=optional_prompt1, partial2=optional_prompt2, subgraph=subgraph, parent_name=parent_name, question=question, context=content)
#     model_id = state['core_model']
#     response = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)
#     response = response.lower().strip()
#     if 'complete' in response:
#         status = 'Complete'
#     elif 'none' in response:
#         status = 'None'
#     elif 'partial' in response:
#         status = 'Partial'
#     else:
#         status = 'Invalid'

#     return GraphState(status=status)

# def get_sibling_contents(state: GraphState) -> GraphState:
    
#     content = state['content']
#     parent_id = state['parent_id']
#     k = state['k']

#     content_query = """
#         MATCH (n)-[:HAS_CONTENTS]->(c)
#         WHERE id(n) = $parent_id AND c.order >= $order_pos
#         RETURN c.text
#         ORDER BY c.order
#         LIMIT $k
#     """       
#     trial = 1
#     order_pos = k * trial
#     params = {"parent_id": parent_id, "k": k, "order_pos": order_pos} 

#     content_results = global_object['graph'].run(content_query, params)
#     sibling_content = [record["c.text"] for record in content_results]
#     content = " ".join([content] + sibling_content)
    
#     return GraphState(content=content)

# def subgraph_level_search(state: GraphState) -> GraphState:
#     question = state['question']
    
#     subgraph = state['subgraph']
#     language = state['language']
#     k = state['k']
#     searching_scheme = state['searching_scheme']

#     csv_list_response_format = "Your response should be a list of comma separated values, eg: `foo, bar` or `foo,bar`"
#     if searching_scheme == "vector":
#         sys_prompt_template = """
#         You are an expert engineer well-versed in AWS. Based on the user's question, please create a question suitable for vector search to find documents in the manual. ({language})

#         Note:
#         - Generate the most relevant and characteristic question considering both the document name and the user's question.
#         - Prefer natural language questions that well represent specific content of the document.
#         """

#         usr_prompt_template = """
#         #Target document name:
#         {subgraph}

#         #Question:
#         {question}

#         #Response format:
#         {csv_list_response_format}
#         """

#         sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, language=language, subgraph=subgraph, question=question, csv_list_response_format=csv_list_response_format)
#         model_id = state['support_model'] 
#         keywords = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)
#         index_name = "content_embedding_index"

#         embeddings = BedrockEmbeddings(model_id="cohere.embed-multilingual-v3", region_name=state['region_name'])
#         question_embedding = embeddings.embed_query(keywords)

#         vector_search_query = """
#         MATCH (root:Title {level: "1", value: $subgraph})
#         MATCH (root)-[:HAS_CHILD*0..]->(title:Title)-[:HAS_CONTENTS]->(content:Content)

#         CALL db.index.vector.queryNodes($index_name, $k, $question_embedding) YIELD node, score
#         WHERE node = content

#         RETURN node.text AS text, score
#         ORDER BY score DESC
#         LIMIT $k
#         """

#         params = {
#             "subgraph": subgraph,
#             "question_embedding": question_embedding,
#             "k": k,
#             "index_name": index_name
#         }

#         vector_store = Neo4jVector.from_existing_index(
#             url=global_object['graph_url'],
#             username=global_object['graph_username'],
#             password=global_object['graph_password'],
#             embedding=embeddings,
#             index_name=index_name,
#             node_label="Content",
#             text_node_property="text", 
#             embedding_node_property="embedding"
#         )

#         search_results = vector_store.query(vector_search_query, params=params)

#     else:
#         sys_prompt_template = """
#         You are an expert engineer well-versed in AWS. Based on the user's question, extract one core keyword from the manual.
#         The keyword must meet the following conditions:
#         1. No special characters like '_', '-' in the keyword (e.g., respond with 'custom model' instead of 'custom_model')
#         2. Choose the most appropriate word within the given document name that fits the context of the question (no need to include the document name in the keyword)
#         3. Select specific words that represent the particular function or concept you're searching for, rather than content already included in the document name
#         4. Provide the keyword in {language}

#         Note:
#         - Consider both the document name and the question to select the most relevant and characteristic word within that document.
#         - Prefer words that represent specific content of the document rather than overly general terms.
#         """

#         usr_prompt_template = """
#         #Target document name:
#         {subgraph}

#         #Question:
#         {question}

#         #Response format:
#         {csv_list_response_format}
#         """
#         sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, language=language, subgraph=subgraph, question=question, csv_list_response_format=csv_list_response_format)
#         model_id = state['support_model'] 
#         keywords = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)
    
        
#         if searching_scheme == "full_text":
#             search_query = """
#                 MATCH (root:Title {level: "1", value: $subgraph})
#                 MATCH (root)-[:HAS_CHILD*0..]->(title:Title)-[:HAS_CONTENTS]->(content:Content)

#                 CALL db.index.fulltext.queryNodes("Search_Content_by_FullText", $keywords) YIELD node, score
#                 WHERE node = content

#                 RETURN node.text as text, score, title.name as title_name, title.level as title_level
#                 ORDER BY score DESC
#                 LIMIT $k
#             """
#         elif searching_scheme == "keyword":
#             search_query = """
#                 MATCH (root:Title {level: "1", value: $subgraph})
#                 MATCH (root)-[:HAS_CHILD*0..]->(title:Title)-[:HAS_CONTENTS]->(content:Content)
#                 WITH content, title, $keywords AS keyword
#                 WHERE content.text CONTAINS keyword
#                 RETURN content.text AS text, 
#                     size(split(toLower(content.text), toLower(keyword))) - 1 AS score,
#                     {
#                         title: title.name,
#                         level: title.level,
#                         value: title.value
#                     } AS metadata
#                 ORDER BY score DESC
#                 LIMIT $k
#             """
        
#         params = { "subgraph": subgraph, "k": k, "keywords": keywords}
#         search_results = global_object['graph'].run(search_query, params)
        
#     content = "\n\n\n".join(f"{record['text']} (Score: {record['score']})" for record in search_results)
#     return GraphState(content=content)


# def global_search(state: GraphState) -> GraphState:
#     question = state['question']
    
#     language = state['language']
#     searching_scheme = state['searching_scheme']
#     k = state['k']
#     csv_list_response_format = "Your response should be a list of comma separated values, eg: `foo, bar` or `foo,bar`"
    
#     if searching_scheme == "vector":
#         sys_prompt_template = """
#         You are an expert engineer well-versed in AWS. Based on the user's question, please create a question suitable for vector search to find documents in the manual. ({language})

#         Note:
#         - Generate the most relevant and characteristic question considering both the document name and the user's question.
#         - Prefer natural language questions that well represent specific content of the document.
#         """

#         usr_prompt_template = """
#         #Question:
#         {question}

#         #Response format:
#         {csv_list_response_format}
#         """
    
#         sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, language=language, question=question, csv_list_response_format=csv_list_response_format)
#         model_id = state['support_model'] 
#         keywords = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)

#         index_name = "content_embedding_index"

#         embeddings = BedrockEmbeddings(model_id="cohere.embed-multilingual-v3", region_name=state['region_name'])
#         question_embedding = embeddings.embed_query(keywords)

#         vector_search_query = """
#         CALL db.index.vector.queryNodes($index_name, $k, $question_embedding) YIELD node, score
#         WITH DISTINCT node, score
#         WHERE node:Content
#         RETURN node.text AS text, score
#         ORDER BY score DESC
#         """

#         params = {
#             "question_embedding": question_embedding,
#             "k": k,
#             "index_name": index_name
#         }

#         vector_store = Neo4jVector.from_existing_index(
#             url=global_object['graph_url'],
#             username=global_object['graph_username'],
#             password=global_object['graph_password'],
#             embedding=embeddings,
#             index_name=index_name,
#             node_label="Content",
#             text_node_property="text", 
#             embedding_node_property="embedding"
#         )

#         search_results = vector_store.query(vector_search_query, params=params)

#     else:
#         sys_prompt_template = """
#         You are an expert engineer well-versed in AWS. Based on the user's question, extract one core keyword from the manual.
#         The keyword must meet the following conditions:
#         1. No special characters like '_', '-' in the keyword (e.g., respond with 'custom model' instead of 'custom_model')
#         2. Choose the most appropriate word within the given document name that fits the context of the question (no need to include the document name in the keyword)
#         3. Select specific words that represent the particular function or concept you're searching for, rather than content already included in the document name
#         4. Provide the keyword in {language}

#         Note:
#         - Consider both the document name and the question to select the most relevant and characteristic word within that document.
#         - Prefer words that represent specific content of the document rather than overly general terms.
#         """

#         usr_prompt_template = """
#         #Question:
#         {question}

#         #Response format:
#         {csv_list_response_format}
#         """
#         sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, language=language, question=question, csv_list_response_format=csv_list_response_format)
#         model_id = state['support_model'] 
#         keywords = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)
#         print(keywords)
        
#         if searching_scheme == "full_text":
#             search_query ="""
#             CALL db.index.fulltext.queryNodes("Search_Content_by_FullText", $keywords) YIELD node, score
#             WHERE node:Content
#             OPTIONAL MATCH (title:Title)-[:HAS_CONTENTS]->(node)
#             RETURN node.text as text, score, title.name as title_name, title.level as title_level
#             ORDER BY score DESC
#             LIMIT $k
#             """
#         elif searching_scheme == "keyword":
#             search_query = """
#             MATCH (content:Content)
#             WITH content, $keywords AS keyword
#             WHERE content.text CONTAINS keyword
#             OPTIONAL MATCH (title:Title)-[:HAS_CONTENTS]->(content)
#             RETURN content.text AS text, 
#                 size(split(toLower(content.text), toLower(keyword))) - 1 AS score,
#                 {
#                     title: title.name,
#                     level: title.level,
#                     value: title.value
#                 } AS metadata
#             ORDER BY score DESC
#             LIMIT $k
#             """
        
#         params = {"k": k, "keywords": keywords}
#         search_results = global_object['graph'].run(search_query, params)
        
#     content = "\n\n\n".join(f"{record['text']} (Score: {record['score']})" for record in search_results)
#     return GraphState(content = content)

# def generate_answer(state: GraphState) -> GraphState:
#     question = state['question']
#     context = state['content']

#     sys_prompt_template = """
#     You are an expert engineer well-versed in AWS. 
#     Generate an answer to the user's question using only the given preliminary information. """

#     usr_prompt_template = """
#     #Preliminary information: 
#     {context}

#     #User question:
#     {question}
#     """
    
#     sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, context=context, question=question)
#     model_id = state['core_model'] 
#     answer = converse_with_bedrock(global_object['boto3_client'], sys_prompt, usr_prompt, model_id)

#     return GraphState(answer=answer)

# def next_step_by_subgraph(state: GraphState) -> str:
#     return state['next_step']

# def next_step_by_traverse_state(state: GraphState) -> str:
#     if state['traverse_state'][-1]['next_action'] == "get_contents":
#         next_action = "get_contents"
#     else:
#         next_action = "traverse_subgraph"
#     return next_action

# def next_step_by_context(state: GraphState) -> str:
#     return state['search_type']

# def next_step_by_relevance(state: GraphState) -> str:
#     return state['status']


def build_langgraph_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("select_subgraph", select_subgraph)
    workflow.set_entry_point("select_subgraph")
    workflow.add_node("traverse_subgraph", traverse_subgraph)
    workflow.add_node("get_contents", get_contents)
    workflow.add_node("node_level_search", node_level_search)
    workflow.add_node("check_relevance", check_relevance)
    workflow.add_node("get_sibling_contents", get_sibling_contents)
    workflow.add_node("subgraph_level_search", subgraph_level_search)
    workflow.add_node("global_search", global_search)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_conditional_edges(
        "select_subgraph",
        next_step_by_subgraph,
        {
            "global_search": "global_search",
            "traverse_subgraph": "traverse_subgraph",
        }
    )
    workflow.add_conditional_edges(
        "traverse_subgraph",
        next_step_by_traverse_state,
        {
            "get_contents": "get_contents",
            "traverse_subgraph": "traverse_subgraph",
        }
    )
    workflow.add_conditional_edges(
        "get_contents",
        next_step_by_context,
        {
            "get_short_documents": "check_relevance",
            "node_level_search": "node_level_search"
        }
    )
    workflow.add_edge("node_level_search", "check_relevance")

    workflow.add_conditional_edges(
        "check_relevance",
        next_step_by_relevance,
        {
            "Complete": "generate_answer",
            "Partial": "get_sibling_contents",
            "None": "subgraph_level_search"
        }
    )
    workflow.add_edge("get_sibling_contents", "generate_answer")
    workflow.add_edge("subgraph_level_search", "generate_answer")
    workflow.add_edge("global_search", "generate_answer")
    workflow.add_edge("generate_answer", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def display_traversal_progress(output, progress_container):
    if 'select_subgraph' in output:
        progress_container.info(f"🔍 Selecting subgraph: {output['select_subgraph']['subgraph']}")
    elif 'traverse_subgraph' in output:
        traverse_state = output['traverse_subgraph']['traverse_state']
        if traverse_state:
            current_node = traverse_state[-1]
            progress_container.info(f"🚶 Traversing: {current_node['parent_name']}")
            # if current_node['child_names']:
            #     content_str = "📄 Content: " + " | ".join([f" {child} " for child in current_node['child_names']])
            #     progress_container.success(content_str)
    elif 'get_contents' in output:
        progress_container.info("📚 Retrieving contents...")
    elif 'node_level_search' in output:
        progress_container.info("🔬 Performing node-level search...")
    elif 'check_relevance' in output:
        status = output['check_relevance']['status']
        if status == 'None':
            progress_container.warning(f"✅ Checking relevance: {status}")
        else:
            progress_container.success(f"✅ Checking relevance: {status}")
    elif 'get_sibling_contents' in output:
        progress_container.info("👥 Getting sibling contents...")
    elif 'subgraph_level_search' in output:
        progress_container.info("🔍 Performing subgraph-level search...")
    elif 'global_search' in output:
        progress_container.info("🌐 Performing global search...")
    elif 'generate_answer' in output:
        progress_container.info("💡 Generating final answer...")

def run_workflow(prompt, app, core_model, support_model, region_name, progress_container):
    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
    inputs = GraphState(
        question=prompt,
        core_model=core_model,
        support_model=support_model,
        region_name=region_name,
        searching_scheme="vector",  
        k=5,  
        language="English",  
        traverse_state=[]
    )

    try:
        final_output = None
        for output in app.stream(inputs, config=config):
            final_output = output
            display_traversal_progress(output, progress_container)

        if final_output:
            if 'generate_answer' in final_output and 'answer' in final_output['generate_answer']:
                return final_output['generate_answer']['answer']
            elif 'generate_answer' in final_output:
                return str(final_output['generate_answer'])
            else:
                for key, value in final_output.items():
                    if isinstance(value, str) and len(value) > 0:
                        return value

        return "I'm sorry, I couldn't generate a response."
    except GraphRecursionError as e:
        return f"An error occurred: Recursion limit reached. {str(e)}"