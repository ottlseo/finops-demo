import boto3
import json
import copy
from typing import TypedDict
from botocore.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from src.opensearch import OpenSearchVectorRetriever, OpenSearchClient
from src.common_utils import SQLDatabase

class GraphState(TypedDict):
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
    
global_object = {
    "boto3_client": None,
    "sql_search_client": None,
    "table_search_client": None,
    "sql_retriever": None,
    "table_retriever": None,
    "engine": None,
    "db": None,
    "Session": None,
    "region_name": None,
    "llm_model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "DIALECT": "sqlite",
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
def init_boto3_client(region_name: str = None):
    """Initialize Bedrock runtime client"""
    if region_name is None:
        region_name = global_object.get("region_name")
    
    retry_config = Config(
        region_name=region_name,
        retries={"max_attempts": 10, "mode": "standard"}
    )
    return boto3.client("bedrock-runtime", region_name=region_name, config=retry_config)

def init_search_resources(region_name: str = None):
    """Initialize OpenSearch clients and retrievers"""
    if region_name is None:
        region_name = global_object.get("region_name")
    
    sql_search_client = OpenSearchClient(
        region_name=region_name, 
        index_name='example_queries', 
        mapping_name='mappings-sql', 
        vector="input_v", 
        text="input", 
        output=["input", "query"]
    )
    
    table_search_client = OpenSearchClient(
        region_name=region_name, 
        index_name='schema_descriptions', 
        mapping_name='mappings-detailed-schema', 
        vector="table_summary_v", 
        text="table_summary", 
        output=["table_name", "table_summary"]
    )

    sql_retriever = OpenSearchVectorRetriever(sql_search_client, region_name=region_name, k=20)
    table_retriever = OpenSearchVectorRetriever(table_search_client, region_name=region_name, k=10)
    
    return sql_search_client, table_search_client, sql_retriever, table_retriever

def init_resources():
    # Initialize all resources
    global_object["region_name"] = boto3.Session().region_name
    global_object["boto3_client"] = init_boto3_client(global_object["region_name"])
    global_object["engine"] = create_engine("sqlite:///Chinook.db")
    global_object["db"] = SQLDatabase(global_object["engine"])
    global_object["Session"] = sessionmaker(bind=global_object["engine"])
    
    # Initialize search resources
    (global_object["sql_search_client"], 
     global_object["table_search_client"], 
     global_object["sql_retriever"], 
     global_object["table_retriever"]) = init_search_resources()

def converse_with_bedrock(sys_prompt, usr_prompt):
    temperature = 0.0
    top_p = 0.1
    top_k = 1
    inference_config = {"temperature": temperature, "topP": top_p}
    additional_model_fields = {"top_k": top_k}
    response = global_object["boto3_client"].invoke_model(
        modelId=global_object["llm_model"],
        body=json.dumps({
            "messages": usr_prompt,
            "system": sys_prompt,
            "max_tokens": 1024,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        })
    )
    return json.loads(response['body'].read())['completion']

def create_prompt(sys_template: str, user_template: str, **kwargs) -> tuple:
    """Create system and user prompts for LLM interaction"""
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt

def get_column_description(table_name: str) -> dict:
    """Get column descriptions for a specific table"""
    query = {
        "query": {
            "match": {
                "table_name": table_name
            }
        }
    }
    
    table_search_client = global_object.get("table_search_client")
    if not table_search_client:
        return {}
        
    response = table_search_client.conn.search(
        index=table_search_client.index_name, 
        body=query
    )

    if response['hits']['total']['value'] > 0:
        source = response['hits']['hits'][0]['_source']
        columns = source.get('columns', [])
        if columns:
            return {col['col_name']: col['col_desc'] for col in columns}
    return {}

def search_by_keywords(keyword: str) -> str:
    """Search tables and columns by keyword"""
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
    
    table_search_client = global_object.get("table_search_client")
    if not table_search_client:
        return f"{keyword} not found"
        
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
                    
        search_result = json.dumps(results, ensure_ascii=False)
    except Exception as e:
        search_result = f"{keyword} not found (Error: {str(e)})"
        
    return search_result
    
################## SubGraph 1) Schema Linking ##################
def analyze_intent(state: GraphState) -> GraphState:
    """Analyze the intent of the user question"""
    question = state["question"]
    sys_prompt_template = "You are an assistant who understands the intent of user questions. Your task is to classify each user question into one category."
    usr_prompt_template = f"If a database query is needed to answer the user's question, respond with 'database'. Otherwise, respond with 'general'. Skip any preamble. \n\n #Question: {question}"    
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question)
    intent = converse_with_bedrock(sys_prompt, usr_prompt)
    
    return GraphState(intent=intent)

def get_sample_queries(state: GraphState) -> GraphState:
    """Get and rerank similar queries"""
    question = state["question"]
    samples = global_object["sql_retriever"].vector_search(question)
    page_contents = [json.loads(doc.page_content) for doc in samples]

    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', 
                                       region_name=global_object["region_name"])
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
        queries=[{
            "type": "TEXT",
            "textQuery": {
                "text": question
            }
        }],
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
    """Check if enough information is available to generate a query"""
    print(state)
    question = state["question"]
    sample_queries = state["sample_queries"]
    table_details = state.get("table_details", "")

    sys_prompt_template = "You are a skilled database engineer who writes SQL queries for user questions. Your task is to determine whether it's possible to write an SQL query for the user's question based on the given database information."
    usr_prompt_template = "Determine if sufficient information has been provided to generate an SQL query for the question. Respond with 'Ready' if there's enough information, or 'Not Ready' if the information is insufficient. \n\n #Question: {question}\n\n #Sample queries:\n {sample_queries}\n\n #Available tables:\n {table_details} \n\n Skip the preamble or explaination. Only provide 'Ready' or 'Not Ready'"    
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, 
                                         question=question, sample_queries=sample_queries, 
                                         table_details=table_details)
    readiness = converse_with_bedrock(sys_prompt, usr_prompt)
    
    return GraphState(readiness=readiness)

def get_relevant_tables(state: GraphState) -> GraphState:
    """Get relevant tables for the query"""
    question = state["question"]
    tables = global_object["table_retriever"].vector_search(question)
    page_contents = [doc.page_content for doc in tables if doc is not None]
    table_inputs = [json.loads(content)['table_summary'] for content in page_contents]

    sys_prompt_template = "You are a skilled database engineer who writes SQL queries to match user requests. Your task is to select the tables needed to write the SQL query. Select the tables needed to generate an SQL query that matches the user's request, sort them by importance. \n\n - Response Format: {csv_list_response_format}"
    usr_prompt_template = "Table information:\n {table_inputs}\n\n #Question: {question}\n\n Skip the preamble and only provide the valid csv format."
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, 
                                         question=question, table_inputs=table_inputs, 
                                         csv_list_response_format=global_object["csv_list_response_format"])
    selected_tables = converse_with_bedrock(sys_prompt, usr_prompt)

    try:
        if selected_tables == '""':
            return GraphState(tables=[], table_names=[])
        
        table_names_list = [name.strip() for name in selected_tables.split(',') if name.strip()]
        tables = [json.loads(content) for content in page_contents 
                 if json.loads(content)['table_name'] in table_names_list]
        table_names = [table['table_name'] for table in tables]
        return GraphState(tables=tables, table_names=table_names)
    except:
        return GraphState(tables=[], table_names=[])

def describe_schema(state: GraphState) -> GraphState:
    """Get detailed schema information for selected tables"""
    table_names = state["table_names"]
    table_details = []
    inspector = inspect(global_object["engine"])
    
    for table_name in table_names:
        columns = inspector.get_columns(table_name)

        create_table_sql = f"CREATE TABLE {table_name} (\n"
        create_table_sql += ",\n".join([f"    {col['name']} {col['type']}" for col in columns])
        create_table_sql += "\n);"

        with global_object["engine"].connect() as connection:
            sample_query = text(f"SELECT * FROM {table_name} LIMIT 5")
            result = connection.execute(sample_query)
            sample_data = [dict(zip(result.keys(), row)) for row in result]
            
        table_desc = get_column_description(table_name)

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

def next_step_by_intent(state: GraphState) -> str:
    """Determine next step based on intent"""
    return state["intent"]

def next_step_by_readiness(state: GraphState) -> str:
    """Determine next step based on readiness"""
    return state["readiness"]

################## SubGraph 2) Text2SQL ##################
def generate_query(state: GraphState) -> GraphState:
    """Generate SQL query based on user question and context"""
    new_query_state = copy.deepcopy(global_object["initial_query_state"])
    question = state["question"]
    sample_queries = state["sample_queries"]
    table_details = state["table_details"]

    query_state = state.get("query_state", {}) or {}
    error_info = query_state.get("error", {}) or {}
    hint = error_info.get("hint", "None")
    
    sys_prompt_template = "You are a skilled database engineer who writes {dialect} SQL queries in response to user questions. Your task is to create accurate SQL queries that match the user's question based on the given database information."
    usr_prompt_template = "Based on the following sample queries, schema information, and past failure history, create a query that matches the DB dialect. Skip the introduction and provide only the generated SQL query statement. \n\n #Question: {question}\n\n #Sample queries:\n {sample_queries}\n\n #Available tables:\n {table_details}\n\n #Additional information (past failure history, additional acquired information, etc.):\n {hint}"    
    
    sys_prompt, usr_prompt = create_prompt(
        sys_prompt_template, 
        usr_prompt_template, 
        question=question, 
        dialect=global_object["DIALECT"], 
        sample_queries=sample_queries, 
        table_details=table_details, 
        hint=hint
    )
    generated_query = converse_with_bedrock(sys_prompt, usr_prompt)

    new_query_state["query"] = generated_query
    return GraphState(query_state=new_query_state)

def validate_query(state: GraphState) -> GraphState:
    """Validate and optimize generated SQL query"""
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
    
    if global_object["DIALECT"].lower() not in explain_statements:
        query_plan = " "
    else:
        try:
            explain_query = explain_statements[global_object["DIALECT"].lower()].format(query=query)
            with global_object["Session"]() as session:
                result = session.execute(text(explain_query))
                query_plan = "\n".join([str(row) for row in result])
        except Exception as e:
            query_state.update({
                "status": "error",
                "error": {
                    "code": "E01",
                    "message": f"An error occurred while executing the EXPLAIN query: {str(e)}",
                    "failed_step": "validation"
                },
                "query": query
            })
            return GraphState(query_state=query_state)

    sys_prompt_template = "You are a database expert who reviews existing {dialect} SQL queries in response to user questions and optimizes them when necessary. Your task is to examine the query's coherence and potential for optimization based on the given SQL query and additional information, and provide a final query based on this analysis." 
    usr_prompt_template = "Please add aliases to the query to match the user's question. It is not allowed to add tables or columns that were not used in the original SQL query. Skip the introduction and provide only the generated SQL query statement. \n\n #Question: {question}\n\n #Existing query:\n {query}\n\n #Query plan:\n {query_plan}"    
    
    sys_prompt, usr_prompt = create_prompt(
        sys_prompt_template, 
        usr_prompt_template, 
        question=question, 
        dialect=global_object["DIALECT"], 
        query=query, 
        query_plan=query_plan
    )
    validated_query = converse_with_bedrock(sys_prompt, usr_prompt)
    query_state["query"] = validated_query

    return GraphState(query_state=query_state)

def execute_query(state: GraphState) -> GraphState:
    """Execute the validated SQL query"""
    query_state = copy.deepcopy(state["query_state"])
    query = query_state["query"]
    try:
        with global_object["Session"]() as session:
            result = session.execute(text(query))
            query_state["result"] = "\n".join([str(row) for row in result])
        return GraphState(query_state=query_state)
    except Exception as e:
        query_state.update({
            "status": "error",
            "error": {
                "code": "E02",
                "message": f"An error occurred while executing the validated query: {str(e)}",
                "failed_step": "execution"
            }
        })
        return GraphState(query_state=query_state)

def handle_failure(state: GraphState) -> GraphState:
    """Handle SQL query failures and determine next steps"""
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

    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, 
        query=query, message=message)
    result = converse_with_bedrock(sys_prompt, usr_prompt)

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
    """Get relevant column suggestions for failed queries"""
    query_state = copy.deepcopy(state["query_state"])
    question = state["question"]
    query = query_state["query"]
    message = query_state['error']['message']
    
    sys_prompt_template = "You are an expert SQL query troubleshooter. Your task is to analyze failed queries and suggest relevant keywords for schema exploration to resolve the issue."
    usr_prompt_template = """Given a user question, a failed SQL query, and an error message, provide 3-5 relevant keywords or phrases for database schema exploration. These should help in finding the correct table and column names to fix the query.\n\n#User question: {question}\n\n#Failed query:\n{query}\n\n#Error message:\n{message}\n\nRespond only with a comma-separated list of keywords or short phrases, without any additional text or explanation.\n\n#Format: {csv_list_response_format}"""
    
    sys_prompt, usr_prompt = create_prompt(
        sys_prompt_template, 
        usr_prompt_template, 
        question=question, 
        query=query, 
        message=message, 
        csv_list_response_format=global_object["csv_list_response_format"]
    )
    keywords = converse_with_bedrock(sys_prompt, usr_prompt)
    return keywords

def next_step_by_query_state(state: GraphState) -> str:
    """Determine next step based on query state"""
    return state["query_state"]["status"]

def next_step_by_next_action(state: GraphState) -> str:
    """Determine next step based on next action"""
    return state["next_action"]

    
################## Answer generation nodes ##################
def get_general_answer(state: GraphState) -> GraphState:
    """Generate a general answer for non-database questions"""
    question = state["question"]
    
    sys_prompt_template = """
    You are a capable assistant who answers general questions from users. 
    If you don't know the answer to a question, admit that you don't know.
    """
    
    usr_prompt_template = "#Question: {question}"
    
    sys_prompt, usr_prompt = create_prompt(
        sys_prompt_template, 
        usr_prompt_template, 
        question=question
    )
    
    answer = converse_with_bedrock(sys_prompt, usr_prompt)
    return GraphState(answer=answer)

def get_database_answer(state: GraphState) -> GraphState:
    """Generate an answer based on database query results"""
    question = state["question"]
    query_state = state["query_state"]
    query = query_state["query"]
    
    sys_prompt_template = """
    You are a competent assistant who answers user questions based on database information. 
    Your task is to provide thorough answers to user questions, referencing the given information.
    """
    
    if query_state["status"] == "success":
        data = query_state["result"]
        usr_prompt_template = """
        The answer should include the used query, dataframe (as a Markdown Table), and a brief response to the question. 

        #Question: {question}

        #Used query: {query}

        #Data: {data}
        """
        
        sys_prompt, usr_prompt = create_prompt(
            sys_prompt_template, 
            usr_prompt_template, 
            question=question, 
            query=query, 
            data=data
        )
    
    else:
        failed_step = query_state["error"]["failed_step"]
        message = query_state["error"]["message"]
        usr_prompt_template = """
        The following is a record of a failed query execution for a user question. 
        Based on this, explain why the request processing failed.

        #Question: {question}

        #Used query: {query}

        #Failed step: {failed_step}

        #Error message: {message}
        """
        
        sys_prompt, usr_prompt = create_prompt(
            sys_prompt_template, 
            usr_prompt_template, 
            question=question, 
            query=query, 
            failed_step=failed_step, 
            message=message
        )
        
    answer = converse_with_bedrock(sys_prompt, usr_prompt)
    return GraphState(answer=answer)

    