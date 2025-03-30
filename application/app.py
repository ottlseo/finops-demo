import streamlit as st 
import boto3
import json
import copy
import re
from typing import TypedDict
from botocore.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from pyathena import connect
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
# import src.ddb as ddb
from src.opensearch import OpenSearchVectorRetriever, OpenSearchClient
from src.common_utils import SQLDatabase

st.set_page_config(layout="wide")
st.title("FinOps Text2SQL Demo 💸") 
st.markdown('''- [Github](https://github.com/ottlseo/finops-demo/)에서 코드를 확인하실 수 있습니다.''')

boto_session = boto3.Session()
region_name = "us-west-2" #boto_session.region_name
athena_region_name = "us-east-1"

HAIKU = "us.anthropic.claude-3-haiku-20240307-v1:0" # HAIKU35 = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
SONNET = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
NOVA_PRO = "us.amazon.nova-pro-v1:0"
llm_model = SONNET #NOVA_PRO # TODO: 프롬프트 개선 작업이 필요해서 우선은 Sonnet으로 테스트 진행

ATHENA_URL = f"athena.{athena_region_name}.amazonaws.com" 
ATHENA_DATABASE = 'cur'
ATHENA_RESULTS_S3_BUCKET = 's3://athena-query-result-finops-cost-and-usage/'
athena_connection_string = f"awsathena+rest://@{ATHENA_URL}:443/{ATHENA_DATABASE}?s3_staging_dir={ATHENA_RESULTS_S3_BUCKET}" # /&work_group={athena_wkgrp}"
engine = create_engine(athena_connection_string, echo=True)
db = SQLDatabase(engine)

DIALECT = "amazon_athena"
Session = sessionmaker(bind=engine)

csv_list_response_format = "Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`"
json_response_format = """'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"setup": {"title": "Setup", "description": "question to set up a joke", "type": "string"}, "punchline": {"title": "Punchline", "description": "answer to resolve the joke", "type": "string"}}, "required": ["setup", "punchline"]}\n```'"""

################## graph functions ##################

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
    
def converse_with_bedrock(sys_prompt, usr_prompt):
    temperature = 0.0
    top_p = 0.1
    top_k = 1
    inference_config = {"temperature": temperature, "topP": top_p}
    additional_model_fields = {"top_k": top_k} if llm_model != NOVA_PRO else {}
    response = boto3_client.converse(
        modelId=llm_model, 
        messages=usr_prompt, 
        system=sys_prompt,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )
    return response['output']['message']['content'][0]['text']

def init_boto3_client(region: str):
    retry_config = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "standard"}
    )
    return boto3.client("bedrock-runtime", region_name=region, config=retry_config)

def init_search_resources():  
    EXAMPLE_QUERIES_INDEX = 'example_queries'
    TABLE_DESCRIPTION_INDEX = 'schema_description'

    sql_search_client = OpenSearchClient(region_name=region_name, index_name=EXAMPLE_QUERIES_INDEX, mapping_name='mappings-sql', vector="input_v", text="input", output=["input", "query"])
    table_search_client = OpenSearchClient(region_name=region_name, index_name=TABLE_DESCRIPTION_INDEX, mapping_name='mappings-detailed-schema', vector="table_summary_v", text="table_summary", output=["table_name", "table_summary"])

    sql_retriever = OpenSearchVectorRetriever(sql_search_client, region_name=region_name, k=20)
    table_retriever = OpenSearchVectorRetriever(table_search_client, region_name=region_name, k=10)
    return sql_search_client, table_search_client, sql_retriever, table_retriever

def get_column_description(table_name):
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

def search_by_keywords(keyword):
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

def create_prompt(sys_template, user_template, **kwargs):
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt

################## SubGraph 1) Schema Linking ##################

def analyze_intent(state: GraphState) -> GraphState:
    print(state)
    question = state["question"]
    sys_prompt_template = """당신은 사용자 질문의 의도를 파악하는 비서입니다. 당신의 임무는 사용자 질문을 하나로 분류하는 것입니다. 오직 'database' 또는 'general' 중 하나로만 응답해야 합니다."""
    usr_prompt_template = """주어진 질문이 데이터베이스 조회가 필요한지 판단하세요.
    어떠한 설명이나 이유도 포함하지 말고, 오직 아래 두 단어 중 하나만 답변으로 제공하세요:
    - 데이터베이스 조회가 필요한 경우: database
    - 그 외의 경우: general
    #질문: 
    {question}\n
    응답 (database 또는 general 중 하나만): """
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question)
    intent = converse_with_bedrock(sys_prompt, usr_prompt)

    # 응답 검증 및 정제
    if intent not in ['general', 'database']:
        # 잘못된 응답의 경우 기본값 설정
        print(f"Unexpected response: {intent}. Defaulting to 'general'")
        intent = 'general'
    
    return GraphState(intent=intent)

def get_sample_queries(state: GraphState) -> GraphState: # TODO: Hybrid search 구현
    question = state["question"]
    samples = sql_retriever.vector_search(question)
    page_contents = [json.loads(doc.page_content) for doc in samples]

    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region_name)
    rerank_model_id = "cohere.rerank-v3-5:0"
    model_package_arn = f"arn:aws:bedrock:{region_name}::foundation-model/{rerank_model_id}"

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
    ## TODO: table (view) 선택 과정 추가
    ## TODO: 새로운 노드 추가하기 -> get_relevant_columns: 적절한 column을 vector 검색해 매핑
    
    # sys_prompt_template = "You are a skilled database engineer who writes SQL queries for user questions. Your task is to determine whether it's possible to write an SQL query for the user's question based on the given database information."
    # usr_prompt_template = "Determine if sufficient information has been provided to generate an SQL query for the question. Respond with 'Ready' if there's enough information, or 'Not Ready' if the information is insufficient. \n\n #Question: {question}\n\n #Sample queries:\n {sample_queries}\n\n #Available tables:\n {table_details} \n\n Skip the preamble or explaination. Only provide 'Ready' or 'Not Ready'"    
    
    sys_prompt_template = """당신은 사용자 질문에 대한 SQL 쿼리 작성 가능 여부를 판단하는 시스템입니다. 
    오직 'Ready' 또는 'Not Ready' 중 하나로만 응답해야 합니다.

    지침
    1. 주어진 질문, 샘플 쿼리 및 사용 가능한 테이블을 분석합니다.
    2. 제공된 정보로 SQL 쿼리를 작성할 수 있는지 확인합니다.
    3. 충분한 정보가 있으면 'Ready'으로만 응답하고, 그렇지 않은 경우 'Not Ready'으로만 응답합니다.
    4. 어떠한 설명이나 이유도 포함하지 말고, 오직 'Ready' 또는 'Not Ready' 중 하나만 답변으로 제공하세요.
    """
    usr_prompt_template = """주어진 정보를 바탕으로 SQL 쿼리 생성이 가능한지 판단하세요.\n
    #질문: 
    {question}\n
    #샘플 쿼리:
    {sample_queries}\n
    #사용 가능한 테이블: {table_details}\n
    응답 (Ready 또는 Not Ready 중 하나만):"""
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, sample_queries=sample_queries, table_details=table_details)
    readiness = converse_with_bedrock(sys_prompt, usr_prompt)
    
    # 응답 검증 및 정제
    # if readiness not in ['Ready', 'Not Ready']:
    #     # 잘못된 응답의 경우 기본값 설정
    #     print(f"Unexpected response: {readiness}. Defaulting to 'Not Ready'")
    #     readiness = 'Not Ready'
    
    return GraphState(readiness=readiness)

def get_relevant_tables(state: GraphState) -> GraphState:
    question = state["question"]
    tables = table_retriever.vector_search(question)
    page_contents = [doc.page_content for doc in tables if doc is not None]
    table_inputs = [json.loads(content)['table_summary'] for content in page_contents]

    sys_prompt_template = """당신은 사용자 요청에 맞는 SQL 쿼리를 작성하는 유능한 데이터베이스 엔지니어입니다. 당신의 임무는 SQL 쿼리 작성에 필요한 테이블을 선택하는 것입니다."""
    usr_prompt_template = """사용자 요청에 맞는 SQL 쿼리를 생성하기 위해 필요한 테이블을 선택하여, 이를 중요도 순서로 정렬한 후 인덱스 번호(0부터 시작)로 응답하세요. 요구한 사항 외의 설명을 절대 추가하지 마세요.
    \n\n #질문: {question}\n\n #테이블 정보:\n {table_inputs}\n\n #형식: {csv_list_response_format}""" #사용자 요청에 관련된 테이블이 없으면 빈 목록("")으로 응답하세요.
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, table_inputs=table_inputs, csv_list_response_format=csv_list_response_format)
    selected_tables = converse_with_bedrock(sys_prompt, usr_prompt)

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
    table_names = ['cur.hourly_view_all'] #state["table_names"] #['cur.hourly_view_all', 'cur.summary_view_all'] 
    table_details = []
    inspector = inspect(engine)
    
    for table_name in table_names:
        columns = inspector.get_columns(table_name)

        create_table_sql = f"CREATE TABLE {table_name} (\n"
        create_table_sql += ",\n".join([f"    {col['name']} {col['type']}" for col in columns])
        create_table_sql += "\n);"

        with engine.connect() as connection:
            sample_query = text(f"SELECT * FROM {table_name} LIMIT 5")
            result = connection.execute(sample_query)
            sample_data = [dict(zip(result.keys(), row)) for row in result]
            
        table_desc = get_column_description(table_name) if 'table_search_client' in globals() else {}

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

def describe_schema_from_view(state: GraphState) -> GraphState:
    view_names = ['cur.hourly_view_all']
    table_details = ['cur.hourly_view_all']
    
    with engine.connect() as connection:
        for view_name in view_names:
            try:
                # SHOW COLUMNS 쿼리 실행
                columns_query = f"SHOW COLUMNS FROM {view_name}"
                columns_result = connection.execute(text(columns_query))
                
                # 결과의 실제 구조를 로깅하여 확인
                first_row = columns_result.first()
                if first_row is None:
                    print(f"No columns found for {view_name}")
                    continue
                
                # 컬럼 정보 추출 (에러 처리 추가)
                columns = []
                for row in columns_result:
                    try:
                        # row의 구조를 출력하여 확인
                        print(f"Column row structure: {row}")
                        
                        # 첫 번째 필드를 컬럼 이름으로, 두 번째 필드를 타입으로 가정
                        column_info = {
                            "name": str(row[0]) if row[0] is not None else "unknown",
                            "type": str(row[1]) if len(row) > 1 and row[1] is not None else "unknown"
                        }
                        columns.append(column_info)
                    except Exception as e:
                        print(f"Error processing column row: {e}")
                        continue
                
                # 테이블 정보 생성
                create_view_sql = f"/* View structure for {view_name} */\n"
                create_view_sql += ",\n".join([f"    {col['name']} {col['type']}" for col in columns])
                
                # 샘플 데이터 조회
                try:
                    sample_query = text(f"SELECT * FROM {view_name} LIMIT 5")
                    result = connection.execute(sample_query)
                    sample_data = [dict(zip(result.keys(), row)) for row in result]
                except Exception as e:
                    print(f"Error fetching sample data: {e}")
                    sample_data = []
                
                table_detail = {
                    "table": view_name,
                    "cols": {col['name']: col['type'] for col in columns},
                    "create_table_sql": create_view_sql,
                    "sample_data": str(sample_data) if sample_data else "No sample data available"
                }
                
                table_details.append(table_detail)
                
            except Exception as e:
                print(f"Error describing view {view_name}: {e}")
                continue
                    
    return GraphState(table_details=table_details)

def next_step_by_intent(state: GraphState) -> GraphState:
    return state["intent"]

def next_step_by_readiness(state: GraphState) -> GraphState:
    return state["readiness"]

################## SubGraph 2) Text2SQL ##################

initial_query_state = {
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

def get_valid_service_codes():
    return {
        'AmazonEC2',
        'AmazonRDS', 
        'AWSKMS',
        'AmazonS3',
        'AWSLambda',
        'AmazonDynamoDB',
        # Add other valid service codes as needed
    }

def validate_and_fix_query(query: str, valid_services: set) -> tuple[str, bool, str]:
    # Find all service names in WHERE service = 'X' or service LIKE 'X' patterns
    service_patterns = [
        r"service\s*=\s*'([^']*)'",
        r"service\s*LIKE\s*'([^']*)'",
        r"service\s*IN\s*\(([^)]*)\)"
    ]
    
    found_services = set()
    invalid_services = set()
    
    for pattern in service_patterns:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            if 'IN' in pattern:
                # Handle IN clause separately
                services_in_clause = [s.strip().strip("'") for s in match.group(1).split(',')]
                for service in services_in_clause:
                    if service not in valid_services:
                        invalid_services.add(service)
                    found_services.add(service)
            else:
                service_name = match.group(1)
                if service_name not in valid_services:
                    invalid_services.add(service_name)
                found_services.add(service_name)
    
    if invalid_services:
        error_msg = f"Invalid service names found: {', '.join(invalid_services)}. Valid services are: {', '.join(valid_services)}"
        return query, False, error_msg
    
    return query, True, ""

def generate_query(state: GraphState) -> GraphState:
    print("Current state:", state) 
    dialect = DIALECT
    new_query_state = copy.deepcopy(initial_query_state)
    question = state["question"]
    sample_queries = state["sample_queries"]
    table_details = ['cur.hourly_view_all']
    
    query_state = state.get("query_state", {}) or {}
    error_info = query_state.get("error", {}) or {}
    hint = error_info.get("hint", "None")
    
    # sys_prompt_template = "You are a skilled database engineer who writes {dialect} SQL queries in response to user questions. Your task is to create accurate SQL queries that match the user's question based on the given database information."
    # usr_prompt_template = "Based on the following sample queries, schema information, and past failure history, create a query that matches the DB dialect. Skip the introduction and provide only the generated SQL query statement. \n\n #Question: {question}\n\n #Sample queries:\n {sample_queries}\n\n #Available tables:\n {table_details}\n\n #Additional information (past failure history, additional acquired information, etc.):\n {hint}"    

    sys_prompt_template = """당신은 {dialect} SQL 쿼리를 작성하는 전문 데이터베이스 엔지니어입니다. 
    AWS 서비스 이름을 사용할 때는 정확한 서비스 코드를 사용해야 합니다 (예: 'AmazonEC2', 'AWSKMS').
    오직 SQL 쿼리만을 생성해야 하며, 어떠한 설명이나 추가 텍스트도 포함해서는 안 됩니다."""
    
    usr_prompt_template = """다음 정보를 바탕으로 사용자 질문에 대한 {dialect} SQL 쿼리를 생성하세요. 
    WHERE절에서 service 조건을 사용할 때는 반드시 정확한 AWS 서비스 코드를 사용하세요.
    예시: WHERE service = 'AmazonEC2' (O), WHERE service = 'EC2' (X)
    
    #질문:
    {question}
    
    #샘플 쿼리:
    {sample_queries}
    
    #사용 가능한 테이블:
    {table_details}
    
    #추가 정보:
    {hint}
    
    응답 (SQL 쿼리만):"""
    
    sys_prompt, usr_prompt = create_prompt(
        sys_prompt_template, 
        usr_prompt_template, 
        question=question, 
        dialect=dialect, 
        sample_queries=sample_queries, 
        table_details=table_details, 
        hint=hint
    )
    
    generated_query = converse_with_bedrock(sys_prompt, usr_prompt)
    
    # Validate the generated query
    valid_services = get_valid_service_codes()
    validated_query, is_valid, error_message = validate_and_fix_query(generated_query, valid_services)
    
    if not is_valid:
        new_query_state["status"] = "error"
        new_query_state["error"] = {
            "code": "E03",
            "message": error_message,
            "failed_step": "generation",
            "hint": "Please use valid AWS service codes in the WHERE clause"
        }
    else:
        new_query_state["query"] = validated_query
    
    return GraphState(query_state=new_query_state)

def validate_query(state: GraphState) -> GraphState:
    dialect = DIALECT
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
        'amazon_athena': "EXPLAIN ANALYZE {query}", # == presto
        'sqlserver': "SET STATISTICS PROFILE ON; {query}; SET STATISTICS PROFILE OFF;"
    }
    
    if dialect.lower() not in explain_statements:
        query_plan = " "
    else:
        try:
            explain_query = explain_statements[dialect.lower()].format(query=query)
            with Session() as session:
                result = session.execute(text(explain_query))
                query_plan = "\n".join([str(row) for row in result])
        except Exception as e:
            query_state["status"] = "error"
            query_state["error"]["code"] = "E01"
            query_state["error"]["message"] = f"An error occurred while executing the EXPLAIN query: {str(e)}"
            query_state["error"]["failed_step"] = "validation"
            query_state["query"] = query
            return GraphState(query_state=query_state)

    sys_prompt_template = """당신은 사용자 질문에 대한 {dialect} SQL 쿼리를 검토하고 필요한 경우 최적화하는 데이터베이스 전문가입니다. 
    주어진 SQL 쿼리와 추가 정보를 바탕으로 쿼리의 일관성과 최적화 가능성을 검토하고, 이를 기반으로 최종 쿼리를 제공하는 것이 당신의 임무입니다."""

    usr_prompt_template = """사용자의 질문에 맞게 쿼리에 alias를 추가하세요. 
    원본 SQL 쿼리에서 사용되지 않은 테이블이나 컬럼을 추가하는 것은 허용되지 않습니다.
    서문이나 설명 없이 생성된 SQL 쿼리문만 제공하세요.

    #질문: {question}

    #기존 쿼리:
    {query}

    #쿼리 실행 계획:
    {query_plan}"""        
    # sys_prompt_template = "You are a database expert who reviews existing {dialect} SQL queries in response to user questions and optimizes them when necessary. Your task is to examine the query's coherence and potential for optimization based on the given SQL query and additional information, and provide a final query based on this analysis." 
    # usr_prompt_template = "Please add aliases to the query to match the user's question. It is not allowed to add tables or columns that were not used in the original SQL query. Skip the introduction and provide only the generated SQL query statement. \n\n #Question: {question}\n\n #Existing query:\n {query}\n\n #Query plan:\n {query_plan}"    

    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, dialect=dialect, query=query, query_plan=query_plan)
    validated_query = converse_with_bedrock(sys_prompt, usr_prompt)
    query_state["query"] = validated_query

    return GraphState(query_state=query_state)
 
def execute_query(state: GraphState) -> GraphState:
    query_state = copy.deepcopy(state["query_state"])
    query = query_state["query"]
    try:
        with Session() as session:
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
    sys_prompt_template = """당신은 SQL 쿼리의 문제를 파악하고 트러블슈팅하는 SQL 전문가입니다. 
    당신의 역할은 실패한 SQL 쿼리를 분석하고 문제를 해결하기 위해 다음 단계를 결정하는 것입니다."""

    usr_prompt_template = """
    실패한 SQL 쿼리의 오류 메시지를 바탕으로 다음 중 하나의 실패 원인(`failure_type`)과 해결을 위한 힌트(`hint`)를 제공하세요.\n
    실패 유형 예시:
    - 부정확한 쿼리 문법: `syntax_check`
    - 스키마 불일치 (테이블이나 컬럼이 없음): `schema_check`
    - 외부 DB 요인 (권한, 연결 문제 등): `stop`
    - 일시적인 DB 오류 (쿼리 재실행 필요): `retry`

    #실패한 쿼리: {query}\n
    #오류 메시지: {message}\n
    #응답 형식: \n
    {{
        "failure_type": "<위에서 언급된 실패 유형 중 하나>",
        "hint": "<해결을 위한 간단한 설명이나 제안>"
    }}
    \n
    서문이나 추가 설명 없이 유효한 JSON 문서만 제공하세요."""

    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, query=query, message=message)
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
    query_state = copy.deepcopy(state["query_state"])
    question = state["question"]
    query = query_state["query"]
    message = query_state['error']['message']
    
    sys_prompt_template = "당신은 SQL 쿼리의 문제를 파악하고 트러블슈팅하는 SQL 전문가입니다. 당신의 역할은 실패한 SQL 쿼리를 분석하고 문제를 해결하기 위해 스키마 탐색에 관련된 키워드를 제안하는 것입니다."
    usr_prompt_template = """사용자의 질문과, 실패한 SQL 쿼리, 오류 메시지가 주어지면 데이터베이스 스키마 탐색을 위한 3-5개의 관련 키워드 또는 구문을 제공합니다. 
    이것들은 쿼리를 수정할 올바른 테이블과 열 이름을 찾는 데 도움이 될 것입니다.\n\n
    #사용자의 질문: {question}\n\n
    #실패한 SQL 쿼리: {query}\n\n
    #오류 메시지: {message}\n\n
    추가 텍스트나 설명 없이 쉼표로 구분된 키워드 목록이나 짧은 문구로만 응답하세요.\n\n
    #Format: {csv_list_response_format}""" ## TODO: update prompt
    
    sys_prompt, usr_prompt = create_prompt(
        sys_prompt_template, 
        usr_prompt_template, 
        question=question, 
        query=query, 
        message=message, 
        csv_list_response_format=csv_list_response_format
    )
    
    keywords = converse_with_bedrock(sys_prompt, usr_prompt)
    search_results = search_by_keywords(keywords) if keywords else ""
    
    # query_state를 업데이트하여 검색 결과 포함
    query_state["relevant_columns"] = {
        "keywords": keywords,
        "search_results": search_results
    }
    
    # GraphState의 예상 필드 중 하나인 query_state를 반환
    return GraphState(query_state=query_state)

def next_step_by_query_state(state:GraphState) -> GraphState:
    return state["query_state"]["status"]

def next_step_by_next_action(state:GraphState) -> GraphState:
    return state["next_action"]

################## Answer generation ##################
def get_general_answer(state: GraphState) -> GraphState:
    question = state["question"]
    sys_prompt_template = "사용자의 일반적인 질문에 답하는 유능한 어시스턴트입니다. 질문에 대한 답을 모를 경우, 솔직하게 모른다고 인정하세요. 한국어로 답변하세요."
    usr_prompt_template = "#Question: {question}"
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question)
    answer = converse_with_bedrock(sys_prompt, usr_prompt)

    return GraphState(answer=answer)

def get_database_answer(state: GraphState) -> GraphState:
    question = state["question"]
    query_state = state["query_state"]
    query = query_state["query"]
    data = query_state["result"]
    failed_step = query_state["error"]["failed_step"]
    message = query_state["error"]["message"]
    sys_prompt_template = "당신은 데이터베이스 정보를 기반으로 사용자의 질문에 답변하는 전문 어시스턴트입니다. 주어진 정보를 참조하여 사용자의 질문에 대해 상세하고 정확한 답변을 제공하는 것이 당신의 역할입니다."
    #"You are a competent assistant who answers user questions based on database information. Your task is to provide thorough answers to user questions, referencing the given information."
    
    if query_state["status"] == "success":
        usr_prompt_template = """답변은 사용된 쿼리, 데이터프레임(markdown table 형식), 그리고 질문에 대한 간단한 설명을 포함해야 합니다. 
        만약 쿼리가 정상 실행되었는데 데이터가 비어있다면, 조회된 데이터가 없다고 답하고, 다른 기간이나 조건으로 검색해볼 것을 사용자에게 추천하세요.
        현재 날짜가 언제인지는 신경쓸 필요가 없습니다. 쿼리가 정상 실행되었다면 결과를 데이터 프레임으로 만들어 출력하세요. \n\n
        #질문: {question}\n\n
        #실행된 쿼리: {query}\n\n
        #데이터: {data}\n
        """
        #"The answer should include the used query, dataframe (as a Markdown Table), and a brief response to the question. \n\n#Question: {question}\n\n#Used query: {query}\n\n#Data: {data}\n\n"
        sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query=query, data=data)
    else:
        usr_prompt_template = """다음은 사용자 질문에 대한 쿼리 실행 실패 기록입니다. 이를 바탕으로 요청 처리가 실패한 이유를 설명하세요.\n\n
        #질문: {question}\n\n
        #실행된 쿼리: {query}\n\n
        #실패 단계: {failed_step}\n\n
        #오류 메시지: {message}\n
        """
        #"The following is a record of a failed query execution for a user question. Based on this, explain why the request processing failed.\n\n#Question: {question}\n\n#Used query: {query}\n\n#Failed step: {failed_step}\n\n#Error message: {message}\n\n"
        sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query=query, failed_step=failed_step, message=message)    
        
    answer = converse_with_bedrock(sys_prompt, usr_prompt)
    return GraphState(answer=answer)

def build_langgraph_workflow():
    workflow = StateGraph(GraphState)

    # Global Nodes
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("get_general_answer", get_general_answer)
    workflow.add_node("get_database_answer", get_database_answer)
    workflow.set_entry_point("analyze_intent")

    # SubGraph1 Nodes - Schema Linking
    workflow.add_node("get_sample_queries", get_sample_queries)
    workflow.add_node("check_readiness", check_readiness)
    workflow.add_node("get_relevant_tables", get_relevant_tables)
    workflow.add_node("describe_schema", describe_schema_from_view)

    # SubGraph2 Nodes - Query Generation & Execution
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("handle_failure", handle_failure)
    workflow.add_node("get_relevant_columns", get_relevant_columns)

    # Edge from Entry to SubGraph1
    workflow.add_conditional_edges(
        "analyze_intent",
        next_step_by_intent,
        {
            "database": "get_sample_queries",
            "general": "get_general_answer",
        }
    )

    # Edges in SubGraph1
    workflow.add_edge("get_sample_queries", "check_readiness")
    workflow.add_conditional_edges(
        "check_readiness",
        next_step_by_readiness,
        {
            "Ready": "generate_query",
            "Not Ready": "get_relevant_tables"
        }
    )
    workflow.add_edge("get_relevant_tables", "describe_schema")
    workflow.add_edge("describe_schema", "check_readiness")

    # Edges in SubGraph2
    workflow.add_edge("generate_query", "validate_query")
    workflow.add_conditional_edges(
        "validate_query",
        next_step_by_query_state,
        {
            "success": "execute_query",
            "error": "handle_failure"
        }
    )
    workflow.add_conditional_edges(
        "execute_query",
        next_step_by_query_state,
        {
            "success": "get_database_answer",
            "error": "handle_failure"
        }
    )
    workflow.add_conditional_edges(
        "handle_failure",
        next_step_by_next_action,
        {
            "schema_check": "get_relevant_columns",
            "syntax_check": "generate_query",
            "retry": "validate_query",
            "stop": "get_database_answer"
        }
    )
    workflow.add_edge("get_relevant_columns", "generate_query")

    # Edges to END
    workflow.add_edge("get_general_answer", END)
    workflow.add_edge("get_database_answer", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def print_graph_results(app, query: str):
    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
    inputs = GraphState(question=query)

    # 어시스턴트 응답
    with st.chat_message("assistant"):
        progress_container = st.container() # 진행 상황 컨테이너
        response_container = st.container() # 최종 응답 컨테이너
        
        try:
            current_node = None
            for output in app.stream(inputs, config=config):
                for key, value in output.items():
                    # 새로운 노드 처리 시작
                    if current_node != key:
                        current_node = key
                        with progress_container:
                            if key == "analyze_intent":
                                st.info("🤔 질문을 분석하고 있습니다...")
                            elif key == "get_sample_queries":
                                st.info("🔍 비슷한 쿼리를 찾고 있습니다...")
                            elif key == "generate_query":
                                st.info("⚙️ SQL 쿼리를 생성하고 있습니다...")
                            elif key == "get_relevant_columns":
                                st.info("🔍 관련된 스키마 정보를 탐색하고 있습니다...")
                            elif key == "handle_failure":
                                st.info("✅ 오류를 분석하고 있습니다...")
                            elif key == "execute_query":
                                st.info("🚀 생성한 쿼리를 실행합니다...")
                            elif key == "generate_answer":
                                st.info("📝 응답을 생성하고 있습니다...")
                    
                    # 최종 답변 처리
                    if 'answer' in value:
                        with response_container:
                            st.markdown(value['answer'])
                            # 답변을 세션에 저장
                            st.session_state.messages.append(
                                {"role": "assistant", "content": value['answer']}
                            )
            # 진행 상황 컨테이너 정리
            progress_container.empty()
            
        except GraphRecursionError as e:
            st.error(f"⚠️ I encountered an error: {str(e)}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}
            )

def print_graph_results_with_details(app, query: str):
    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
    inputs = GraphState(question=query)

    with st.chat_message("assistant"):
        progress_container = st.container()
        response_container = st.container()
        
        try:
            current_node = None
            node_results = {}
            previous_states = {}
            
            for output in app.stream(inputs, config=config):
                if isinstance(output, str):
                    output = {"result": {"value": output}}
                elif not isinstance(output, dict):
                    output = {"result": {"value": str(output)}}

                for key, value in output.items():
                    current_state = str(value)
                    
                    if key not in previous_states or previous_states[key] != current_state:
                        previous_states[key] = current_state
                        
                        if current_node != key:
                            current_node = key
                            with progress_container:
                                if key == "analyze_intent":
                                    intent = value.get('intent', {}) if isinstance(value, dict) else str(value)
                                    with st.expander("🤔 질문을 분석하고 있습니다...", expanded=False):  # expanded=False로 변경
                                        st.write(intent)
                                    node_results[key] = ("🤔", "Question Analysis", intent)
                                elif key == "get_sample_queries":
                                    # if isinstance(value, dict) and 'sample_queries' in value:
                                    #     formatted_queries = "\n\n".join([
                                    #         f"Query: {q['query']}\nContext: {q['input']}"
                                    #         for q in value['sample_queries']
                                    #     ])
                                    # else:
                                    #     formatted_queries = str(value)
                                    sample_queries = value.get('sample_queries', {}) if isinstance(value, dict) else str(value)
                                    with st.expander("🔍 비슷한 쿼리를 찾고 있습니다...", expanded=False):
                                        st.write(sample_queries)
                                    node_results[key] = ("🔍", "Similar Queries", sample_queries)
                                elif key == "describe_schema": 
                                    schema_description = value.get('query_state', {}).get('query', '') if isinstance(value, dict) else str(value)
                                    with st.expander("👀 스키마를 분석하고 있습니다...", expanded=False):
                                        st.write(schema_description)
                                    node_results[key] = ("👀", "Describe Schema", schema_description)
                                elif key == "get_relevant_columns":
                                    relevant_schema = value.get('query_state', {}).get('relevant_columns', '') if isinstance(value, dict) else str(value)
                                    with st.expander("🔍 관련된 스키마 정보를 탐색하고 있습니다...", expanded=False):
                                        st.write(relevant_schema)
                                    node_results[key] = ("👀", "Describe Relevant Schema", relevant_schema)
                                elif key == "handle_failure":
                                    error_message = value.get('query_state', {}).get('hint', '') if isinstance(value, dict) else str(value)
                                    with st.expander("✅ 오류를 분석하고 있습니다...", expanded=False):
                                        st.write(error_message)
                                    node_results[key] = ("✅", "Describe Error Message", error_message) 
                                elif key == "generate_query":
                                    query_value = value.get('query_state', {}).get('query', '') if isinstance(value, dict) else str(value)
                                    with st.expander("⚙️ SQL 쿼리를 생성하고 있습니다...", expanded=False):
                                        st.code(str(query_value))
                                    node_results[key] = ("⚙️", "Generated SQL Query", query_value)
                                elif key == "execute_query":
                                    execution_result = value.get('query_state', {}) if isinstance(value, dict) else {"result": str(value)}
                                    with st.expander("🚀 생성한 쿼리를 실행합니다...", expanded=False):
                                        st.write(execution_result)
                                    node_results[key] = ("🚀", "Query Execution Results", execution_result)
                                elif key == "generate_answer":
                                    answer_value = {"answer": value} if isinstance(value, str) else value
                                    with st.expander("📝 응답을 생성하고 있습니다...", expanded=False):
                                        st.write(answer_value)
                                    node_results[key] = ("📝", "Final Response", answer_value)
                    
                    if isinstance(value, dict) and 'answer' in value:
                        with response_container:
                            st.markdown(value['answer'])
                            st.session_state.messages.append(
                                {"role": "assistant", "content": value['answer']}
                            )
            
            # 최종 결과 표시
            with progress_container:
                st.markdown("### 🔍 Execution Details")
                for node, (icon, title, result) in node_results.items():
                    with st.expander(f"{icon} {title}", expanded=False):  # expanded=False로 변경
                        if isinstance(result, dict):
                            for k, v in result.items():
                                st.markdown(f"**{k}:**")
                                st.code(str(v))
                        else:
                            st.code(str(result))
            
        except GraphRecursionError as e:
            st.error(f"⚠️ I encountered an error: {str(e)}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}
            )

################## setting ##################

boto3_client = init_boto3_client(region_name)
sql_search_client, table_search_client, sql_retriever, table_retriever = init_search_resources()
app = build_langgraph_workflow()
# normalizer = ddb.ServiceNameNormalizer() 

################## chatbot ui ##################
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요, 무엇이 궁금하세요?"}
    ]
# 지난 답변 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 유저가 쓴 chat을 query 변수에 담음
query = st.chat_input("Search documentation")
if query:
    # Session에 메세지 저장
    st.session_state.messages.append({"role": "user", "content": query})
    
    # UI에 출력
    st.chat_message("user").write(query)

    # print_graph_results(app, query=query)
    print_graph_results_with_details(app, query=query)

    # Session 메세지 저장
    # st.session_state.messages.append({"role": "assistant", "content": graph_results})
        