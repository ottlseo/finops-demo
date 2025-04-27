import streamlit as st
import boto3
import json
import copy
import re
import base64
import io
import os
import traceback
from typing import TypedDict
from IPython.display import Image, display
from botocore.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from pyathena import connect
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
from src.opensearch import OpenSearchVectorRetriever, OpenSearchClient
from src.common_utils import SQLDatabase
from src.config import *

boto_session = boto3.Session()
engine = create_engine(ATHENA_CONNECTION_STRING, echo=True)
db = SQLDatabase(engine)
Session = sessionmaker(bind=engine)

llm_model = SONNET
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
    sample_questions: list
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
    sql_search_client = OpenSearchClient(region_name=REGION, index_name=EXAMPLE_QUERIES_INDEX, mapping_name='mappings-sql', vector="input_v", text="input", output=["sql", "input", "description"])
    table_search_client = OpenSearchClient(region_name=REGION, index_name=TABLE_DESCRIPTION_INDEX, mapping_name='mappings-detailed-schema', vector="table_summary_v", text="table_summary", output=["table_name", "table_summary"])
    sql_retriever = OpenSearchVectorRetriever(sql_search_client, region_name=REGION, k=20)
    table_retriever = OpenSearchVectorRetriever(table_search_client, region_name=REGION, k=10)
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
                    "size": 2, 
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

def rerank(query, page_contents):
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=REGION)
    model_package_arn = f"arn:aws:bedrock:{REGION}::foundation-model/{RERANK_MODEL}"
    text_sources = [
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {
                    "text": content['description'], 
                }
            }
        } for content in page_contents
    ]

    response = bedrock_agent_runtime.rerank(
                queries=[
                    {
                        "type": "TEXT",
                        "textQuery": {
                            "text": query
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
    return response 

def create_prompt(sys_template, user_template, **kwargs):
    sys_prompt = [{"text": sys_template.format(**kwargs)}]
    usr_prompt = [{"role": "user", "content": [{"text": user_template.format(**kwargs)}]}]
    return sys_prompt, usr_prompt

################## SubGraph 1) Schema Linking ##################

def analyze_intent(state: GraphState) -> GraphState:
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

def get_sample_queries(state: GraphState) -> GraphState: 
    question = state["question"]
    samples = sql_retriever.vector_search(question) 
    page_contents = [json.loads(doc.page_content) for doc in samples]

    response = rerank(question, page_contents)
    reranked_samples = []

    for result in response['results']:
        index = result['index']
        sample_input = page_contents[index]['input']
        sample_sql = page_contents[index]['sql']
        reranked_samples.append({
            'input': sample_input,
            'sql': sample_sql, 
        })
    return GraphState(sample_queries=reranked_samples)

def check_readiness(state: GraphState) -> GraphState:
    question = state["question"]
    sample_queries = state["sample_queries"]
    table_details = state.get("table_details", "") 
    ## TODO: table (view) 선택 과정 추가
    
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
    
    # 응답 검증 및 정제 - TODO: Nova 적용 시
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
    usr_prompt_template = """사용자 요청에 맞는 SQL 쿼리를 생성하기 위해 필요한 테이블을 선택하여, 이를 중요도 순서로 정렬한 후 인덱스 번호(0부터 시작)로 응답하세요. 요구한 사항 외의 설명을 절대 추가하지 마세요.\n\n 
    #질문: {question}\n
    #테이블 정보: {table_inputs}\n
    #형식: {csv_list_response_format}
    """ #사용자 요청에 관련된 테이블이 없으면 빈 목록("")으로 응답하세요.
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
    view_names = ['cur.hourly_view_all']
    table_details = []
    
    for view_name in view_names:
        table_desc = get_column_description(view_name) if 'table_search_client' in globals() else {}
        table_detail = {
            "table": view_name,
            "cols": table_desc if table_desc else []
        }
        table_details.append(table_detail) 
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

def validate_and_fix_service_name(query: str, valid_services: set) -> tuple[str, bool, str]:
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
    # print("Current state:", state) 
    dialect = DIALECT
    new_query_state = copy.deepcopy(initial_query_state)
    question = state["question"]
    sample_queries = state["sample_queries"]
    
    query_state = state.get("query_state", {}) or {}
    table_details = query_state.get("table_details", []) or []
    relavant_columns = query_state.get("relevant_columns", {}).get("search_results", []) or []
    error_info = query_state.get("error", {}) or {}
    hint = error_info.get("hint", "None")

    # sys_prompt_template = "You are a skilled database engineer who writes {dialect} SQL queries in response to user questions. Your task is to create accurate SQL queries that match the user's question based on the given database information."
    # usr_prompt_template = "Based on the following sample queries, schema information, and past failure history, create a query that matches the DB dialect. Skip the introduction and provide only the generated SQL query statement. \n\n #Question: {question}\n\n #Sample queries:\n {sample_queries}\n\n #Available tables:\n {table_details}\n\n #Additional information (past failure history, additional acquired information, etc.):\n {hint}"    

    sys_prompt_template = """당신은 {dialect} SQL 쿼리를 작성하는 전문 데이터베이스 엔지니어입니다. 
    AWS 서비스 이름을 사용할 때는 정확한 서비스 코드를 사용해야 합니다 (예: 'AmazonEC2', 'AWSKMS').
    오직 SQL 쿼리만을 생성해야 하며, 어떠한 설명이나 추가 텍스트도 포함해서는 안 됩니다."""
    
    usr_prompt_template = """다음 정보를 바탕으로 사용자 질문에 대한 {dialect} SQL 쿼리를 생성하세요. \n
    WHERE절에서 service 조건을 사용할 때는 반드시 정확한 AWS 서비스 코드를 사용하세요.\n
    예시: WHERE service = 'AmazonEC2' (O), WHERE service = 'EC2' (X)\n
    #질문: {question}\n
    #샘플 쿼리: {sample_queries}\n
    #사용 가능한 테이블: {table_details}\n
    #추가 정보: {hint}\n
    다음 Column을 확인하고, 적절한 column을 골라 쿼리에 활용하세요: {relavant_columns}\n
    응답 (SQL 쿼리만):"""
    
    sys_prompt, usr_prompt = create_prompt(
        sys_prompt_template, 
        usr_prompt_template, 
        question=question, 
        dialect=dialect, 
        sample_queries=sample_queries, 
        table_details=table_details, 
        relavant_columns=relavant_columns,
        hint=hint
    )
    generated_query = converse_with_bedrock(sys_prompt, usr_prompt)
    
    # Validate the generated query
    valid_services = get_valid_service_codes()
    validated_query, is_valid, error_message = validate_and_fix_service_name(generated_query, valid_services)
    
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
    #질문: {question}\n
    #기존 쿼리: {query}\n
    #쿼리 실행 계획: {query_plan}"""        
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
    usr_prompt_template = """사용자의 질문과, 실패한 SQL 쿼리, 오류 메시지가 주어지면 데이터베이스 스키마 탐색을 위한 3-5개의 관련 키워드 또는 구문을 제공합니다. 이것들은 쿼리를 수정할 올바른 테이블과 열 이름을 찾는 데 도움이 될 것입니다.\n\n
    #사용자의 질문: {question}\n
    #실패한 SQL 쿼리: {query}\n
    #오류 메시지: {message}\n
    추가 텍스트나 설명 없이, 쉼표로 구분된 키워드 목록으로만 응답하세요.\n
    #응답 예시: example_column_name, sample_column, sample_column_key"""
    
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
        #질문: {question}\n
        #실행된 쿼리: {query}\n
        #데이터: {data}\n
        """
        #"The answer should include the used query, dataframe (as a Markdown Table), and a brief response to the question. \n\n#Question: {question}\n\n#Used query: {query}\n\n#Data: {data}\n\n"
        sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query=query, data=data)
    else:
        usr_prompt_template = """다음은 사용자 질문에 대한 쿼리 실행 실패 기록입니다. 이를 바탕으로 요청 처리가 실패한 이유를 설명하세요.\n\n
        #질문: {question}\n
        #실행된 쿼리: {query}\n
        #실패 단계: {failed_step}\n
        #오류 메시지: {message}\n
        """
        #"The following is a record of a failed query execution for a user question. Based on this, explain why the request processing failed.\n\n#Question: {question}\n\n#Used query: {query}\n\n#Failed step: {failed_step}\n\n#Error message: {message}\n\n"
        sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query=query, failed_step=failed_step, message=message)    
        
    answer = converse_with_bedrock(sys_prompt, usr_prompt)

    # query_state["answer"] = answer
    
    return GraphState(answer=answer, query_state=query_state)

def check_text2chart_readiness(state: GraphState) -> GraphState:
    question = state["question"]
    query_state = copy.deepcopy(state["query_state"])
    query_result = query_state["result"]
    
    sys_prompt_template = """당신은 SQL 쿼리 실행 결과로 나온 데이터에 대해 시각화된 Chart 생성 가능 여부를 판단하는 시스템입니다. 
    오직 'Ready' 또는 'Not Ready' 중 하나로만 응답해야 합니다.

    지침
    1. 주어진 질문, SQL 쿼리 실행 결과 데이터를 분석합니다.
    2. 제공된 데이터로 유효한 Chart를 생성할 수 있는지 확인합니다.
    3. 차트를 생성하기에 적합한 데이터라면 'Ready'으로만 응답하고, 그렇지 않은 경우 'Not Ready'으로만 응답합니다. 만약 표시할 데이터의 종류가 1개이거나, 데이터가 충분하지 않은 경우 차트를 만드는 것이 불필요하므로 'Not Ready'로 응답해야 합니다. 
    4. 어떠한 설명이나 이유도 포함하지 말고, 오직 'Ready' 또는 'Not Ready' 중 하나만 답변으로 제공하세요.
    """
    usr_prompt_template = """주어진 정보를 바탕으로 Chart 생성이 가능한지 판단하세요.\n
    #질문: 
    {question}\n
    #데이터베이스 쿼리 결과:
    {query_result}\n
    응답 (Ready 또는 Not Ready 중 하나만):"""
    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question, query_result=query_result)
    readiness = converse_with_bedrock(sys_prompt, usr_prompt)
    print(readiness)
    
    return GraphState(readiness=readiness)

def generate_code_for_chart(state: GraphState) -> GraphState:
    question = state["question"]
    dataset_description = state["answer"]
    query_state = copy.deepcopy(state["query_state"])
    dataset = query_state["result"]
    chart_error = query_state.get("chart_error", "None")

    sys_prompt_template =  '''
                당신은 데이터 분석과 시각화 전문가입니다.
                주어진 structured dataset, dataset의 컬럼 정보, 그리고 사용자의 분석 요청사항을 바탕으로 적절한 차트를 생성하는 Python 코드를 작성하는 것이 당신의 임무입니다.

                <task>
                사용자의 요청에 적합한 차트생성 python 코드 작성
                </task>

                <input>
                1. question: 어떤 분석을 원하는지에 대한 설명
                2. dataset: question의 결과로써, 차트로 시각화할 데이터셋
                3. dataset_description: 결과 데이터에 대한 자세한 설명
                </input>
                
                <output_format>
                JSON 형식으로 다음 형태로 응답하세요. 절대 JSON 포멧 외 텍스트는 넣지 마세요.:
                {{
                    "code": """사용자의 요청을 충족시키는 차트를 생성하는 Python 코드"""
                    "img_path": """생성된 차트의 저장 경로"""
                }}
                </output_format>

                <instruction>
                1. 데이터셋과 컬럼 정보를 신중히 분석하세요.
                2. 사용자의 분석 요청사항을 정확히 이해하세요.
                3. 요청사항에 가장 적합한 차트 유형을 선택하세요 (예: 막대 그래프, 선 그래프, 산점도, 파이 차트 등).
                4. 선택한 차트 유형에 맞는 Python 라이브러리를 사용하세요 (예: matplotlib, seaborn, plotly 등).
                5. 데이터 전처리가 필요한 경우 pandas를 사용하여 데이터를 적절히 가공하세요.
                6. 차트의 제목, 축 레이블, 범례 등을 명확하게 설정하세요.
                7. 필요한 경우 차트의 색상, 스타일, 크기 등을 조정하여 가독성을 높이세요.
                8. 코드에 대한 설명 (주석, "#")은 제외합니다.
                9. 코드 실행 시 발생할 수 있는 예외 상황을 고려하여 적절한 예외 처리를 포함하세요.
                10. 생성된 차트를 저장하거나 표시하는 코드를 포함하세요.
                11. 생성된 코드 수행에 필요한 패키지들은 반드시 import 하세요.
                12. 차트는 모두 영어로 표현해 주세요.
                </instruction>

                <consideration>
                1. 사용자가 제공한 데이터셋의 구조와 크기에 따라 코드를 최적화하세요.
                2. 복잡한 분석 요청의 경우, 단계별로 접근하여 중간 결과를 확인할 수 있도록 코드를 구성하세요.
                3. 데이터의 특성에 따라 적절한 정규화나 스케일링을 고려하세요.
                4. 대규모 데이터셋의 경우 성능을 고려하여 코드를 작성하세요.
                5. "plt.style.use('seaborn')" 코드는 사용하지 마세요.
                6. python의 string code 수행방법(exec())을 사용하려고 합니다. "unterminated string literal" 에러가 발생하지 않게 코드를 작성하세요.\n
                7. 코드가 길어 다음 라인에 연속해서 작성해야 하는 경우, backslash(\)를 사용하여 라인을 연결하세요.
                8. 이 지침을 따라 사용자의 요청에 맞는 정확하고 효과적인 차트 생성 코드를 작성하고, JSON 형식으로 출력하세요.
                9. 차트는 show()함수를 통해 시각화하며, "./output/chart.png"로 저장하고, 경로는 output_format에 맞춰 저장하세요.
                10. 만약 코드 수행에 대한 에러(<error_log>가 주어질 경우, 에러를 고려해서 코드를 수정하세요.
                </consideration>
                '''
    usr_prompt_template = """
                #질문: {question}
                #결과: {dataset}
                #결과 설명: {dataset_description}
                #에러 로그: {error_log}
                Variable `df: pd.DataFrame` is already declared."""

    sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, 
                                           question=question, 
                                           dataset=dataset, 
                                           dataset_description=dataset_description, 
                                           error_log="None" if chart_error == "None" else chart_error
                                           )
    response = converse_with_bedrock(sys_prompt, usr_prompt)
    results = eval(response)
    chart_code = results["code"]
    chart_img_path = results["img_path"]

    query_state["chart_code"] = chart_code
    query_state["chart_img_path"] = chart_img_path
    
    return GraphState(query_state=query_state)

def generate_chart(state: GraphState) -> GraphState:
    query_state = copy.deepcopy(state["query_state"])
    chart_code = query_state["chart_code"]
    chart_img_path = query_state["chart_img_path"]
    print(chart_code)

    try:
        exec(chart_code)
        print(f"Chart is saved to: {chart_img_path}")
        
        # 파일이 존재하는지 확인
        if os.path.exists(chart_img_path):
            with open(chart_img_path, "rb") as image_file:
                img_bytes = image_file.read()
                # img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                # base64_string = base64.b64decode(img_base64)
            image_stream = io.BytesIO(img_bytes)
            # with st.expander("📊 차트를 생성하고 있습니다...", expanded=False):
            st.image(image_stream)

            # query_state에 성공 정보 추가
            query_state["status"] = "success"
            # query_state["chart_img"] = image_stream
            return GraphState(query_state=query_state)
        else:
            pass 
    
    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)
        error_traceback = traceback.format_exc()

        error = f"Error Type: {error_type}\nError Message: {error_message}\n\nTraceback:\n{error_traceback}"
        print(f"error: {error}")
        
        # query_state에 오류 정보 추가
        query_state["status"] = "error"
        query_state["chart_error"] = error
        return GraphState(query_state=query_state)

def generate_followup_questions(state: GraphState) -> GraphState:
    # sys_prompt_template = """당신은 AWS 비용 분석 전문가입니다. 사용자가 CUR 데이터로 더 다양한 비용 인사이트를 얻을 수 있도록, 현재 질문과 비슷한 CUR 관련 질문 3가지를 추천해주세요.
    # 질문은 한국어로 작성되어야 하고, 항상 3개의 배열 형태여야 합니다. EC2 서비스 관련 비용 질문만 생성합니다. 
    # 이유나 부가 설명은 작성하지 말고 출력 형식을 따르세요. 절대 비용과 관련 없는 질문(예: RI 만료 예정일)을 생성하지 마세요.
    # #출력 형식: ["EC2 비용 관련 질문 1", "EC2 비용 관련 질문 2", "EC2 비용 관련 질문 3"]"""
    # usr_prompt_template = "#이전질문: {question}"

    # sys_prompt, usr_prompt = create_prompt(sys_prompt_template, usr_prompt_template, question=question)     
    # response = converse_with_bedrock(sys_prompt, usr_prompt) 
    # sample_questions = json.loads(response)
    question = state["question"]
    samples = state["sample_queries"]
    
    sample_questions = [sample["input"] for sample in samples if sample["input"] != question]
    if len(sample_questions) > 3:
        sample_questions = sample_questions[:3]

    return GraphState(sample_questions=sample_questions)

def build_langgraph_workflow(chart_option=True):
    workflow = StateGraph(GraphState)

    # Global Nodes
    workflow.add_node("analyze_intent", analyze_intent)
    workflow.add_node("get_general_answer", get_general_answer)
    workflow.add_node("get_database_answer", get_database_answer)
    workflow.add_node("generate_followup_questions", generate_followup_questions)
    workflow.set_entry_point("analyze_intent")

    # SubGraph1 Nodes - Schema Linking
    workflow.add_node("get_sample_queries", get_sample_queries)
    workflow.add_node("check_readiness", check_readiness)
    workflow.add_node("get_relevant_tables", get_relevant_tables)
    workflow.add_node("describe_schema", describe_schema)

    # SubGraph2 Nodes - Query Generation & Execution
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("validate_query", validate_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("handle_failure", handle_failure)
    workflow.add_node("get_relevant_columns", get_relevant_columns)

    # SubGraph3 Nodes - Chart Generation
    workflow.add_node("check_text2chart_readiness", check_text2chart_readiness)
    workflow.add_node("generate_code_for_chart", generate_code_for_chart)
    workflow.add_node("generate_chart", generate_chart)

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
    ## Not ready 일 경우에만 스키마 탐색 노드를 거침 
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

    # Edges to chart processing (if enabled) and then to followup questions
    if chart_option:
        workflow.add_edge("get_database_answer", "check_text2chart_readiness")
         
        workflow.add_conditional_edges(
            "check_text2chart_readiness",
            next_step_by_readiness,
            {
                "Ready": "generate_code_for_chart",
                "Not Ready": "generate_followup_questions"
            }
        )
        workflow.add_edge("generate_code_for_chart", "generate_chart")
        workflow.add_conditional_edges(
            "generate_chart",
            next_step_by_query_state,
            {
                "success": "generate_followup_questions",
                "error": "generate_code_for_chart",
            },
        )

    else:
        workflow.add_edge("get_database_answer", "generate_followup_questions")
    
    # General answers always go directly to followup questions
    workflow.add_edge("get_general_answer", "generate_followup_questions")
    
    # Followup questions is always the final step
    workflow.add_edge("generate_followup_questions", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def print_graph_results(app, query: str):
    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
    inputs = GraphState(question=query)

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
                                st.info("🔍 비슷한 질문을 찾고 있습니다...")
                            elif key == "generate_query":
                                st.info("⚙️ 비용 분석을 위한 SQL 쿼리를 생성하고 있습니다...")
                            elif key == "get_relevant_columns":
                                st.info("🔍 관련 데이터를 탐색하고 있습니다...")
                            elif key == "handle_failure":
                                st.info("✅ 오류를 분석하고 있습니다...")
                            elif key == "execute_query":
                                st.info("🚀 분석 결과를 확인하고 있습니다...")
                            elif key == "generate_code_for_chart":
                                st.info("📊 분석 결과를 시각화하고 있습니다...")
                            elif key == "generate_chart":
                                st.info("📝 응답을 생성하고 있습니다...")
                            elif key == "generate_followup_questions":
                                if isinstance(value, dict) and 'sample_questions' in value:
                                    st.session_state.followup_questions = value['sample_questions']
                    
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
                                    with st.expander("🤔 질문을 분석하고 있습니다...", expanded=False): 
                                        st.write(intent)
                                    node_results[key] = ("🤔", "Question Analysis", intent)
                                elif key == "get_sample_queries":
                                    sample_queries = value.get('sample_queries', {}) if isinstance(value, dict) else str(value)
                                    with st.expander("🔍 비슷한 쿼리를 찾고 있습니다...", expanded=False):
                                        if isinstance(sample_queries, list):
                                            for query in sample_queries:
                                                if isinstance(query, dict):
                                                    for k, v in query.items():
                                                        if k == "sql":
                                                            st.code(str(v), language="sql", wrap_lines=True)
                                                        else:
                                                            st.write(v)
                                    node_results[key] = ("🔍", "Similar Queries", sample_queries)
                                elif key == "describe_schema": 
                                    schema_description = value.get('query_state', {}).get('table_details', []) if isinstance(value, dict) else str(value)
                                    with st.expander("👀 스키마를 분석하고 있습니다...", expanded=False):
                                        st.write(schema_description)
                                    node_results[key] = ("👀", "Describe Schema", schema_description)
                                elif key == "get_relevant_columns":
                                    relevant_schema = value.get('query_state', {}).get('relevant_columns', '') if isinstance(value, dict) else str(value)
                                    with st.expander("🔍 관련된 스키마 구조를 탐색하고 있습니다...", expanded=False):
                                        if isinstance(relevant_schema, dict):
                                            for k, v in relevant_schema.items():
                                                st.markdown(f"**{k}:**")
                                                st.code(str(v), wrap_lines=True)
                                    node_results[key] = ("👀", "Describe Relevant Schema", relevant_schema)
                                elif key == "handle_failure":
                                    error_message = value.get('query_state', {}).get('hint', '') if isinstance(value, dict) else str(value)
                                    with st.expander("✅ 오류를 분석하고 있습니다...", expanded=False):
                                        st.write(error_message)
                                    node_results[key] = ("✅", "Describe Error Message", error_message) 
                                elif key == "generate_query":
                                    query_value = value.get('query_state', {}).get('query', '') if isinstance(value, dict) else str(value)
                                    with st.expander("⚙️ SQL 쿼리를 생성하고 있습니다...", expanded=False):
                                        st.code(str(query_value), language="sql", wrap_lines=True)
                                    node_results[key] = ("⚙️", "Generated SQL Query", query_value)
                                elif key == "execute_query":
                                    execution_result = value.get('query_state', {}) if isinstance(value, dict) else {"result": str(value)}
                                    with st.expander("🚀 생성한 쿼리를 실행합니다...", expanded=False):
                                        if isinstance(execution_result, dict):
                                            for k, v in execution_result.items():
                                                st.markdown(f"**{k}:**")
                                                st.code(str(v), wrap_lines=True)
                                    node_results[key] = ("🚀", "Query Execution Results", execution_result)
                                elif key == "generate_answer":
                                    answer_value = {"answer": value} if isinstance(value, str) else value
                                    with st.expander("📝 응답을 생성하고 있습니다...", expanded=False):
                                        if isinstance(answer_value, dict):
                                            for k, v in answer_value.items():
                                                st.markdown(f"**{k}:**")
                                                st.code(str(v), wrap_lines=True)
                                    node_results[key] = ("📝", "Final Response", answer_value)
                                elif key == "generate_code_for_chart":
                                    chart_code = value.get('query_state', {}).get('chart_code', '') if isinstance(value, dict) else str(value)
                                    with st.expander("📊 결과를 시각화하고 있습니다...", expanded=False):
                                        st.code(str(chart_code), language="python", wrap_lines=True)
                                    node_results[key] = ("📊", "Chart Generation Code", chart_code)
                                elif key == "generate_chart":
                                    chart_img_path = value.get('query_state', {}).get('chart_img_path', '') if isinstance(value, dict) else str(value)
                                    with st.expander("✍️ 차트를 생성합니다..."):
                                        st.image(chart_img_path)
                                    node_results[key] = ("✍️", "Generated Chart", chart_img_path)
                                elif key == "generate_followup_questions":
                                    if isinstance(value, dict) and 'sample_questions' in value:
                                        st.session_state.followup_questions = value['sample_questions']
            
                    with response_container:
                        if isinstance(value, dict) and 'answer' in value: #if isinstance(value, dict) and 'sample_questions' in value:
                            st.markdown(value['answer'])
                            st.session_state.messages.append(
                                {"role": "assistant", "content": value['answer']}
                            )
                        if isinstance(value, dict) and 'chart_img_path' in value:
                            img_path = value['chart_img_path']
                            print(value['chart_img_path'])
                            st.image(img_path, use_column_width=True)
                            # TODO: session_state에 img_bytes 저장

        except GraphRecursionError as e:
            st.error(f"⚠️ I encountered an error: {str(e)}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}
            )

def handle_query(query):
    st.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query})    
    if show_log:
        print_graph_results_with_details(app, query=query)
    else:
        print_graph_results(app, query=query)

################## setting ##################

boto3_client = init_boto3_client(REGION)
sql_search_client, table_search_client, sql_retriever, table_retriever = init_search_resources()

initial_questions = [
    "온디맨드와 예약 인스턴스 사용량을 인스턴스 타입별로 비교해주세요.",
    "이번 달에 새로 생성된 EC2 인스턴스의 생성 날짜와 타입, 그리고 비용을 보여주세요.",
    "775638497521 어카운트 리소스 중에 SP 적용이 가장 시급한 인스턴스 패밀리를 알려주세요."
]
if "followup_questions" not in st.session_state:
    st.session_state["followup_questions"] = initial_questions

if "text2chart" not in st.session_state:
    st.session_state["text2chart"] = True

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요, FinOps 챗봇입니다. AWS 비용과 관련해 궁금하신 점은 무엇이든 물어보세요!"}
    ]


app = build_langgraph_workflow(chart_option=st.session_state["text2chart"])

################## chatbot ui ##################

st.set_page_config(layout="wide")
st.title("FinOps - AWS Cost and Usage Report Analysis") 

col1, col2 = st.columns(2)
with col1:
    st.caption('''[Github](https://github.com/ottlseo/finops-demo/)에서 코드를 확인하실 수 있습니다.''')
with col2: 
    show_log = st.toggle("Text2SQL 로그 확인하기", value=True)
    st.session_state["text2chart"] = st.toggle("분석 내용을 Chart로 시각화하기", value=True)

# 채팅 히스토리 표시
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 새로운 쿼리가 입력되면 followup_questions 제거
query = st.chat_input("질문을 입력하세요.")
if query:
    if "followup_questions" in st.session_state:
        del st.session_state.followup_questions

# session_state에 저장된 query 처리
if "query" in st.session_state:
    query = st.session_state.query
    del st.session_state.query
    handle_query(query)
elif query:
    handle_query(query)

# 마지막 메시지가 assistant의 응답일 때만 버튼 표시
if (st.session_state.messages and 
    st.session_state.messages[-1]["role"] == "assistant" and 
    "followup_questions" in st.session_state):

    with st.container(border=True):
        st.markdown("##### 💡 이런 질문은 어떠세요?")
        for idx, button_text in enumerate(st.session_state.followup_questions):        
            if st.button(button_text, key=f"btn_{idx}"):
                st.session_state.query = button_text
                if "followup_questions" in st.session_state:
                    del st.session_state.followup_questions
                st.rerun()
