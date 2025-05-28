import boto3
import json
import os
import yaml
import time
import streamlit as st
from typing import Dict, List, Any, Optional, NamedTuple
from opensearchpy import OpenSearch, RequestsHttpConnection
import sys

# 상위 디렉토리를 경로에 추가하여 config.py를 직접 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class Document(NamedTuple):
    page_content: str
    metadata: dict

class OpenSearchClient:
    def __init__(self, region_name: str, index_name: str, mapping_name: str, vector: str, text: str, output: List[str]):
        self.region_name = region_name
        self.index_name = index_name
        self.mapping_name = mapping_name
        self.vector = vector
        self.text = text
        self.output = output
        
        # OpenSearch 연결 설정
        auth = (OPENSEARCH_USER_ID, OPENSEARCH_USER_PASSWORD)
        host = OPENSEARCH_DOMAIN_ENDPOINT.replace("https://", "").split(':')[0]
        
        self.conn = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20
        )
        
        # 설정 로드 (필요한 경우)
        self.config = self.load_opensearch_config()
        if self.config:
            self.mapping = {"settings": self.config['settings'], "mappings": self.config[mapping_name]}
    
    def load_opensearch_config(self):
        """OpenSearch 설정 파일 로드"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "opensearch.yml")
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                return config
            return None
        except Exception as e:
            print(f"Error loading OpenSearch config: {e}")
            return None
    
    def create_index(self):
        """인덱스 생성 또는 재생성"""
        if not self.mapping:
            print(f"No mapping defined for index {self.index_name}")
            return
            
        if not self.conn.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} does not exist. Creating now...")
        else:
            self.conn.indices.delete(index=self.index_name)
            print(f"Existing index '{self.index_name}' has been deleted. Create new one.")
            time.sleep(2)
        self.conn.indices.create(self.index_name, body=self.mapping)
    
    def search(self, query: str, k: int = 10):
        """기본 검색 기능 (필요시 구현)"""
        pass

class OpenSearchVectorRetriever:
    def __init__(self, client: OpenSearchClient, region_name: str, k: int = 10):
        self.client = client
        self.region_name = region_name
        self.k = k
        self.emb_model = "amazon.titan-embed-text-v2:0"
    
    def _embedding(self, input_text):
        """텍스트를 임베딩 벡터로 변환"""
        boto3_client = boto3.client("bedrock-runtime", region_name=self.region_name)
        response = boto3_client.invoke_model(
                modelId=self.emb_model,
                body=json.dumps({"inputText": input_text})
            )
        return json.loads(response['body'].read())['embedding']
    
    def vector_search(self, query: str):
        """벡터 검색 수행"""
        embedding = self._embedding(query)
        semantic_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                self.client.vector: {
                                    "vector": embedding,
                                    "k": self.k,
                                }
                            }
                        },
                    ]
                }
            },
            "size": self.k
        }

        result = self.client.conn.search(index=self.client.index_name, body=semantic_query)
        documents = []
        for hit in result['hits']['hits']:
            source = hit['_source']
            page_content = {k: source[k] for k in self.client.output if k in source}
            documents.append({"page_content": json.dumps(page_content), "metadata": {}})

        return documents

def init_search_resources(region_name: str, example_queries_index: str, table_description_index: str):
    """OpenSearch 클라이언트와 리트리버 초기화"""
    sql_search_client = OpenSearchClient(
        region_name=region_name, 
        index_name=example_queries_index, 
        mapping_name='mappings-sql', 
        vector="input_v", 
        text="input", 
        output=["sql", "input", "description"]
    )
    
    table_search_client = OpenSearchClient(
        region_name=region_name, 
        index_name=table_description_index, 
        mapping_name='mappings-detailed-schema', 
        vector="table_summary_v", 
        text="table_summary", 
        output=["table_name", "table_summary"]
    )
    
    sql_retriever = OpenSearchVectorRetriever(sql_search_client, region_name=region_name, k=20)
    table_retriever = OpenSearchVectorRetriever(table_search_client, region_name=region_name, k=10)
    
    return sql_search_client, table_search_client, sql_retriever, table_retriever

def initialize_os_client(enable_flag, client_params, indexing_function, lang_config):
    """UI에서 사용할 OpenSearch 클라이언트 초기화"""
    if enable_flag:
        client = OpenSearchClient(**client_params)
        indexing_function(client, lang_config)
    else:
        client = ""
    return client


def sample_query_indexing(os_client, lang_config):
    rag_query_file = st.text_input(lang_config['rag_query_file'], value="./db_metadata/chinook_example_queries.jsonl")
    if not os.path.exists(rag_query_file):
        st.warning(lang_config['file_not_found'])
        return

    if st.sidebar.button(lang_config['process_file'], key='query_file_process'):
        with st.spinner("Now processing..."):
            os_client.delete_index()
            os_client.create_index() 

            with open(rag_query_file, 'r') as file:
                bulk_data = file.read()

            response = os_client.conn.bulk(body=bulk_data)
            if response["errors"]:
                st.error("Failed")
            else:
                st.success("Success")


def schema_desc_indexing(os_client, lang_config):
    schema_file = st.text_input(lang_config['schema_file'], value="./db_metadata/chinook_detailed_schema.json")
    if not os.path.exists(schema_file):
        st.warning(lang_config['file_not_found'])
        return

    if st.sidebar.button(lang_config['process_file'], key='schema_file_process'):
        with st.spinner("Now processing..."):
            os_client.delete_index()
            os_client.create_index() 

            with open(schema_file, 'r', encoding='utf-8') as file:
                schema_data = json.load(file)

            bulk_data = []
            for table in schema_data:
                for table_name, table_info in table.items():
                    table_doc = {
                        "table_name": table_name,
                        "table_desc": table_info["table_desc"],
                        "columns": [{"col_name": col["col"], "col_desc": col["col_desc"]} for col in table_info["cols"]],
                        "table_summary": table_info["table_summary"],
                        "table_summary_v": table_info["table_summary_v"]
                    }
                    bulk_data.append({"index": {"_index": os_client.index_name, "_id": table_name}})
                    bulk_data.append(table_doc)
            
            bulk_data_str = '\n'.join(json.dumps(item) for item in bulk_data) + '\n'

            response = os_client.conn.bulk(body=bulk_data_str)
            if response["errors"]:
                st.error("Failed")
            else:
                st.success("Success")


def init_opensearch(region_name, lang_config):
    """Streamlit UI에서 사용할 OpenSearch 초기화"""
    with st.sidebar:
        enable_rag_query = st.sidebar.checkbox(lang_config['rag_query'], value=True, disabled=True)
        sql_os_client = initialize_os_client(
            enable_rag_query,
            {
                "region_name": region_name,
                "index_name": 'example_queries',
                "mapping_name": 'mappings-sql',
                "vector": "input_v",
                "text": "input",
                "output": ["input", "query"]
            },
            sample_query_indexing,
            lang_config
        )

        enable_schema_desc = st.sidebar.checkbox(lang_config['schema_desc'], value=True, disabled=True)
        schema_os_client = initialize_os_client(
            enable_schema_desc,
            {
                "region_name": region_name,
                "index_name": 'schema_descriptions',
                "mapping_name": 'mappings-detailed-schema',
                "vector": "table_summary_v",
                "text": "table_summary",
                "output": ["table_name", "table_summary"]
            },
            schema_desc_indexing,
            lang_config
        )

    return sql_os_client, schema_os_client

def get_column_description(table_search_client: OpenSearchClient, table_name: str) -> Dict[str, str]:
    """테이블의 컬럼 설명 가져오기"""
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

def search_by_keywords(table_search_client: OpenSearchClient, keyword: str) -> str:
    """키워드로 컬럼 검색"""
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
