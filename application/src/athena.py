import os
import json
import yaml
import boto3
from botocore.config import Config
import streamlit as st
from dotenv import load_dotenv
from .ssm import parameter_store
import time
from datetime import datetime

from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine


class AthenaClient:
    def __init__(self, region_name, db_name, result_s3_dir):
        pm = parameter_store('us-west-2')
        # credentials = boto3.Session().get_credentials()
        self.conn = boto3.client(
            "athena",
            region_name=region_name
        )
        self.db_name = db_name # "text2sql"
        self.result_s3_dir = result_s3_dir # "s3://text2sql-test-ottlseo/results/"

        self.EXAMPLE_QUERY_1 = "SHOW TABLES"
        self.EXAMPLE_QUERY_2 = "SELECT * FROM employee LIMIT 5"
        self.EXAMPLE_QUERY_3 = """
        SELECT c.customerid, c.firstname, c.lastname, SUM(i.total) AS total_spent
        FROM customer c
        JOIN invoice i ON c.customerid = i.customerid
        GROUP BY c.customerid, c.firstname, c.lastname
        ORDER BY total_spent DESC
        LIMIT 10
        """

    def wait_for_query_completion(self, query_execution_id): # 쿼리 실행이 완료될 때까지 대기
        athena_client = self.conn
        while True:
            response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            state = response['QueryExecution']['Status']['State']
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                return state
            print(f"Waiting for query to complete. Current state: {state}")
            time.sleep(1)  # 1초 대기

    def get_query_results(self, query_execution_id): # 쿼리 결과 가져오기
        athena_client = self.conn
        response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        columns = [col['Label'] for col in response['ResultSet']['ResultSetMetadata']['ColumnInfo']] # 컬럼 이름 추출
        
        # 결과 데이터 추출
        results = []
        for row in response['ResultSet']['Rows'][1:]: # 첫 번째 행은 헤더이므로 건너뜀
            values = [field.get('VarCharValue', '') for field in row['Data']]
            results.append(dict(zip(columns, values)))
        return results

    def query(self, query):
        try:
            # 쿼리 실행
            query_response = self.conn.start_query_execution(
                QueryString=query,
                QueryExecutionContext={"Database": self.db_name},
                ResultConfiguration={
                    "OutputLocation": self.result_s3_dir,
                    "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
                },
            )
            
            query_execution_id = query_response['QueryExecutionId']
            state = self.wait_for_query_completion(query_execution_id)

            if state == 'SUCCEEDED':
                # 성공적인 결과 반환
                return self.get_query_results(query_execution_id)
            else:
                # 실패 정보 가져오기
                failure_info = self.conn.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']
                
                error_message = failure_info.get('StateChangeReason', 'Unknown error')
                
                # 상세한 오류 정보 반환
                return {
                    "status": "error",
                    "error": {
                        "code": f"ATHENA_{state}",
                        "message": error_message,
                        "failed_step": "execution",
                        "athena_error": failure_info,
                        "query_execution_id": query_execution_id,
                        "state": state
                    }
                }
                
        except Exception as e:
            # Athena API 호출 자체의 오류 처리
            return {
                "status": "error",
                "error": {
                    "code": "ATHENA_API_ERROR",
                    "message": str(e),
                    "failed_step": "execution",
                    "athena_error": {
                        "exception_type": type(e).__name__,
                        "details": str(e)
                    }
                }
            }
        
    # def query(self, query):
    #     athena_client = self.conn
    #     # 쿼리 실행
    #     query_response = athena_client.start_query_execution(
    #         QueryString=query,
    #         QueryExecutionContext={"Database": self.db_name},
    #         ResultConfiguration={
    #             "OutputLocation": self.result_s3_dir,
    #             "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
    #         },
    #     )
    #     query_execution_id = query_response['QueryExecutionId']
    #     state = self.wait_for_query_completion(query_execution_id) # 쿼리 완료 대기

    #     if state == 'SUCCEEDED': # 쿼리 완료 시, 결과 가져오기
    #         response = self.get_query_results(query_execution_id)
    #         # print(response)
    #         return response
    #     else:
    #         # 실패 정보 가져오기
    #         failure_info = athena_client.get_query_execution(
    #             QueryExecutionId=query_execution_id
    #             )['QueryExecution']['Status']
    #         error_message = failure_info.get('StateChangeReason', 'Unknown error')
    #         return {
    #             "status": "error",
    #             "error": {
    #                 "code": "ATHENA_" + state,  # ATHENA_FAILED, ATHENA_CANCELLED 등
    #                 "message": f"Athena query execution failed: {error_message}",
    #                 "failed_step": "execution",
    #                 "athena_error": failure_info  # 상세 에러 정보 저장
    #             }
    #         }
            