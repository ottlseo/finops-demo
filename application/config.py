import os
import boto3
import json
from dotenv import load_dotenv

load_dotenv("./.env")

def get_ssm_parameter(parameter_name, region='us-west-2', default_value=None):
    """SSM Parameter Store에서 파라미터 값을 가져오는 함수"""
    try:
        ssm_client = boto3.client('ssm', region_name=region)
        response = ssm_client.get_parameter(Name=parameter_name)
        return response['Parameter']['Value']
    except Exception as e:
        print(f"SSM 파라미터 '{parameter_name}'가 조회되지 않아 기본값인 '{default_value}' 사용: {e}")
        return default_value

def get_secret(secret_name, region='us-west-2', default_value=None):
    """AWS Secrets Manager에서 시크릿 값을 가져오는 함수"""
    try:
        secrets_client = boto3.client('secretsmanager', region_name=region)
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret_string = response['SecretString']
        
        # JSON 형태인지 확인하고 파싱
        try:
            secret_dict = json.loads(secret_string)
            # OpenSearch 비밀번호의 경우 pwkey 값을 반환
            if 'pwkey' in secret_dict:
                return secret_dict['pwkey']
            else:
                return secret_string
        except json.JSONDecodeError:
            return secret_string
            
    except Exception as e:
        print(f"Secrets Manager 시크릿 '{secret_name}'가 조회되지 않아 기본값인 '{default_value}' 사용: {e}")
        return default_value

REGION = os.getenv("REGION", "us-west-2")

SONNET = os.getenv("SONNET", "anthropic.claude-3-5-sonnet-20241022-v2:0")
HAIKU = os.getenv("HAIKU", "anthropic.claude-3-haiku-20240307-v1:0")
NOVA_PRO = os.getenv("NOVA_PRO", "amazon.nova-pro-v1:0")

# 환경 변수를 우선적으로 사용하고, 없을 경우에만 SSM과 Secrets Manager에서 값을 가져옴
OPENSEARCH_DOMAIN_ENDPOINT = os.getenv("OPENSEARCH_DOMAIN_ENDPOINT") or get_ssm_parameter("opensearch_domain_endpoint", default_value="")
OPENSEARCH_USER_ID = os.getenv("OPENSEARCH_USER_ID") or get_ssm_parameter("opensearch_user_id", default_value="raguser")
OPENSEARCH_USER_PASSWORD = os.getenv("OPENSEARCH_USER_PASSWORD") or get_secret("opensearch_user_password", default_value="MarsEarth1!")

TABLE_DESCRIPTION_INDEX = os.getenv("TABLE_DESCRIPTION_INDEX", "schema_description")
EXAMPLE_QUERIES_INDEX = os.getenv("EXAMPLE_QUERIES_INDEX", "sample_queries")

DIALECT = os.getenv("DIALECT", "")

if DIALECT == "amazon_athena":    
    DATABASE_NAME = get_ssm_parameter("database_name", default_value="cur") # os.getenv("DATABASE_NAME", "")
    ATHENA_REGION = get_ssm_parameter("athena_region", default_value="us-east-1") # os.getenv("ATHENA_REGION", "")
    ATHENA_RESULTS_S3_BUCKET = get_ssm_parameter("athena_results_s3_bucket", default_value="") # os.getenv("ATHENA_RESULTS_S3_BUCKET", "")
    DATABASE_PORT = get_ssm_parameter("database_port", default_value="443") # os.getenv("DATABASE_PORT", "443")
    DATABASE_CONNECTION_STRING = f"awsathena+rest://@athena.{ATHENA_REGION}.amazonaws.com:{DATABASE_PORT}/{DATABASE_NAME}?s3_staging_dir={ATHENA_RESULTS_S3_BUCKET}"

elif DIALECT == "postgresql": # redshift
    DATABASE_NAME = os.getenv("DATABASE_NAME", "")
    REDSHIFT_REGION = os.getenv("REDSHIFT_REGION", "us-east-1")
    DATABASE_USERNAME=os.getenv("DATABASE_USERNAME", "admin")
    DATABASE_PASSWORD=os.getenv("DATABASE_PASSWORD", "")
    DATABASE_HOST=os.getenv("DATABASE_HOST", "")
    DATABASE_PORT=os.getenv("DATABASE_PORT", "5439")
    DATABASE_CONNECTION_STRING = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}?sslmode=require"

else: # other db engines
    DATABASE_NAME = os.getenv("DATABASE_NAME", "")
    DATABASE_USERNAME=os.getenv("DATABASE_USERNAME", "")
    DATABASE_PASSWORD=os.getenv("DATABASE_PASSWORD", "")
    DATABASE_HOST=os.getenv("DATABASE_HOST", "")
    DATABASE_PORT=os.getenv("DATABASE_PORT", "")
    
    if DIALECT == "oracle":
        DATABASE_CONNECTION_STRING=f"oracle://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/?service_name={DATABASE_NAME}"
    elif DIALECT == "mysql":
        DATABASE_CONNECTION_STRING=f"mysql+pymysql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    else: 
        DATABASE_CONNECTION_STRING=f"{DIALECT}://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST=os.getenv("LANGFUSE_HOST", "")
