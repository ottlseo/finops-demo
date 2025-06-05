import os
from dotenv import load_dotenv

load_dotenv("./.env")

REGION = os.getenv("REGION", "us-west-2")

SONNET = os.getenv("SONNET", "anthropic.claude-3-5-sonnet-20241022-v2:0")
HAIKU = os.getenv("HAIKU", "anthropic.claude-3-haiku-20240307-v1:0")
NOVA_PRO = os.getenv("NOVA_PRO", "amazon.nova-pro-v1:0")

OPENSEARCH_DOMAIN_ENDPOINT = os.getenv("OPENSEARCH_DOMAIN_ENDPOINT", "")
OPENSEARCH_USER_ID = os.getenv("OPENSEARCH_USER_ID", "")
OPENSEARCH_USER_PASSWORD = os.getenv("OPENSEARCH_USER_PASSWORD", "")

TABLE_DESCRIPTION_INDEX = os.getenv("TABLE_DESCRIPTION_INDEX", "schema_description")
EXAMPLE_QUERIES_INDEX = os.getenv("EXAMPLE_QUERIES_INDEX", "sample_queries")

DIALECT = os.getenv("DIALECT", "")
DATABASE_NAME = os.getenv("DATABASE_NAME", "")

if DIALECT == "amazon_athena":    
    ATHENA_REGION = os.getenv("ATHENA_REGION", "us-east-1")
    ATHENA_RESULTS_S3_BUCKET = os.getenv("ATHENA_RESULTS_S3_BUCKET", "")
    DATABASE_PORT = os.getenv("DATABASE_PORT", "443")
    DATABASE_CONNECTION_STRING = f"awsathena+rest://@athena.{ATHENA_REGION}.amazonaws.com:{DATABASE_PORT}/{DATABASE_NAME}?s3_staging_dir={ATHENA_RESULTS_S3_BUCKET}"

elif DIALECT == "postgresql": # redshift
    REDSHIFT_REGION = os.getenv("REDSHIFT_REGION", "us-east-1")
    DATABASE_USERNAME=os.getenv("DATABASE_USERNAME", "admin")
    DATABASE_PASSWORD=os.getenv("DATABASE_PASSWORD", "")
    DATABASE_HOST=os.getenv("DATABASE_HOST", "")
    DATABASE_PORT=os.getenv("DATABASE_PORT", "5439")
    DATABASE_CONNECTION_STRING = f"postgresql+psycopg2://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}?sslmode=require"

else: # other db engines
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
