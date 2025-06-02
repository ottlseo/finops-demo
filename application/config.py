import os
from dotenv import load_dotenv

load_dotenv("./.env")

REGION=os.getenv("REGION", "us-west-2")

SONNET=os.getenv("SONNET", "anthropic.claude-3-5-sonnet-20241022-v2:0")
HAIKU=os.getenv("HAIKU", "anthropic.claude-3-haiku-20240307-v1:0")
NOVA_PRO=os.getenv("NOVA_PRO", "amazon.nova-pro-v1:0")

OPENSEARCH_DOMAIN_ENDPOINT=os.getenv("OPENSEARCH_DOMAIN_ENDPOINT", "")
OPENSEARCH_USER_ID=os.getenv("OPENSEARCH_USER_ID", "")
OPENSEARCH_USER_PASSWORD=os.getenv("OPENSEARCH_USER_PASSWORD", "")

TABLE_DESCRIPTION_INDEX=os.getenv("TABLE_DESCRIPTION_INDEX", "")
EXAMPLE_QUERIES_INDEX=os.getenv("EXAMPLE_QUERIES_INDEX", "")

ATHENA_CONNECTION_STRING=os.getenv("ATHENA_CONNECTION_STRING", "")
DIALECT=os.getenv("DIALECT", "amazon_athena")

LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST=os.getenv("LANGFUSE_HOST", "")
