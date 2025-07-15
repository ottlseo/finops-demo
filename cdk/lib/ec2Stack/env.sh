#!/bin/bash

# AWS 리전 설정
export AWS_DEFAULT_REGION=us-west-2

# OpenSearch 관련 정보 가져오기
OPENSEARCH_DOMAIN_ENDPOINT=$(aws ssm get-parameter --name opensearch_domain_endpoint --query "Parameter.Value" --output text --region us-west-2 || echo "")
OPENSEARCH_DOMAIN_ENDPOINT="https://$OPENSEARCH_DOMAIN_ENDPOINT"
OPENSEARCH_USER_ID=$(aws ssm get-parameter --name opensearch_user_id --query "Parameter.Value" --output text --region us-west-2 || echo "raguser")
OPENSEARCH_USER_PASSWORD=$(aws secretsmanager get-secret-value --secret-id opensearch_user_password --query "SecretString" --output text --region us-west-2 | jq -r .pwkey || echo "MarsEarth1!")

# .env 파일 생성
cat > /home/ubuntu/finops-demo/application/.env << EOF
REGION=us-west-2

SONNET=anthropic.claude-3-5-sonnet-20241022-v2:0
HAIKU=anthropic.claude-3-haiku-20240307-v1:0
NOVA_PRO=amazon.nova-pro-v1:0

OPENSEARCH_DOMAIN_ENDPOINT=${OPENSEARCH_DOMAIN_ENDPOINT}
OPENSEARCH_USER_ID=${OPENSEARCH_USER_ID}
OPENSEARCH_USER_PASSWORD=${OPENSEARCH_USER_PASSWORD}

TABLE_DESCRIPTION_INDEX=schema_description
EXAMPLE_QUERIES_INDEX=sample_queries

DIALECT=amazon_athena
EOF

# 권한 설정
chown ubuntu:ubuntu /home/ubuntu/finops-demo/application/.env
