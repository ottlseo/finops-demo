# AWS FinOps Assistant: AWS Cost and Usage Reports Analysis

AWS FinOps Assistant provides a Text2SQL & Text2Chart chatbot interface that allows for easy analysis of AWS Cost and Usage Report (CUR) data through natural language questions. It makes complex AWS billing data queryable with simple natural language questions, making cost analysis more accessible and efficient.

AWS FinOps 어시스턴트는 AWS 비용 및 사용 보고서(CUR) 데이터를 자연어 질문으로 쉽게 분석할 수 있도록 Text2SQL 챗봇 인터페이스를 제공합니다. 복잡한 AWS 청구 데이터를 간단한 자연어 질문으로 조회할 수 있게 해주어, 비용 분석을 더욱 접근하기 쉽고 효율적으로 만듭니다.

![finops-demo](https://github.com/user-attachments/assets/1ffeefa1-c668-4ccd-8324-9bf1bc3bf534)

# How to build
### Prerequisites
- Install Python 3.8 or higher
- Request access to Amazon Bedrock 
- AWS CLI configured with appropriate credentials

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ottlseo/finops-demo.git
```

2. Run all notebooks in [`/data_preparation`](./data_preparation) directory to create indexes to Amazon OpenSearch Services
   - [0.setup_opensearch.ipynb](./data_preparation/0.setup_opensearch.ipynb)
   - [1.generate_sample_queries.ipynb](./data_preparation/1.generate_sample_queries.ipynb)
   - [2.generate_schema_description.ipynb](./data_preparation/2.generate_schema_description.ipynb)

3. Install dependencies:
```bash
cd application
pip install -r requirements.txt
```

4. Create `.env` file (under `./application` directory)
```env
REGION=us-west-2

SONNET=anthropic.claude-3-5-sonnet-20241022-v2:0
HAIKU=anthropic.claude-3-haiku-20240307-v1:0
NOVA_PRO=amazon.nova-pro-v1:0

OPENSEARCH_DOMAIN_ENDPOINT=[ENTER_YOUR_OPENSEARCH_ENDPOINT]
OPENSEARCH_USER_ID=[ENTER_YOUR_OPENSEARCH_USERID] # raguser
OPENSEARCH_USER_PASSWORD=[ENTER_YOUR_OPENSEARCH_PW] # MarsEarth1!

TABLE_DESCRIPTION_INDEX=schema_description
EXAMPLE_QUERIES_INDEX=sample_queries

DIALECT=amazon_athena
ATHENA_REGION=us-east-1
ATHENA_RESULTS_S3_BUCKET=[ENTER_YOUR_S3_BUCKET_NAME] # s3://example-bucket-name/
DATABASE_NAME=[ENTER_YOUR_DB_NAME_TO_QUERY]

# Optional - if you want to use another DB engine to query instead of Athena
DATABASE_CONNECTION_STRING=[ENTER_YOUR_DB_CONNECTION_URL]

# Optional - If you want to use Langfuse for debugging and logging
LANGFUSE_PUBLIC_KEY=[ENTER_YOUR_LANGFUSE_PUBLIC_KEY]
LANGFUSE_SECRET_KEY=[ENTER_YOUR_LANGFUSE_SECRET_KEY]
LANGFUSE_HOST=[ENTER_YOUR_LANGFUSE_HOST_URL]
```

5. Start the Streamlit application:
```bash
streamlit run app.py
```

# Example queries
- 지난 1년 간의 'AmazonEC2' 서비스의 월별 사용 비용을 조회해줘.
- 각 리전별로 'AmazonEC2' 서비스의 월별 비용을 확인하고 싶습니다. 최근 날짜부터 리전별로 정리해서 보여주세요.
- 2024년 7월 한 달 동안 비용을 가장 많이 사용한 AmazonEC2 인스턴스 타입 5가지를 알려주세요. 비용이 높은 순서대로 알려주세요.
- To be updated

# Repository Structure
```
.
├── application/                 # 메인 애플리케이션 directory
│   ├── app.py
│   ├── graph.py
│   ├── config.py
│   ├── .env
│   ├── requirements.txt        
│   └── lib/                    # 애플리케이션 소스 코드 (하위 구조는 생략)
│
└── data_preparation/          # Text2SQL 작업에 필요한 데이터 증강 및 OpenSearch 인덱싱을 위한 코드 directory
    ├── input_data/           # Source schema definitions and example queries
    │   ├── cur_example_queries.sql    # Sample CUR analysis queries
    │   ├── cur_schema.json            # Base CUR schema definition
    │   └── enhanced_cur_schema.json   # Extended CUR schema with detailed descriptions
    ├── libs/                 # Shared libraries for data preparation
    └── output_data/         # Processed schema files
```
