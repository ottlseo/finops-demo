# AWS FinOps Assistant: AWS Cost and Usage Reports with Text2SQL

AWS FinOps 어시스턴트는 AWS 비용 및 사용 보고서(CUR) 데이터를 자연어 질문으로 쉽게 분석할 수 있도록 Text2SQL 챗봇 인터페이스를 제공합니다. 복잡한 AWS 청구 데이터를 간단한 자연어 질문으로 조회할 수 있게 해주어, 비용 분석을 더욱 접근하기 쉽고 효율적으로 만듭니다.

# How to build
### Prerequisites
- Python 3.8 or higher
- Amazon Bedrock 액세스 요청
- [data_preparation 디렉토리](./data_preparation) 내의 노트북 3개를 실행하여 OpenSearch에 인덱스 생성
- AWS CLI configured with appropriate credentials

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ottlseo/finops-demo.git
cd application
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Streamlit application:
```bash
cd application
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
