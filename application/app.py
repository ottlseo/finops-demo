import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from lib.bedrock import BedrockClient
from lib.opensearch import init_search_resources
from graph import WorkflowBuilder
from app_utils import AppUtils
from config import *

# Initialize database connection
engine = create_engine(ATHENA_CONNECTION_STRING, echo=True)
Session = sessionmaker(bind=engine)

# Initialize Bedrock client
bedrock_client = BedrockClient(region=REGION, llm_model=SONNET)

# Initialize OpenSearch resources
sql_search_client, table_search_client, sql_retriever, table_retriever = init_search_resources(
    region_name=REGION,
    example_queries_index=EXAMPLE_QUERIES_INDEX,
    table_description_index=TABLE_DESCRIPTION_INDEX
)

# Initialize workflow builder with refactored structure
workflow_builder = WorkflowBuilder(
    bedrock_client=bedrock_client,
    sql_search_client=sql_search_client,
    table_search_client=table_search_client,
    sql_retriever=sql_retriever,
    table_retriever=table_retriever,
    session_maker=Session,
    dialect=DIALECT
)

# Build the workflow
app = workflow_builder.build_finops_workflow(chart_option=st.session_state.get("text2chart", True))

# Initialize UI
ui = AppUtils(app)

# Setup UI
show_log = ui.setup_ui()

# Display chat history
ui.display_chat_history()

# Handle new query
query = st.chat_input("질문을 입력하세요.")
if query:
    if "followup_questions" in st.session_state:
        del st.session_state.followup_questions

# Handle session state query
if "query" in st.session_state:
    query = st.session_state.query
    del st.session_state.query
    ui.handle_query(query, show_log)
elif query:
    ui.handle_query(query, show_log)

# Display followup questions
ui.display_followup_questions()
