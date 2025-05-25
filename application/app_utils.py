import streamlit as st
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig

initial_label = "안녕하세요, FinOps 챗봇입니다. AWS 비용과 관련해 궁금하신 점은 무엇이든 물어보세요!"
initial_questions = [
                "온디맨드와 예약 인스턴스 사용량을 인스턴스 타입별로 비교해주세요.",
                "이번 달에 새로 생성된 EC2 인스턴스의 생성 날짜와 타입, 그리고 비용을 보여주세요.",
                "775638497521 어카운트 리소스 중에 SP 적용이 가장 시급한 인스턴스 패밀리를 알려주세요."
            ]

class AppUtils:
    def __init__(self, app):
        self.app = app
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": initial_label}
            ]
            
        if "followup_questions" not in st.session_state:
            st.session_state["followup_questions"] = initial_questions 

        if "text2chart" not in st.session_state:
            st.session_state["text2chart"] = True
    
    def setup_ui(self):
        """Set up the Streamlit UI components"""
        st.set_page_config(layout="wide")
        st.title("FinOps - AWS Cost and Usage Report Analysis") 

        col1, col2 = st.columns(2)
        with col1:
            st.caption('''[Github](https://github.com/ottlseo/finops-demo/)에서 코드를 확인하실 수 있습니다.''')
        with col2: 
            show_log = st.toggle("Text2SQL 로그 확인하기", value=True)
            st.session_state["text2chart"] = st.toggle("분석 내용을 Chart로 시각화하기", value=True)
            
        return show_log
        
    def display_chat_history(self):
        """Display the chat history"""
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    def print_graph_results(self, query: str):
        """Display graph results in a streamlined format"""
        config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
        inputs = {"question": query}

        with st.chat_message("assistant"):
            progress_container = st.container() # 진행 상황 컨테이너
            response_container = st.container() # 최종 응답 컨테이너
            
            try:
                current_node = None
                for output in self.app.stream(inputs, config=config):
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
                        if isinstance(value, dict) and 'answer' in value:
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

    def print_graph_results_with_details(self, query: str):
        """Display detailed graph results with intermediate steps"""
        config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
        inputs = {"question": query}

        with st.chat_message("assistant"):
            progress_container = st.container()
            response_container = st.container()
            
            try:
                current_node = None
                node_results = {}
                previous_states = {}
                
                for output in self.app.stream(inputs, config=config):
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
                            if isinstance(value, dict) and 'answer' in value:
                                st.markdown(value['answer'])
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": value['answer']}
                                )
                            if isinstance(value, dict) and 'chart_img_path' in value:
                                img_path = value['chart_img_path']
                                print(value['chart_img_path'])
                                st.image(img_path, use_column_width=True)

            except GraphRecursionError as e:
                st.error(f"⚠️ I encountered an error: {str(e)}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ Error: {str(e)}"}
                )

    def handle_query(self, query: str, show_log: bool = False):
        """Process a user query and display results"""
        st.chat_message("user").write(query)
        st.session_state.messages.append({"role": "user", "content": query})    
        if show_log:
            self.print_graph_results_with_details(query=query)
        else:
            self.print_graph_results(query=query)
            
    def display_followup_questions(self):
        """Display followup question buttons"""
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
