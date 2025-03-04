import streamlit as st 
import utils.incoder as bedrock

st.set_page_config(layout="wide")
st.title("Welcome to AWS Multi-modal RAG Demo!") 

st.markdown('''- 이 데모는 검색 증강 생성 (RAG)을 활용한 생성형 AI 애플리케이션을 빠르게 구성하고 테스트해볼 수 있도록 간단한 챗봇 형태로 제공됩니다.''')
st.markdown('''- 복잡하게 느껴질 수 있는 RAG 구성, 예를 들면 VectorStore Embedding 작업부터 Amazon OpenSearch 클러스터 생성 및 문서 인덱싱, Bedrock 세팅까지 모든 작업을 템플릿으로 자동화함으로써 한 번의 CDK 배포만으로도 RAG 개발 및 테스트를 하고싶은 누구든 빠르게 활용할 수 있도록 돕는 것을 목표로 하고 있습니다.''')
st.markdown('''- [Github](https://github.com/ottlseo/bedrock-rag-chatbot/)에서 코드를 확인하실 수 있습니다.''')

normalizer = bedrock.ServiceNameNormalizer()

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    btn1 = st.button("👉 **이 RAG의 아키텍처를 보여주세요.**")
with col2:
    btn2 = st.button("👉 **이 애플리케이션의 UI는 어떻게 만들어졌나요?**")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요, 무엇이 궁금하세요?"}
    ]
# 지난 답변 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 유저가 쓴 chat을 query라는 변수에 담음
query = st.chat_input("Search documentation")
if query:
    # Session에 메세지 저장
    st.session_state.messages.append({"role": "user", "content": query})
    
    # UI에 출력
    st.chat_message("user").write(query)

    # UI 출력
    # answer = bedrock.query(query)
    answer = normalizer.process_text(query)
    st.chat_message("assistant").write(answer)

    # Session 메세지 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})
        