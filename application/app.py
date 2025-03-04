import streamlit as st 
import utils.ddb as ddb

st.set_page_config(layout="wide")
st.title("FinOps Demo (Text2Sql & Text2Chart)💸") 
st.markdown('''- [Github](https://github.com/ottlseo/finops-demo/)에서 코드를 확인하실 수 있습니다.''')

normalizer = ddb.ServiceNameNormalizer()

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
    answer = normalizer.process_text(query) # TODO: 추후 Nova 로직 추가 (text2sql.py)
    st.chat_message("assistant").write(answer)

    # Session 메세지 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})
        