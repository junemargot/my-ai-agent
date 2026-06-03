import streamlit as st

# 페이지 설정
st.set_page_config(
  page_title="AI Agent Chat",
  page_icon=":robot_face",
  layout="centered"
)

st.title("🤖 AI Agent Chat")

# 세션 상태로 대화 기록 관리
if "messages" not in st.session_state:
  st.session_state.messages = []

# 이전 대화 표시
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요."):
  #  사용자 메시지 추가
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

    # AI 응답 (임시 에코 응답)
    response=f"입력하신 메시지: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
      st.markdown(response)