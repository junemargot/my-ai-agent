import os
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

load_dotenv()

# 세션별 대화 기록 저장소
chat_histories = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
  """세션 ID에 해당하는 대화 기록을 반환합니다."""
  if session_id not in chat_histories:
    chat_histories[session_id] = InMemoryChatMessageHistory()
  return chat_histories[session_id]

# 프롬프트 템플릿 (대화 기록 포함)
prompt = ChatPromptTemplate.from_messages([
  ("system", "당신은 친절한 HR 상담 어시스턴트입니다. 직원의 휴가, 복리후생, 인사 관련 질문에 답변합니다."),
  MessagesPlaceholder(variable_name = "history"),
  ("human", "{input}")
])

# 모델 설정
model = init_chat_model(
  model_provider = "google-genai",
  model = "gemini-2.5-flash",
  temperature = 0.7
)

# 체인 생성
chain = prompt | model

# 메시지 히스토리가 포함된 체인
chain_with_history = RunnableWithMessageHistory(
  chain,
  get_chat_history,
  input_messages_key = "input",
  history_messages_key = "history"
)

# 대화 실행
config = {"configurable": {"session_id": "user-123"}}

response1 = chain_with_history.invoke(
  {"input": "저는 개발팀 김철수입니다. 연차 잔여 일수 확인하고 싶어요."},
  config = config
)

print(response1.text)

response2 = chain_with_history.invoke(
  {"input": "제 이름이 뭐였죠?"},
  config = config
)

# print(response2.text)


## 대화 기록 확인하기
history = get_chat_history("user-123")

print("=== 저장된 대화 내용 ===")
for message in history.messages:
  role = "사용자" if message.type == "human" else "AI"
  print(f"{role}: {message.content}")

# session_id
# 김철수와의 대화
config_철수 = {"configurable": {"session_id": "emp-김철수"}}
chain_with_history.invoke({"input": "저는 개발팀 김철수입니다."}, config = config_철수)

config_영희 = {"configurable": {"session_id": "emp-박영희"}}
chain_with_history.invoke({"input": "저는 디자인팀 박영희입니다."}, config = config_영희)

# 김철수의 대화 이어가기
response = chain_with_history.invoke({"input": "제가 어느 팀이었죠?"}, config = config_철수)

print(response.text)