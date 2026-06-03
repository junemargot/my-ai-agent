from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
  ("system", "당신은 친절한 AI 어시스턴트입니다."),
  ("user", "{input}")
])

# 프롬프트 확인
messages = prompt.invoke({"input": "안녕하세요!"})
print("생성된 프롬프트:")
for msg in messages.messages:
  print(f" [{msg.type}] {msg.text}")