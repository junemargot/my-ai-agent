from langchain.chat_models import init_chat_model

model = init_chat_model(
  model_provider = "google-genai",
  model="gemini-2.5-flash",
  temperature=0)

# 학습 이후의 정보를 물어보면?
# response = model.invoke("2026년 6월 현재 원/달러 환율은 얼마인가요?")
response = model.invoke("""
우리 회사 '넥스트랩'의 재택근무 정책에 대해 설명해 주세요.
며칠까지 재택이 가능한가요?
""")
print(response.text)