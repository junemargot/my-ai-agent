import os

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

prompt = PromptTemplate(
  template = "{subject}에 대해 간략하게 설명해 주세요.",
  input_variables = ["subject"]
)

model = init_chat_model(
  model_provider = "google-genai",
  model="gemini-2.5-flash",
  api_key=os.environ["GEMINI_API_KEY"]
)

def add_ai_indicator(response):
  """AI 생성 표시 추가"""
  return response.text + "(이 응답은 AI에 의해 생성되었습니다.)"

# 프롬프트 -> 모델 -> 후처리 함수 연결
chain = prompt | model | add_ai_indicator

result = chain.invoke({"subject": "LangChain"})
print(result)
