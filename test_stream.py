import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
  raise ValueError("GEMINI_API_KEY is not set")

model = init_chat_model(
  model_provider="google_genai",
  model="gemini-2.5-flash",
  api_key=api_key
)

messages = [
  SystemMessage(content="당신은 사용자의 모든 발화를 영어로 번역합니다."),
  HumanMessage(content="LangChain의 구조를 설명해주세요.")
]

response = model.invoke(messages)
print(response.text)

# response_stream = model.stream("한국의 계절은?")

# for chunk in response_stream:
#   if chunk.text:
#     print(chunk.text, end="", flush=True)