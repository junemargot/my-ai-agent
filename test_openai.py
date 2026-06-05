import os

from dotenv import load_dotenv
from langchain_openai import init_chat_model

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
  raise ValueError("OPENAI_API_KEY is not set")

model = init_chat_model(
  model_provider="openai",
  model="gpt-5.2",
  api_key=api_key
)

response = model.invoke("안녕하세요! 간단히 자기소개 해 주세요.")
print(response.content)