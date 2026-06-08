import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
  raise ValueError("GEMINI_API_KEY is not set")

model = init_chat_model(
  model_provider="google_genai",
  model="gemini-2.5-flash",
  api_key=api_key
)

response = model.invoke("안녕하세요! 간단히 자기소개 해 주세요.")
print(response.content)