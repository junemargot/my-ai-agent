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

response_stream = model.stream("한국의 계절은?")

for chunk in response_stream:
  if chunk.text:
    print(chunk.text, end="", flush=True)