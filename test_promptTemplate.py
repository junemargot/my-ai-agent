import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

load_dotenv()

# input_variables로 필수 변수를 명시하면 런타임에 누락 여부를 검증할 수 있다.
prompt=PromptTemplate(
  template="{subject}에 대해 간략하게 설명해 주세요",
  input_variables=["subject"],
)

filled_prompt=prompt.invoke({"subject": "LLM"})
print("filled_prompt:", filled_prompt.to_string())

print("type(filled_prompt):", type(filled_prompt))

model=init_chat_model(
  model_provider="google_genai",
  model="gemini-2.5-flash",
  api_key=os.environ["GEMINI_API_KEY"]
)

response=model.invoke(filled_prompt)
print("response text:", response.text)