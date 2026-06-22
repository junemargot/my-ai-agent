import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
  template = "{subject}에 대해 간략하게 설명해 주세요.",
  input_variables = ["subject"],
)

model = init_chat_model(
  model_provider = "google-genai",
  model="gemini-2.5-flash",
  api_key=os.environ["GEMINI_API_KEY"]
)

# LCEL: 파이프 연산자로 컴포넌트 연결
chain = prompt | model

response = chain.invoke(input={"subject": "커피"})
# print(response)
print(response.content)