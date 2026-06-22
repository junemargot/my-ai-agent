from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

model_deterministic = init_chat_model(model_provider="google-genai", model="gemini-2.5-flash")
model_creative = init_chat_model(model_provider="google-genai", model="gemini-2.5-flash", temperature=1.0)

prompt = ChatPromptTemplate.from_template("다음 제품에 대한 마케팅 슬로건을 작성하세요: {product}")
response1 = (prompt | model_deterministic).invoke({"product": "스마트워치"})
response2 = (prompt | model_creative).invoke({"product": "스마트워치"})

print(response1.content) # 결정론적 결과
print(response2.content) # 창의적 결과
