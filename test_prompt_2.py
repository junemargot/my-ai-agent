from langchain_core.prompts import load_prompt, load_template
from langchain.chat_models import init_chat_model

model = init_chat_model(model_provider="google-genai", model="gemini-2.5-flash")
prompt = load_prompt("prompts/persona_with_tone.yaml")

chain = prompt | model
response = chain.invoke(input = {"subject": "API 보안의 주요 위험 요소"})
print(response.text)