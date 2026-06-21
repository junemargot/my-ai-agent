from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

# 입력 전처리 함수
def preprocess(text: str) -> dict:
  return {"question": text.strip().lower()}

# 출력 후처리 함수
def postprocess(message) -> str:
  return message.text.upper()

# 각 함수를 RunnableLambda로 변환
preprocess_runnable = RunnableLambda(preprocess)
postprocess_runnable = RunnableLambda(postprocess)

# 프롬프트와 모델 정의
prompt = ChatPromptTemplate.from_template("{question}에 대해 한 문장으로 답해 주세요.")
model = init_chat_model(model_provider="google_genai", model="gemini-2.5-flash")

# 전처리 -> 프롬프트 -> 모델 -> 후처리 체인 구성
chain = preprocess_runnable | prompt | model | postprocess

# 실행
result = chain.invoke("What is INFJ?")
print(result)