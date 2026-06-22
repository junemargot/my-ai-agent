import os

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chat_models import init_chat_model

model = init_chat_model(
  model_provider = "google-genai",
  model="gemini-2.5-flash",
  api_key=os.environ["GEMINI_API_KEY"]
)

# A. 긍정적인 Prompt를 통해 설명 생성
prompt_pros = PromptTemplate.from_template(
  template = "{subject}에 대해 긍정적인 설명을 해 주세요."
)
pros_chain = prompt_pros | model

# B. 비판적인 Prompt를 통해 설명 생성
prompt_cons = PromptTemplate.from_template(
  template = "{subject}에 대해 비판적인 설명을 해 주세요."
)
cons_chain = prompt_cons | model

# C. 두 설명을 종합하는 프롬프트
prompt_aggregate = PromptTemplate.from_template(
  template = "다음 두 설명을 종합해서 {subject}에 대한 균형 잡힌 시각을 제시해 주세요. \n\n긍정적인 설명: {pros} \n\n비판적인 설명: {cons}"
)

controversial_chain = (
  RunnableParallel(
    {
      "pros": pros_chain,
      "cons": cons_chain,
      "subject": RunnablePassthrough(),
    }
  )
  | prompt_aggregate
  | model
)

response = controversial_chain.invoke({"subject": "인공지능"})
print(response.text)