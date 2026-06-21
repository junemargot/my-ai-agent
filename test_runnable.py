from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 각 컴포넌트는 모두 Runnable을 상속받는다.
model = init_chat_model(model_provider="google_genai", model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_template("{topic}에 대해 설명해 주세요.")
parser = StrOutputParser()

# 모두 동일한 invoke 인터페이스 실행된다.
prompt_value = prompt.invoke({"topic": "양자 컴퓨팅"})
model_output = model.invoke(prompt_value)
final_result = parser.invoke(model_output)

# 또는 파이프 연산자로 체인을 구성
chain = prompt | model | parser
result = chain.invoke({"topic": "양자 컴퓨팅"})
print(result)