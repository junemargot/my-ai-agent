import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 벡터 스토어 설정
vectorstore = OpenSearchVectorSearch(
  opensearch_url = "http://localhost:9200",
  index_name = "company-docs",
  embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
  ),
  vector_field = "embedding",
  text_field = "content"
)

# 검색기 생성
retriever = vectorstore.as_retriever(
  search_type = "similarity", # 유사도 기반 검색
  search_kwargs = {"k": 3, "vector_field": "embedding", "text_field": "content"},
)

# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_template("""
  다음 문서를 참고하여 질문에 답변하세요.
  문서에 없는 내용은 "해당 정보는 문서에서 확인할 수 없습니다."라고 답변하세요.

  [참고 문서]
  {context}

  [질문]
  {question}
  """)

# LLM 설정
llm = init_chat_model(
  model_provider = "google-genai",
  model="gemini-2.5-flash",
  temperature=0
)

def format_docs(docs):
  """검색된 문서를 문자열로 포맷합니다."""
  return "\n\n---\n\n".join(doc.page_content for doc in docs)

# LCEL로 RAG 체인 구성
rag_chain = (
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)

# 질문
answer = rag_chain.invoke("육아휴직 기간을 얼마나 되나요?")
print(f"답변: {answer}")