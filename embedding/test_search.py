from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from embedding.test_embedding import search_documents

# LLM 설정
llm = init_chat_model(
  model_provider = "google-genai",
  model = "gemini-2.5-flash",
  temperature = 0
)

# RAG 프롬프트 템플릿
rag_prompt = ChatPromptTemplate([
  ("system", """
    당신은 회사 정책 안내 도우미입니다.
    주어진 문서만을 참고하여 질문에 답변하세요.

    규칙:

    1. 문서에 있는 내용만 답변하세요.

    2. 문서에 없는 내용은 "해당 정보는 문서에서 확인할 수 없습니다."라고 답변하세요.

    3. 답변 끝에 참고한 문서 출처를 명시하세요.
  """), ("user", """[참고 문서]
  {context}
  
  [질문]
  {question}
  """),
])

def ask_rag(question: str) -> str:
  """RAG를 사용하여 질문에 답변합니다."""

  # 1. 관련 문서 검색
  search_results = search_documents(question, k=3)

  # 2. 검색 결과를 컨텍스트로 구성
  context = "\n\n---\n\n".join([
    f"[출처: {r['metadata']['source']}, 청크 {r['metadata']['chunk_index']}]\n{r['content']}"
    for r in search_results
  ])

  # 3. LLM에 질문과 컨텍스트 전달
  chain = rag_prompt | llm
  response = chain.invoke({
    "context": context,
    "question": question,
  })

  return response.text

# 테스트
questions = [
  "신입사원 연차는 며칠인가요?",
  "배우자 출산휴가는 며칠인가요?",
  "연차 이월이 가능한가요?"
]

for q in questions:
  print(f"Q: {q}")
  print(f"A: {ask_rag(q)}")
  print("-" * 50)