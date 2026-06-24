from langchain.chat_models import init_chat_model

model = init_chat_model(
  model_provider = "google-genai",
  model="gemini-2.5-flash",
  temperature=0)

# 검색된 문서 예시(실제로는 검색 시스템에서 가져옴)
retrieved_docs = """
  [넥스트랩 재택근무 정책 v2.3 - 2026년 개정]

  1. 재택근무 가능 일수: 주 3일까지 가능
  2. 신청방법: HR 시스템에서 최소 하루 전 신청
  3. 승인 절차: 자동 승인(특별 프로젝트 기간 제외)
  4. 필수 출근일: 매주 화요일은 전사 출근일
"""

# 질문과 검색 결과를 LLM에 함께 전달
prompt = f"""
  다음 문서를 참고하여 질문에 답변해 주세요.
  문서에 없는 내용은 "문서에서 확인할 수 없습니다."라고 답변해 주세요.

  [참고 문서]
  {retrieved_docs}

  [질문]
  우리 회사 재택근무는 며칠까지 가능한가요?
"""

response = model.invoke(prompt)
print(response.text)