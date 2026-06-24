from langchain_text_splitters import RecursiveCharacterTextSplitter

# 예시 문서: 회사 휴가 정책
documents = [
  """
    # 넥스트랩 휴가 정책(2026년 개정)
    ## 1. 연차휴가
    ### 1.1 부여 기준
    - 입사 1년 미만: 매월 1일씩 부여(최대 11일)
    - 입사 1년 이상: 연 15일 일괄 부여
    - 3년 이상 근속 시 매 2년마다 1일 추가(최대 25일)

    ### 1.2 사용 방법
    - HR 시스템에서 최소 3일 전 신청
    - 반차(0.5일) 단위로 사용 가능
    - 미사용 연차는 다음 해로 이월 불가(단, 연말 5일까지 이월 가능)

    ## 2. 특별휴가
    ### 2.1 경조사 휴가
    - 본인 결혼: 5일
    - 자녀 결혼: 1일
    - 배우자/본인 부모 사망: 5일
    - 조부모/형제자매 사망: 3일

    ### 2.2 출산/육아 휴가
    - 출산휴가: 90일(산전후휴가)
    - 배우자 출산휴가: 10일
    - 육아휴직: 최대 1년 (만 8세 이하 자녀)
    """,
]

# # 청크로 분할
# text_splitter = RecursiveCharacterTextSplitter(
#   chunk_size = 200, # 각 청크의 최대 수
#   chunk_overlap = 20, # 청크 간 겹치는 글자 수
#   separators=["\n## ", "\n### ", "\n\n", "\n", " "], # 분할 기준
# )

# chunks = text_splitter.create_documents(documents)
# print(f"총 {len(chunks)}개의 청크로 분할됨")

# # 첫 번째 청크 확인
# print(f"\n[청크 1]\n{chunks[0].page_content[:200]}...")


# 설정값에 따른 청킹 결과 확인을 위한 예시
sample_text = """
  ## 2.2 출산/육아 휴가
  - 출산 휴가: 90일(산전후휴가)
  - 배우자 출산휴가: 10일
  - 육아휴직: 최대 1년(만 8세 이하 자녀)
  
  ## 3. 병가
  - 유급 병가: 연 60일
  - 무급 병가: 연 30일 추가 가능
"""

# 작은 청크(100자)
small_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
small_chunks = small_splitter.split_text(sample_text)

# 큰 청크(300자)
large_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
large_chunks = large_splitter.split_text(sample_text)

print(f"작은 청크: {len(small_chunks)}개")
for i, chunk in enumerate(small_chunks):
  print(f" [{i} {chunk[:50]}...]")

print(f"\n큰 청크: {len(large_chunks)}개")
for i, chunk in enumerate(large_chunks):
  print(f" [{i} {chunk[:50]}...]")