from opensearchpy import OpenSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

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

# 1단계 - 청킹(chunking)
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size = 200,
  chunk_overlap = 20,
  separators = ["\n## ", "\n### ", "\n\n", "\n", " "],
)
chunks = text_splitter.create_documents(documents)

# OpenSearch 클라이언트 설정
client = OpenSearch(
  hosts=[{"host": "localhost", "port": 9200}],
  http_auth=None,  # 개발 환경에서 인증 비활성화
  use_ssl=False
)

# 임베딩 모델 설정
embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001",
  google_api_key = os.getenv("GEMINI_API_KEY")
)

# 인덱스 생성(벡터 검색 설정 포함)
index_name = "company-docs"

index_body = {
  "settings": {
    "index": {
      "knn": True,  # 벡터 검색 활성화
    }
  },
  "mappings": {
    "properties": {
      "content": {"type": "text"},  # 문서 내용(키워드 검색용)
      "embedding": {
        "type": "knn_vector",
        # "dimension": 1536,  # text-embedding-3-small 차원
        "dimension": 3072,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil", # 코사인 유사도
          "engine": "nmslib",
        }
      },
      "metadata": {"type": "object"} # 추가 메타데이터
    }
  }
}

# 기존 인덱스가 있으면 삭제 후 생성
if client.indices.exists(index=index_name):
  client.indices.delete(index=index_name)
client.indices.create(index=index_name, body=index_body)

print(f"인덱스 '{index_name}' 생성 완료")

# 청크를 임베딩하고 인덱싱
for i, chunk in enumerate(chunks):
  # 텍스트를 벡터로 변환
  vector = embeddings.embed_query(chunk.page_content)

  # OpenSearch에 저장
  doc = {
    "content": chunk.page_content,
    "embedding": vector,
    "metadata": {
      "source": "휴가정책.md",
      "chunk_index": i,
    }
  }

  client.index(index=index_name, body=doc, id=str(i))

# 인덱스 새로고침(검색 가능하도록)
client.indices.refresh(index=index_name)
print(f"{len(chunks)}개 문서 인덱싱 완료")

def search_documents(query: str, k: int = 3) -> list[dict]:
  """질문과 유사항 문서를 검색합니다."""

  # 질문을 벡터로 변환
  query_vector = embeddings.embed_query(query)

  # 벡터 검색 쿼리
  search_query = {
    "size": k,
    "query": {
      "knn": {
        "embedding": {
          "vector": query_vector,
          "k": k
        }
      }
    }
  }

  # 검색 실행
  response = client.search(index = index_name, body = search_query)

  # 결과 추출
  results = []
  for hit in response["hits"]["hits"]:
    results.append({
      "content": hit["_source"]["content"],
      "score": hit["_score"],
      "metadata": hit["_source"]["metadata"]
    })

  return results