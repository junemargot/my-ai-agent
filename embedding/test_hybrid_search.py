from opensearchpy import OpenSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001"
)

# 검색 파이프라인 등록 (최초 1회)
pipeline_body = {
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {"technique": "min_max"},
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.7, 0.3]
          }
        }
      }
    }
  ]
}
client.transport.perform_request(
  "PUT", "/_search/pipeline/hybrid-pipeline", body=pipeline_body
)

def hybrid_search(
  query: str,
  index_name: str,
  k: int = 5,
  vector_weight: float = 0.7,
  keyword_weight: float = 0.3,
) -> list[dict]:
  """하이브리드 검색을 수행합니다.

    Args:
      query: 검색 질문
      index_name: 검색할 인덱스 이름
      k: 반환할 결과 수
      vector_weight: 벡터 검색 가중치(0~1)
      keyword_weight: 키워드 검색 가중치(0~1)

    Returns:
      검색 결과 리스트
  """

  # 질문을 벡터로 변환
  query_vector = embeddings.embed_query(query)

  # 하이브리드 검색 쿼리
  search_query = {
    "size": k,
    "query": {
      "hybrid": {
        "queries": [
          # 벡터 검색(의미 기반)
          {
            "knn": {
              "embedding": {
                "vector": query_vector,
                "k": k * 2,
              }
            }
          },
          # 키워드 검색(BM25)
          {
            "match": {
              "content": {
                "query": query,
                "boost": 1.0,
              }
            }
          }
        ]
      }
    }
  }

  response = client.search(
    index=index_name,
    body=search_query,
    params={"search_pipeline": "hybrid-pipeline"}
  )

  results = []
  for hit in response["hits"]["hits"]:
    results.append({
      "content": hit["_source"]["content"],
      "score": hit["_score"],
      "metadata": hit["_source"].get("metadata", {}),
    })

  return results

results = hybrid_search("육아휴직 기간은?", "company-docs")
for r in results:
  print(f"score: {r['score']:.4f}")
  print(f"content: {r['content'][:100]}")
  print("-" * 50)
