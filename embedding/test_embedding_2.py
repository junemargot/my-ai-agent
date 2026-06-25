import numpy as np

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
  model="models/gemini-embedding-001",
)

# 텍스트를 벡터로 변환
text1 = "오늘 날씨가 좋습니다."
text2 = "오늘 기분이 좋습니다."
text3 = "데이터베이스 인덱스를 생성합니다."

vec1 = embeddings.embed_query(text1)
vec2 = embeddings.embed_query(text2)
vec3 = embeddings.embed_query(text3)

print(f"벡터 차원: {len(vec1)}") # 3072차원
print(f"벡터 예시 (처음 5개): {vec1[:5]}")


# 벡터의 코사인 유사도 계산
def cosine_similarity(v1, v2):
  """두 벡터의 코사인 유사도를 계산합니다."""
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 유사도 계산
sim_1_2 = cosine_similarity(vec1, vec2)
sim_1_3 = cosine_similarity(vec1, vec3)

print(f"'날씨가 좋다' vs '기분이 좋다': {sim_1_2:.3f}")
print(f"'날씨가 좋다' vs '데이터베이스': {sim_1_3:.3f}")