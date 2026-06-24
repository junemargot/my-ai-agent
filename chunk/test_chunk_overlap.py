from langchain_text_splitters import RecursiveCharacterTextSplitter

# overlap 없음
no_overlap = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)

# overlap 있음
with_overlap = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10)

text = "연차는 입사 1년 미만 시 매월 1일씩 부여됩니다. 1년 이상 근무하면 연 15일이 일괄 부여됩니다."

print("overlap 없음:")
for chunk in no_overlap.split_text(text):
  print(f" > {chunk}")

print("overlap 있음:")
for chunk in with_overlap.split_text(text):
  print(f" > {chunk}")