from langchain_core.runnables import RunnablePassthrough, passthrough

passthrough = RunnablePassthrough()

# 입력 데이터
input_data = {"subject": "인공지능", "level": "high"}

output = passthrough.invoke(input_data)
print(output)