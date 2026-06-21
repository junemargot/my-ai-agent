from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
  return x + 1

add_one_runnable = RunnableLambda(add_one)
result = add_one_runnable.invoke(6)
print(result)