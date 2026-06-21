from langchain_core.runnables import chain

@chain
def multiply_by_two(x: int) -> int:
  return x * 2

def subtract_three(x: int) -> int:
  return x - 3

result = multiply_by_two | subtract_three

final_result = result.invoke(5)
print("result:", final_result)