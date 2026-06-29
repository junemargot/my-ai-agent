import os

from langchain_core.tools import Tool, tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from tool.check_leave_balance import check_leave_balance
from chain import execute_tool

@tool(description = "직원 ID를 받아 해당 직원의 연차 잔여일수를 조회합니다.")
def check_leave_balance(employee_id: str) -> dict:
  """직원의 연차 잔여 일수를 조회하는 예제용 도구 함수입니다.

  Args:
    employee_id: 연차 정보를 조회할 직원 ID입니다.

  Returns:
    연차 조회 결과를 담은 딕셔너리입니다.
    실제 서비스에서는 데이터베이스나 외부 API 호출 결과를 담습니다.
  """

  # 실제 구현에서는 데이터베이스나 외부 API를 호출하여 연차 정보를 조회한다.
  total_leave = 15 # 총 연차 일수
  used_leave = 3   # 사용한 연차 일수
  return {
    "status": "success",
    "data": {
      "employee_id": employee_id,
      "total_days": total_leave,
      "used_days": used_leave,
      "remaining_days": total_leave - used_leave
    },
  }

chat_model = init_chat_model(
  model_provider = "google-genai",
  model = "gemini-2.5-flash",
)

tool_check_leave_balance = Tool.from_function(
  func = check_leave_balance,
  name = "check_leave_balance",
  description = "직원 ID를 받아 해당 직원의 연차 잔여일수를 조회합니다."
)

model_with_tool = chat_model.bind_tools(tools = [tool_check_leave_balance])

# 1. 사용자의 자연어 요청을 LLM에 전달
user_request = HumanMessage(content = "직원 EMP001의 연차 잔여일수 알려 줘.")
ai_message = model_with_tool.invoke(input = [user_request])
print("호출 정보: ", ai_message.tool_calls)

# 2. LLM이 제안한 도구 호출 정보를 바탕으로 실제 도구를 실행합니다.
tool_response = execute_tool(ai_message, tools = {"check_leave_balance": tool_check_leave_balance})
print("연차 조회 실행 결과: ", tool_response)

# 3. 도구 실행 결과를 다시 LLM에 전달해 최종 자연어 응답을 생성합니다.
final_response = model_with_tool.invoke(input=[user_request, ai_message] + tool_response)
print("최종 응답: ", final_response.content)

# 4. 일반적인 대화 요청의 경우에는 도구를 사용하지 않고 바로 응답을 생성할 수 있습니다.
user_request = HumanMessage(content="반가워")
ai_message = model_with_tool.invoke(input = [user_request])
print("일반 답변: ", ai_message.content)