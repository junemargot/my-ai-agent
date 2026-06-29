
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import Tool


def execute_tool(message: AIMessage, tools: dict[str, Tool]) -> list[ToolMessage]:
  """LLM이 제안한 도구 호출 정보를 해석해 실제 도구를 실행한다.

  Args:
    message: LLM이 생성한 AIMessage입니다. tool_calls 필드에 도구 호출 정보가 들어있습니다.
    tools: 도구 이름을 키로 하는 Tool 객체 딕셔너리입니다.

  Returns:
    각 도구 실행 결과를 담은 ToolMessage 리스트입니다.
  """

  results: list[ToolMessage] = []

  # LLM이 제안한 각 도구 호출 정보를 순회하며 실제 도구를 실행합니다.
  for call in message.tool_calls:
    tool = tools[call["name"]]
    # call["args"]에는 LLM이 JSON 스키마에 맞춰 채운 인수가 들어 있습니다.
    try :
      output = tool.invoke(call["args"])
    except Exception as e:
      output = {"status": "error", "message": str(e)}

      # 도구 실행 결과는 ToolMessage로 래핑해 다시 LLM에 전달한다.
      results.append(
        ToolMessage(
          content = str(output),
          tool_call_id = call["id"], # 어떤 도구 호출 결과인지 연결하기 위한 ID
        )
      )

  return results
