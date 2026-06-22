import os

from langchain.chat_models import init_chat_model
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.configurable import RunnableConfigurableFields
from langchain_core.runnables.config import RunnableConfig

model = init_chat_model(
  model_provider="google-genai",
  model="gemini-2.5-flash",
  api_key=os.environ["GEMINI_API_KEY"]
)

configurable_model = model.configurable_fields(
  max_tokens=ConfigurableField(id="output_token_number")
)

response_1 = configurable_model.invoke(
  "오늘 날씨 어떄?",
  config = RunnableConfig(
    configurable = {"output_token_number": 10},
  )
)

response_2 = configurable_model.invoke(
  "오늘 날씨 어떄?",
  config = RunnableConfig(
    configurable = {"output_token_number": 100},
  )
)

print("response_1 text: ", response_1.text)
print("response_1 token: ", response_1.usage_metadata["output_tokens"])

print("response_2 text: ", response_2.text)
print("response_2 token: ", response_2.usage_metadata["output_tokens"])