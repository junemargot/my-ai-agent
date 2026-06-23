# "pizza" 토큰 ID 찾기 예시
# import tiktoken
# enc = tiktoken.encoding_for_model("gemini-2.5-flash") # 원하는 모델명
# tokens = enc.encode("pizza")
# print(tokens) # [175136]

from langchain.chat_models import init_chat_model

model = init_chat_model(model_provider="openai", model="gpt-4o")
logit_bias = {
  175136: -100, # " pizza"
  27941: -100, # "pizza"
  91351: -100, # "Pizza"
}

response = model.invoke(
  "select pasta vs pizza (only word)",
  logit_bias = logit_bias,
)

print(response.text)  # "Pasta" 출력