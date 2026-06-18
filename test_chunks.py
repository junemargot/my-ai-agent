from langchain_core.messages import AIMessageChunk
from langchain_core.messages.ai import UsageMetadata

# 스트리밍 응답으로 받은 청크 묶음
stream_chunks = [
    AIMessageChunk(
        content="LangChain은",
        usage_metadata=UsageMetadata(
            input_tokens=10,
            total_tokens=15,
            output_tokens=5,
        ),
    ),
    AIMessageChunk(
        content="스트리밍 응답을",
        usage_metadata=UsageMetadata(
            input_tokens=0,
            total_tokens=4,
            output_tokens=4,
        ),
    ),
    AIMessageChunk(
        content="잘 처리합니다.",
        usage_metadata=UsageMetadata(
            input_tokens=0,
            total_tokens=5,
            output_tokens=5,
        ),
    ),
]

complete_message = stream_chunks[0]
for chunk in stream_chunks[1:]:
    complete_message += chunk

print("응답:", complete_message.text)
print("총 토큰 수:", complete_message.usage_metadata)
