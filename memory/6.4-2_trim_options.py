import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import trim_messages, HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model_provider="google-genai",
    model="gemini-2.5-flash",
    temperature=0.7
)

# 실험용 메시지 목록
messages = [
    SystemMessage("당신은 친절한 HR 상담 어시스턴트입니다."),
    HumanMessage("저는 개발팀 김철수입니다."),
    AIMessage("안녕하세요, 김철수님!"),
    HumanMessage("연차가 며칠 남았나요?"),
    AIMessage("연차 잔여 일수를 확인해드리겠습니다."),
    HumanMessage("다음 주에 3일 휴가 쓸 수 있을까요?"),
    AIMessage("네, 가능합니다."),
    HumanMessage("제가 어느 부서에서 일하고 있죠?"),
]

def print_messages(label, msgs):
    print(f"\n=== {label} (총 {len(msgs)}개) ===")
    for m in msgs:
        role = type(m).__name__.replace("Message", "")
        print(f"  [{role}] {m.content}")

print_messages("원본 메시지", messages)

# ── 방법 1: 메시지 수로 제한 ──────────────────────────────
# token_counter=len 으로 설정하면 토큰이 아닌 메시지 개수로 카운트
trimmer_by_count = trim_messages(
    max_tokens=4,           # 최대 4개 메시지 유지
    strategy="last",        # 최근 메시지 우선
    token_counter=len,      # 메시지 개수로 카운트
    include_system=True,    # system 메시지는 항상 유지
    start_on="human",       # human 메시지부터 시작
)

result_count = trimmer_by_count.invoke(messages)
print_messages("메시지 수 제한 (max=4)", result_count)

# ── 방법 2: 토큰 수로 제한 ───────────────────────────────
# token_counter=model 로 설정하면 실제 토큰 수로 카운트
trimmer_by_token = trim_messages(
    max_tokens=20,         # 최대 1000토큰 유지
    strategy="last",
    token_counter=model,    # 모델이 직접 토큰 계산 (모델의 토크나이저 사용)
    include_system=True,
    start_on="human",
)

result_token = trimmer_by_token.invoke(messages)
print_messages("토큰 수 제한 (max=1000)", result_token)
