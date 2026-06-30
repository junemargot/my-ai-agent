from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# 1. 스키마 정의
class LeaveRequest(BaseModel):
	"""휴가 신청 정보를 추출한 결과입니다."""
	
	employee_id: str = Field(
		description = "직원 ID, 예: EMP001, EMP002"
	)
	
	leave_type: str = Field(
		description = "휴가 유형. 연차, 반차, 경조사 중 하나를 선택합니다."
	)
	
	start_date: str = Field(
		description = "휴가 시작일. YYYY-MM-DD 형식으로 작성합니다."
	)
	
	end_date: str = Field(
		description = "휴가 종료일. YYYY-MM-DD 형식으로 작성합니다."
	)
	
	reason: str = Field(
		description = "휴가 사유를 간단히 요약합니다."
	)

# 2. 모델 생성 및 스키마 연결
model = init_chat_model(
  model_provider = "google-genai",
  model="gemini-2.5-flash",
)
structured_model = model.with_structured_output(LeaveRequest)

# 3. 실행
inquiry = "EMP001 직원이 2026년 7월 1일부터 7월 6일까지 가족 여행으로 연차 신청합니다."
result = structured_model.invoke(inquiry)

# 4. 결과 사용
print(f"직원 ID: {result.employee_id}")
print(f"휴가 유형: {result.leave_type}")
print(f"휴가 시작일: {result.start_date}")
print(f"휴가 종료일: {result.end_date}")
print(f"휴가 사유: {result.reason}")
