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