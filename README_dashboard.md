# Free-Dog Realtime Dashboard

이 대시보드는 하이레벨 API를 통해 50Hz로 상태값을 스트리밍하고, 웹 UI에서 실시간으로 표시합니다.

## 수집 항목
- 배터리: SoC, Current, 전체 전압(mV)
- IMU: Gyroscope(deg/s), Accelerometer(m/s²), RPY(deg)
- Pose/Velocity: Position(x, y), Velocity(vx, vy)
- YawSpeed(rad/s)
- Stability Score(패널티 기반)

## 실행 방법
1) 의존성
```bash
pip install fastapi uvicorn
```

2) 서버 실행
```bash
python -m dashboard.app
```

3) 브라우저 접속
- http://localhost:8000/

## 안정성 스코어
- 초기값 100점에서 다음 조건 충족 시 패널티 차감
  - R/P 절대값: >20°: -10, >10°: -5, >5°: -2
  - Gyro 크기: >300 deg/s: -8, >200: -5, >120: -2
  - Accel 크기 편차(||a||-9.81|): >6: -6, >3: -3
  - YawSpeed 절대값: >2.5 rad/s: -4, >1.5: -2

임계값은 `dashboard/app.py`의 `_compute_stability`에서 변경 가능합니다.

## 참고
- 연결 설정은 `HIGH_WIFI_DEFAULTS`를 사용합니다. 네트워크 환경에 따라 변경 필요 시 `dashboard/app.py`에서 조정하세요.
