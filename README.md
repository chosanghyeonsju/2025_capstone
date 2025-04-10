# 2025_capstone

## LLM 이용한 사용자 심리 상담 시스템
- 개요: 사용자가 대화를 통해 심리 상태의 개선을 이룰 수 있는 LLM 기반 어플리케이션을 개발함
- 요구사항
  
  1.  단순한 챗봇 시스템을 구축해서는 안되며, 심리 상담 이론에 근거한 대화가 이뤄져야 함
  2. 대화가 가능한 GUI가 구축되어야 함
  3. 사용자가 나중에 중간에 대화를 종료하고, 나중에 대화를 이어 나가길 원할 경우 다시 맥락을 이어 나갈 수 있는 기능이 구현되어야 함
  4. 사용자를 대상으로 개발된 어플리케이션의 효과성(심리 상태 개선)에 대한 검증이 이뤄져야 함

## 커밋메시지 규칙
feat : 새로운 기능의 추가

fix: 버그 수정

docs: 문서 수정

style: 스타일 관련 기능 (코드 포맷팅, 세미콜론 누락, 코드 자체의 변경이 없는 경우)

refactor: 코드 리펙토링

test: 테스트 코트, 리펙토링 테스트 코드 추가

chore: 빌드 업무 수정, 패키지 매니저 수정(ex .gitignore 수정)


예시) 이슈번호가 5 일때 커밋메시지 -> feat #5: add new function.
                      브랜치 이름 -> feat#5


## 프로젝트 구조도 (예시)
```
2025_CAPSTONE/
│
├── frontend/              # 프론트엔드 (React, Next.js 등)
│   ├── public/
│   ├── src/
│   └── package.json
│
├── backend/               # 백엔드 (FastAPI, Flask, Django 등)
│   ├── app/
│   ├── requirements.txt
│   └── main.py
│
├── model/                 # AI/ML 모델
│   ├── inference.py       # 예측 함수
│   ├── train.py           # 학습용 코드
│   ├── model.pkl          # 저장된 모델 파일
│   └── utils.py
│
├── database/              # DB 관련 구성 (초기 스키마, 마이그레이션, 쿼리 등)
│   ├── schema.sql
│   ├── init_db.py
│   └── config.py
│
├── streamlit/             # 프로토타입용 Streamlit 앱
│   ├── app.py
│   └── components/
│
├── .env                   # 공통 환경 변수 (백/모델에서 공유할 경우)
├── README.md
└── docker-compose.yml     # 전체 시스템 통합 (옵션)
```
