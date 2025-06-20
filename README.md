# CNN-Transformer를 이용한 암호화폐 가격 예측 모델

## 📊 프로젝트 개요
CNN-Transformer 하이브리드 모델을 사용하여 암호화폐 가격을 예측하는 프로젝트입니다. ETH, ADA, XRP 데이터로 학습하여 BTC의 고가/저가를 예측하는 순환 학습(Curriculum Learning) 방식을 적용했습니다.

## 🏗️ 모델 구조
- **CNN**: 1D 컨볼루션으로 단기 패턴 추출 (16→64 채널)
- **Transformer**: 멀티헤드 어텐션으로 장기 의존성 학습 (8헤드, 3레이어)
- **입력**: (batch, 52, 16) → **출력**: (batch, 2) [고가, 저가]
- **파라미터 수**: 약 633,922개

## 📁 파일 구조

### **핵심 파일**
- `cnn_transformer_model.ipynb` - 메인 실험 노트북 (모델 훈련 및 평가)
- `requirements.txt` - Python 의존성 라이브러리 및 버전 정보
- `preprocess_rl.py` - 암호화폐 데이터 전처리 스크립트

### **데이터 파일**
- `preprocessed/` - 전처리된 암호화폐 데이터 파일들
  - `ADA_1h.npy` - 카르다노(ADA) 1시간 봉 데이터
  - `BTC_1h.npy` - 비트코인(BTC) 1시간 봉 데이터
  - `ETH_1h.npy` - 이더리움(ETH) 1시간 봉 데이터
  - `XRP_1h.npy` - 리플(XRP) 1시간 봉 데이터

### **생성되는 파일** (실행 중 생성)
- `models/` - 훈련된 모델 체크포인트 저장 폴더
- `raw/` - 원본 데이터 폴더 (해당시)

## 🔧 실험 환경 설정

### **Python 버전**
- Python 3.10.0

### **라이브러리 설치**
```bash
pip install -r requirements.txt
```

### **주요 의존성**
- torch==2.8.0.dev20250419+cu128 (CUDA 지원 PyTorch)
- numpy==1.23.5
- pandas==2.2.3
- matplotlib==3.10.1
- stable_baselines3==2.6.0 (강화학습 컴포넌트용)

## 🚀 실행 방법

### **1. 환경 설정**
```bash
# 의존성 라이브러리 설치
pip install -r requirements.txt

# GPU 사용 가능 여부 확인 (선택사항, 권장)
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

### **2. 실험 실행**
Jupyter 노트북을 열어서 실행:
```bash
jupyter notebook cnn_transformer_model.ipynb
```

**중요**: `Kernel → Restart & Run All`을 사용하여 전체 코드가 오류 없이 실행되는지 반드시 확인하세요.

### **3. 예상 실행 시간**
- GPU 환경 (RTX 5070): 약 30-60분
- CPU 환경: 약 2-4시간 (권장하지 않음)

## 📈 실험 상세 정보

### **훈련 전략**
- **순환 학습**: ETH → ADA → XRP 순차 훈련 (3사이클)
- **코인별 에포크**: 10 에포크
- **총 훈련 단계**: 90 에포크
- **배치 크기**: 128

### **데이터 구성**
- **훈련 데이터**: ETH, ADA, XRP (3개 암호화폐)
- **테스트 데이터**: BTC (비트코인)
- **시퀀스 길이**: 52 타임스텝
- **특징 수**: 16개 기술지표 (가격, 거래량, RSI, MACD 등)

### **모델 훈련 설정**
- **옵티마이저**: Adam (학습률=0.001, weight_decay=1e-5)
- **손실 함수**: MSE (평균 제곱 오차)
- **스케줄러**: ReduceLROnPlateau (patience=3, factor=0.5)

## 📊 예상 결과

### **훈련 과정**
- 사이클별 점진적 손실 감소
- 4개 서브플롯으로 구성된 훈련 시각화:
  - 코인별 훈련 손실 변화
  - 사이클별 테스트 손실 변화
  - 코인별 최종 훈련 손실 비교
  - 전체 훈련 과정 요약

### **예측 성능 분석**
- 시계열 비교: 실제값 vs 예측값 (고가/저가)
- 성능 지표: MSE, MAE, 방향성 정확도
- 최근 500개 데이터 포인트 시각화

## ⚠️ 주의사항

### **하드웨어 요구사항**
- **권장**: CUDA 호환 GPU (8GB+ VRAM)
- **최소**: CPU 훈련용 16GB RAM

### **파일 의존성**
- `preprocessed/` 폴더에 모든 전처리된 데이터 파일이 있어야 함
- 모델은 체크포인트 저장을 위해 `models/` 폴더를 자동 생성

### **오류 방지**
- 실행 전 모든 파일이 올바른 디렉토리에 있는지 확인
- `Restart & Run All`로 전체 실행 가능 여부 검증
- GPU 훈련 시 CUDA 사용 가능 여부 확인

## 🎯 재현 가능성

이 저장소는 완전한 재현이 가능하도록 설계되었습니다:
1. 저장소 다운로드 및 압축 해제
2. 라이브러리 설치: `pip install -r requirements.txt`
3. 노트북 실행: `jupyter notebook cnn_transformer_model.ipynb`
4. 전체 실행: `Kernel → Restart & Run All`

모든 실험은 오류 없이 완료되며 일관된 결과를 산출합니다.

## 📝 데이터 출처
- 공개 API를 통해 수집된 암호화폐 가격 데이터
- 표준 금융 라이브러리를 사용한 기술지표 계산
- 정규화 및 특징 공학을 포함한 데이터 전처리 