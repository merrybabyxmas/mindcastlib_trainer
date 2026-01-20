# Mindcast Trainer

한국어 텍스트 분류 모델 학습 패키지

## 설치

```bash
# 1. conda 가상환경 생성
conda create -n mindcast python=3.10 -y
conda activate mindcast

# 2. PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 패키지 설치
pip install -r requirements.txt
```

## 실행

```bash
conda activate mindcast

# 기본 실행 (config.json의 train_module 사용)
python main.py

# 특정 모듈 지정
python main.py --module SD        # Sarcasm Detection
python main.py --module SC_only   # Sentiment Classification (통합)
python main.py --module TC        # Topic Classification
python main.py --module SD_SC     # SD + SC 파이프라인 전체 학습
```

## Config 설정

`config.json` 수정:

```json
{
  "train_module": "SC_only",      // 학습할 모듈: SD, SC_only, SC_normal, SC_sarc, TC, SD_SC

  "train_config": {
    "batch_size": 64,             // 배치 크기
    "epochs": 50,                 // 학습 에폭
    "learning_rate": 1e-4         // 학습률
  },

  "wandb_config": {
    "use_wandb": true             // WandB 사용 여부
  },

  "device_config": {
    "cuda_device": "cuda:0"       // GPU 디바이스
  }
}
```

## 환경변수

```bash
export HF_TOKEN="your_token"      # HuggingFace 토큰 (모델 업로드용)
```
