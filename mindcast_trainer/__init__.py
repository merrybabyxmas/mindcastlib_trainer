"""
Mindcast Trainer - 한국어 텍스트 분류 모델 학습 패키지

데이터셋은 HuggingFace에서 자동으로 다운로드됩니다:
- merrybabyxmas/MindCastDataset

사용 가능한 학습 모듈:
- SD: Sarcasm Detection (비꼬는 표현 탐지)
- SC_only: Sentiment Classification (감정 분류 - 통합)
- SC_normal: Sentiment Classification (일반 댓글용)
- SC_sarc: Sentiment Classification (비꼬는 댓글용)
- TC: Topic Classification (주제 분류)
- SD_SC: SD + SC 파이프라인 전체 학습

사용법:
    python main.py                    # config.json의 train_module 사용
    python main.py --module SD        # 특정 모듈 지정

환경변수:
    HF_TOKEN: HuggingFace 토큰 (모델 업로드용)
"""

from .trainer import (
    train_SD,
    train_SC_only,
    train_SC_normal,
    train_SC_sarc,
    train_TC,
    train_SD_SC,
)
from .dataset import (
    load_from_huggingface,
    load_csv_dataset,
    load_sentiment_data,
    load_sarcasm_data,
    load_topic_data,
)
from .model import build_model_with_lora
from .evaluation import evaluate_model, generate_model_card

__version__ = "1.0.0"
__all__ = [
    # Training functions
    "train_SD",
    "train_SC_only",
    "train_SC_normal",
    "train_SC_sarc",
    "train_TC",
    "train_SD_SC",
    # Data loading
    "load_from_huggingface",
    "load_csv_dataset",
    "load_sentiment_data",
    "load_sarcasm_data",
    "load_topic_data",
    # Model building
    "build_model_with_lora",
    # Evaluation
    "evaluate_model",
    "generate_model_card",
]
