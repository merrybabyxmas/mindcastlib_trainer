#!/usr/bin/env python3
"""
Mindcast Trainer - 메인 실행 스크립트

사용법:
    python main.py                     # config.json의 train_module 사용
    python main.py --module SD         # 특정 모듈 지정
    python main.py --module SC_only --config custom_config.json

환경변수:
    HF_TOKEN: HuggingFace 토큰 (config.json보다 우선)
"""
import os
import json
import argparse
import torch

from mindcast_trainer import (
    train_SD,
    train_SC_only,
    train_SC_normal,
    train_SC_sarc,
    train_TC,
    train_SD_SC,
)


TRAIN_FUNCTIONS = {
    "SD": train_SD,
    "SC_only": train_SC_only,
    "SC_normal": train_SC_normal,
    "SC_sarc": train_SC_sarc,
    "TC": train_TC,
    "SD_SC": train_SD_SC,
}


def load_config(config_path: str) -> dict:
    """설정 파일 로드 및 환경변수 처리"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # HuggingFace 토큰: 환경변수 우선
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        cfg["huggingface_config"]["token"] = hf_token
        print("[CONFIG] HuggingFace token loaded from HF_TOKEN environment variable")
    elif cfg["huggingface_config"].get("token") is None:
        print("[WARN] No HuggingFace token found. Set HF_TOKEN environment variable or add token to config.json")
        cfg["huggingface_config"]["use_huggingface"] = False

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Mindcast Trainer")
    parser.add_argument(
        "--module",
        type=str,
        choices=list(TRAIN_FUNCTIONS.keys()),
        help="학습할 모듈 (기본값: config.json의 train_module)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="설정 파일 경로 (기본값: config.json)",
    )
    args = parser.parse_args()

    # 설정 로드
    cfg = load_config(args.config)

    # 디바이스 설정
    device = cfg["device_config"]["cuda_device"]
    cfg["device"] = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[GPU] Using device: {cfg['device']}")

    # 학습 모듈 결정
    train_module = args.module or cfg["train_module"]

    # WandB run name 동적 설정
    if cfg["wandb_config"]["use_wandb"]:
        base_run_name = cfg["wandb_config"].get("run_name", "mindcast")
        cfg["wandb_config"]["run_name"] = f"{base_run_name}-{train_module}"
        print(f"[WANDB] Run name: {cfg['wandb_config']['run_name']}")

    print(f"\n==== TRAINING MODULE : {train_module} ====\n")

    # 학습 함수 실행
    if train_module not in TRAIN_FUNCTIONS:
        raise ValueError(
            f"Unknown train_module: {train_module}. "
            f"Available: {list(TRAIN_FUNCTIONS.keys())}"
        )

    TRAIN_FUNCTIONS[train_module](cfg)


if __name__ == "__main__":
    main()
