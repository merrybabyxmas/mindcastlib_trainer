"""데이터셋 로드 및 전처리 모듈

HuggingFace 데이터셋 (merrybabyxmas/MindCastDataset)을 자동으로 로드합니다.
로컬 파일이 있으면 로컬을 우선 사용합니다.
"""
import os
import pandas as pd
from datasets import Dataset, load_dataset

# HuggingFace 데이터셋 ID
HF_DATASET_ID = "merrybabyxmas/MindCastDataset"

# 라벨 매핑
SENTIMENT_LABEL_MAP = {
    "분노": 0,
    "슬픔": 1,
    "불안": 2,
    "상처": 3,
    "당황": 4,
    "기쁨": 5,
}

TOPIC_LABEL_MAP = {
    "사회": 0,
    "정치": 1,
    "생활문화": 2,
    "세계": 3,
    "경제": 4,
    "IT과학": 5,
    "스포츠": 6,
}

SARCASM_LABEL_MAP = {
    "Non-sarcastic": 0,
    "Sarcastic": 1,
    False: 0,
    True: 1,
    0: 0,
    1: 1,
    "false": 0,
    "true": 1,
    "False": 0,
    "True": 1,
}


def _get_label_map(task_type: str) -> dict:
    """태스크별 라벨 매핑 반환"""
    return {
        "SC": SENTIMENT_LABEL_MAP,
        "SD": SARCASM_LABEL_MAP,
        "TC": TOPIC_LABEL_MAP,
    }.get(task_type, {})


def _apply_label_mapping(df: pd.DataFrame, task_type: str, label_col: str = "label") -> pd.DataFrame:
    """라벨 매핑 적용 및 검증"""
    label_map = _get_label_map(task_type)

    if label_map:
        df[label_col] = df[label_col].map(label_map)
        before = len(df)
        df = df.dropna(subset=[label_col])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[WARN] Dropped {dropped} invalid {task_type} label rows")
        df[label_col] = df[label_col].astype(int)
    else:
        df[label_col] = df[label_col].astype("category").cat.codes

    return df


def load_from_huggingface(task_type: str, split: str) -> Dataset:
    """
    HuggingFace에서 데이터셋 로드

    Args:
        task_type: 작업 유형 (SC, SD, TC)
        split: 데이터 분할 (train, test, validation)

    Returns:
        HuggingFace Dataset
    """
    # HuggingFace 데이터셋 config 이름 매핑
    config_map = {
        "SC": "sentiment",
        "SD": "sarcasm",
        "TC": "topic",
    }

    config_name = config_map.get(task_type)
    if not config_name:
        raise ValueError(f"Unknown task_type: {task_type}")

    print(f"[DATA] Loading {config_name}/{split} from HuggingFace ({HF_DATASET_ID})...")

    ds = load_dataset(HF_DATASET_ID, config_name, split=split)

    # 라벨 매핑 적용 (pandas로 변환해서 처리)
    label_map = _get_label_map(task_type)

    df = ds.to_pandas()
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    ds = Dataset.from_pandas(df, preserve_index=False)

    print(f"[DATA] Loaded {len(ds)} samples from {config_name}/{split}")

    return ds


def load_csv_dataset(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
    task_type: str = "SC"
) -> Dataset:
    """
    CSV 파일에서 데이터셋 로드 (로컬 파일용)

    Args:
        path: CSV 파일 경로
        text_col: 텍스트 컬럼명
        label_col: 라벨 컬럼명
        task_type: 작업 유형 (SC, SD, TC)

    Returns:
        HuggingFace Dataset
    """
    df = pd.read_csv(path)
    df = df[[text_col, label_col]]

    # NaN 제거
    before = len(df)
    df = df.dropna(subset=[label_col])
    after = len(df)
    if before != after:
        print(f"[WARN] Dropped {before - after} rows with NaN labels from {path}")

    df = _apply_label_mapping(df, task_type, label_col)

    return Dataset.from_pandas(df)


def load_dataset_for_task(
    task_type: str,
    split: str,
    local_path: str = None
) -> Dataset:
    """
    태스크별 데이터셋 로드 (로컬 우선, 없으면 HuggingFace에서 다운로드)

    Args:
        task_type: 작업 유형 (SC, SD, TC)
        split: 데이터 분할 (train, test, validation)
        local_path: 로컬 CSV 파일 경로 (선택)

    Returns:
        HuggingFace Dataset
    """
    # 로컬 파일이 있으면 로컬 사용
    if local_path and os.path.exists(local_path):
        print(f"[DATA] Loading from local file: {local_path}")
        return load_csv_dataset(local_path, task_type=task_type)

    # 없으면 HuggingFace에서 다운로드
    return load_from_huggingface(task_type, split)


def load_sentiment_data(split: str = "train", local_paths: list = None) -> Dataset:
    """
    감정 분류 데이터 로드

    Args:
        split: train, test, validation
        local_paths: 로컬 CSV 파일 경로 리스트 (선택)
    """
    if local_paths:
        # 로컬 파일들이 모두 존재하는 경우만 로컬 사용
        if all(os.path.exists(p) for p in local_paths):
            from datasets import concatenate_datasets
            datasets = [load_csv_dataset(p, task_type="SC") for p in local_paths]
            return concatenate_datasets(datasets)

    return load_from_huggingface("SC", split)


def load_sarcasm_data(split: str = "train", local_path: str = None) -> Dataset:
    """비꼬기 탐지 데이터 로드"""
    return load_dataset_for_task("SD", split, local_path)


def load_topic_data(split: str = "train", local_path: str = None) -> Dataset:
    """주제 분류 데이터 로드"""
    return load_dataset_for_task("TC", split, local_path)
