"""트레이너 모듈 - 모델 학습 및 업로드"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import concatenate_datasets
from transformers import (
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from huggingface_hub import upload_folder, HfFolder, create_repo

from .dataset import load_from_huggingface, load_csv_dataset
from .model import build_model_with_lora
from .utils import get_metric
from .evaluation import evaluate_model, generate_model_card


class WeightedTrainer(Trainer):
    """클래스 가중치가 적용된 Trainer"""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            device_weights = torch.tensor(
                self.class_weights, device=logits.device, dtype=logits.dtype
            )
            loss_fct = nn.CrossEntropyLoss(weight=device_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _ensure_repo_exists(repo_id: str, token: str):
    """HuggingFace 레포지토리 존재 확인 및 생성"""
    try:
        create_repo(repo_id, token=token, private=False, exist_ok=True)
        print(f"[HF] Repo checked/created: https://huggingface.co/{repo_id}")
    except Exception as e:
        print("[HF] Repo creation error:", e)


def _tokenize_dataset(tokenizer, dataset):
    """데이터셋 토크나이징"""
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=256)

    ds = dataset.map(_tok, batched=True)
    return ds.remove_columns(["text"])


def _load_data(task_type: str, split: str, local_paths: list = None):
    """
    데이터 로드 (로컬 우선, 없으면 HuggingFace)

    Args:
        task_type: SC, SD, TC
        split: train, test, validation
        local_paths: 로컬 파일 경로 리스트
    """
    # 로컬 파일이 있으면 로컬 사용
    if local_paths:
        existing_paths = [p for p in local_paths if os.path.exists(p)]
        if existing_paths:
            print(f"[DATA] Loading from local files: {existing_paths}")
            datasets = [load_csv_dataset(p, task_type=task_type) for p in existing_paths]
            if len(datasets) == 1:
                return datasets[0]
            return concatenate_datasets(datasets)

    # 없으면 HuggingFace에서 다운로드
    return load_from_huggingface(task_type, split)


def _upload_final_model(
    cfg: dict,
    model_key: str,
    save_dir: str,
    test_results: dict = None,
    dataset_info: dict = None,
    usage_example: str = None
):
    """최종 학습된 모델을 HuggingFace에 업로드"""
    hf = cfg["huggingface_config"]
    if not hf["use_huggingface"]:
        return

    repo_id = cfg["huggingface_repos"][model_key]
    _ensure_repo_exists(repo_id, hf["token"])

    config_path = os.path.join(save_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)

        if "num_labels" not in config_data or config_data["num_labels"] is None:
            config_data["num_labels"] = len(config_data.get("id2label", {}))

        if "architectures" not in config_data:
            config_data["architectures"] = ["RobertaForSequenceClassification"]

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

        print(f"[HF] Config verified: num_labels={config_data.get('num_labels')}")
    else:
        print("[HF] config.json missing → creating new config")
        model_config = AutoConfig.from_pretrained(cfg["train_config"]["base_model_name"])
        task_type = "SC"
        model_config.id2label = cfg["task_configs"][task_type]["id2label"]
        model_config.label2id = cfg["task_configs"][task_type]["label2id"]
        model_config.num_labels = len(cfg["task_configs"][task_type]["id2label"])
        model_config.architectures = ["RobertaForSequenceClassification"]
        model_config.save_pretrained(save_dir)

    # 모델 카드 생성
    if test_results and dataset_info and usage_example:
        print("\n[HF] Generating model card...")

        model_descriptions = {
            "SC_only": (
                "Mindcast Emotion Classifier",
                "한국어 텍스트에서 6가지 감정(분노, 슬픔, 불안, 상처, 당황, 기쁨)을 분류하는 모델입니다."
            ),
            "SC_normal": (
                "Mindcast Emotion Classifier (Normal)",
                "한국어 일반 댓글에서 6가지 감정을 분류하는 모델입니다."
            ),
            "SC_sarc": (
                "Mindcast Emotion Classifier (Sarcastic)",
                "한국어 비꼬는 댓글(sarcasm)에서 6가지 감정을 분류하는 모델입니다."
            ),
            "SD": (
                "Mindcast Sarcasm Detector",
                "한국어 텍스트가 비꼬는 표현(sarcasm)인지 아닌지를 판별하는 이진 분류 모델입니다."
            ),
            "TC": (
                "Mindcast Topic Classifier",
                "한국어 텍스트의 주제를 분류하는 모델입니다."
            ),
        }

        model_name, task_desc = model_descriptions.get(
            model_key, (f"Mindcast {model_key}", "Text classification model")
        )

        model_card = generate_model_card(
            model_name=model_name,
            task_description=task_desc,
            cfg=cfg,
            test_results=test_results,
            dataset_info=dataset_info,
            usage_example=usage_example,
        )

        readme_path = os.path.join(save_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)

        print(f"[HF] Model card saved to {readme_path}")

    HfFolder.save_token(hf["token"])

    upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        token=hf["token"],
        commit_message="Upload final merged model with performance metrics",
    )

    print(f"[HF] Final merged model uploaded → https://huggingface.co/{repo_id}")


def _build_trainer(
    cfg: dict,
    model,
    tokenizer,
    train_ds,
    eval_ds,
    out_dir: str,
    class_weights=None
):
    """Trainer 객체 생성"""
    tc = cfg["train_config"]
    wc = cfg["wandb_config"]

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=tc["batch_size"],
        per_device_eval_batch_size=tc["batch_size"],
        learning_rate=tc["learning_rate"],
        num_train_epochs=tc["epochs"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=tc["save_total_limit"],
        logging_steps=tc["logging_steps"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        report_to=["wandb"] if wc["use_wandb"] else [],
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    metric = get_metric(tc["eval_metric"])

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)

        if tc["eval_metric"] == "macro_f1":
            return metric.compute(predictions=preds, references=labels, average="macro")

        return metric.compute(predictions=preds, references=labels)

    if class_weights is not None:
        return WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=compute,
        )
    else:
        return Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            compute_metrics=compute,
        )


# ============================================================
# train_SC_only (full sentiment classifier)
# ============================================================

def train_SC_only(cfg: dict):
    """통합 감정 분류 모델 학습 (HuggingFace sentiment 데이터 사용)"""
    print("\n========== Train SC_only ==========\n")

    # 1. 데이터 로드 (HuggingFace에서 자동 다운로드)
    local_train = cfg.get("data_config", {}).get("SC_only_train_dirs")
    local_test = cfg.get("data_config", {}).get("SC_only_test_dirs")

    full_train = _load_data("SC", "train", local_train)
    test_ds = _load_data("SC", "test", local_test)

    # Train/Valid 분할
    split = full_train.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"[DATA] Train: {len(train_ds)}, Valid: {len(eval_ds)}, Test: {len(test_ds)}")

    # 2. 모델 빌드
    tokenizer, model = build_model_with_lora(
        cfg["train_config"]["base_model_name"],
        num_labels=cfg["task_configs"]["SC"]["num_labels"],
        lora_cfg=cfg["train_config"],
        cfg=cfg,
        task_type="SC",
    )

    # 3. 토크나이징
    train_ds = _tokenize_dataset(tokenizer, train_ds)
    eval_ds = _tokenize_dataset(tokenizer, eval_ds)
    test_ds = _tokenize_dataset(tokenizer, test_ds)

    # 4. 학습
    sc_normal_weights = [0.168, 1.512, 4.033, 10.083, 0.903, 1.152]
    sc_sarc_weights = [0.262, 3.594, 6.889, 27.556, 0.929, 1.531]
    class_weights = np.mean([sc_normal_weights, sc_sarc_weights], axis=0).tolist()
    print(f"[TRAINING] Using averaged class weights: {class_weights}")

    trainer = _build_trainer(
        cfg, model, tokenizer, train_ds, eval_ds, "models/SC_only", class_weights
    )
    trainer.train()

    # 5. LoRA merge
    print("\n[MERGE] Merging LoRA adapter with base model...")
    model = model.merge_and_unload()

    # 6. Test 평가
    print("\n[TEST] Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        device=cfg["device"],
        id2label=cfg["task_configs"]["SC"]["id2label"],
    )

    # 7. 저장
    save_dir = "models/SC_only_merged"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"[SAVE] Merged model saved to {save_dir}")

    # 8. 업로드
    dataset_info = {
        "Train samples": f"{len(train_ds)}",
        "Valid samples": f"{len(eval_ds)}",
        "Test samples": f"{len(test_ds)}",
        "Number of labels": 6,
        "Labels": ", ".join(cfg["task_configs"]["SC"]["label_list"]),
    }

    usage_example = f"""from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load model
model_name = "{cfg['huggingface_repos']['SC_only']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Predict
text = "오늘 정말 기분이 좋아!"
result = classifier(text)
print(result)
"""

    _upload_final_model(cfg, "SC_only", save_dir, test_results, dataset_info, usage_example)

    return trainer


# ============================================================
# train_SC_normal
# ============================================================

def train_SC_normal(cfg: dict):
    """일반 댓글용 감정 분류 모델 학습"""
    print("\n========== Train SC_normal ==========\n")

    local_train = cfg.get("data_config", {}).get("SC_normal_train_dirs")
    local_test = cfg.get("data_config", {}).get("SC_normal_test_dirs")

    # HuggingFace sentiment는 normal/sarcastic 구분이 없으므로 전체 사용
    full_train = _load_data("SC", "train", local_train)
    test_ds = _load_data("SC", "test", local_test)

    split = full_train.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"[DATA] Train: {len(train_ds)}, Valid: {len(eval_ds)}, Test: {len(test_ds)}")

    tokenizer, model = build_model_with_lora(
        cfg["train_config"]["base_model_name"],
        num_labels=cfg["task_configs"]["SC"]["num_labels"],
        lora_cfg=cfg["train_config"],
        cfg=cfg,
        task_type="SC",
    )

    train_ds = _tokenize_dataset(tokenizer, train_ds)
    eval_ds = _tokenize_dataset(tokenizer, eval_ds)
    test_ds = _tokenize_dataset(tokenizer, test_ds)

    class_weights = [0.168, 1.512, 4.033, 10.083, 0.903, 1.152]
    print(f"[TRAINING] Using class weights: {class_weights}")

    trainer = _build_trainer(
        cfg, model, tokenizer, train_ds, eval_ds, "models/SC_normal", class_weights
    )
    trainer.train()

    print("\n[MERGE] Merging LoRA adapter with base model...")
    model = model.merge_and_unload()

    print("\n[TEST] Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        device=cfg["device"],
        id2label=cfg["task_configs"]["SC"]["id2label"],
    )

    save_dir = "models/SC_normal_merged"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    dataset_info = {
        "Train samples": f"{len(train_ds)}",
        "Valid samples": f"{len(eval_ds)}",
        "Test samples": f"{len(test_ds)}",
        "Number of labels": 6,
        "Labels": ", ".join(cfg["task_configs"]["SC"]["label_list"]),
    }

    usage_example = f"""from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "{cfg['huggingface_repos']['SC_normal']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "오늘 정말 행복한 하루였어요"
result = classifier(text)
print(result)
"""

    _upload_final_model(cfg, "SC_normal", save_dir, test_results, dataset_info, usage_example)
    return trainer


# ============================================================
# train_SC_sarc
# ============================================================

def train_SC_sarc(cfg: dict):
    """비꼬는 댓글용 감정 분류 모델 학습"""
    print("\n========== Train SC_sarc ==========\n")

    local_train = cfg.get("data_config", {}).get("SC_sarc_train_dirs")
    local_test = cfg.get("data_config", {}).get("SC_sarc_test_dirs")

    full_train = _load_data("SC", "train", local_train)
    test_ds = _load_data("SC", "test", local_test)

    split = full_train.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"[DATA] Train: {len(train_ds)}, Valid: {len(eval_ds)}, Test: {len(test_ds)}")

    tokenizer, model = build_model_with_lora(
        cfg["train_config"]["base_model_name"],
        num_labels=cfg["task_configs"]["SC"]["num_labels"],
        lora_cfg=cfg["train_config"],
        cfg=cfg,
        task_type="SC",
    )

    train_ds = _tokenize_dataset(tokenizer, train_ds)
    eval_ds = _tokenize_dataset(tokenizer, eval_ds)
    test_ds = _tokenize_dataset(tokenizer, test_ds)

    class_weights = [0.262, 3.594, 6.889, 27.556, 0.929, 1.531]
    print(f"[TRAINING] Using class weights: {class_weights}")

    trainer = _build_trainer(
        cfg, model, tokenizer, train_ds, eval_ds, "models/SC_sarc", class_weights
    )
    trainer.train()

    print("\n[MERGE] Merging LoRA adapter with base model...")
    model = model.merge_and_unload()

    print("\n[TEST] Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        device=cfg["device"],
        id2label=cfg["task_configs"]["SC"]["id2label"],
    )

    save_dir = "models/SC_sarc_merged"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    dataset_info = {
        "Train samples": f"{len(train_ds)}",
        "Valid samples": f"{len(eval_ds)}",
        "Test samples": f"{len(test_ds)}",
        "Number of labels": 6,
        "Labels": ", ".join(cfg["task_configs"]["SC"]["label_list"]),
    }

    usage_example = f"""from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "{cfg['huggingface_repos']['SC_sarc']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "아 진짜 최고다~ 오늘도 망했네"
result = classifier(text)
print(result)
"""

    _upload_final_model(cfg, "SC_sarc", save_dir, test_results, dataset_info, usage_example)
    return trainer


# ============================================================
# train_SD (sarcasm detector)
# ============================================================

def train_SD(cfg: dict):
    """비꼬기 탐지 모델 학습"""
    print("\n========== Train SD ==========\n")

    local_train = cfg.get("data_config", {}).get("SD_train_data_dir")
    local_test = cfg.get("data_config", {}).get("SD_test_data_dir")

    train_data = _load_data("SD", "train", [local_train] if local_train else None)
    test_ds = _load_data("SD", "test", [local_test] if local_test else None)

    split = train_data.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"[DATA] Train: {len(train_ds)}, Valid: {len(eval_ds)}, Test: {len(test_ds)}")

    tokenizer, model = build_model_with_lora(
        cfg["train_config"]["base_model_name"],
        num_labels=cfg["task_configs"]["SD"]["num_labels"],
        lora_cfg=cfg["train_config"],
        cfg=cfg,
        task_type="SD",
    )

    train_ds = _tokenize_dataset(tokenizer, train_ds)
    eval_ds = _tokenize_dataset(tokenizer, eval_ds)
    test_ds = _tokenize_dataset(tokenizer, test_ds)

    class_weights = [0.729, 1.593]
    print(f"[TRAINING] Using class weights: {class_weights}")

    trainer = _build_trainer(
        cfg, model, tokenizer, train_ds, eval_ds, "models/SD", class_weights
    )
    trainer.train()

    print("\n[MERGE] Merging LoRA adapter with base model...")
    model = model.merge_and_unload()

    print("\n[TEST] Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        device=cfg["device"],
        id2label=cfg["task_configs"]["SD"]["id2label"],
    )

    save_dir = "models/SD_merged"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    dataset_info = {
        "Train samples": f"{len(train_ds)}",
        "Valid samples": f"{len(eval_ds)}",
        "Test samples": f"{len(test_ds)}",
        "Number of labels": 2,
        "Labels": "Non-sarcastic, Sarcastic",
    }

    usage_example = f"""from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "{cfg['huggingface_repos']['SD']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "아 진짜 좋네요~ 완전 최고에요~"
result = classifier(text)
print(result)
"""

    _upload_final_model(cfg, "SD", save_dir, test_results, dataset_info, usage_example)
    return trainer


# ============================================================
# train_TC (topic classifier)
# ============================================================

def train_TC(cfg: dict):
    """주제 분류 모델 학습"""
    print("\n========== Train TC ==========\n")

    local_train = cfg.get("data_config", {}).get("TC_train_data_dir")
    local_test = cfg.get("data_config", {}).get("TC_test_data_dir")

    train_data = _load_data("TC", "train", [local_train] if local_train else None)
    test_ds = _load_data("TC", "test", [local_test] if local_test else None)

    split = train_data.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    num_labels = cfg["task_configs"]["TC"]["num_labels"]
    print(f"[DATA] Train: {len(train_ds)}, Valid: {len(eval_ds)}, Test: {len(test_ds)}, Labels: {num_labels}")

    tokenizer, model = build_model_with_lora(
        cfg["train_config"]["base_model_name"],
        num_labels=num_labels,
        lora_cfg=cfg["train_config"],
        cfg=cfg,
        task_type="TC",
    )

    train_ds = _tokenize_dataset(tokenizer, train_ds)
    eval_ds = _tokenize_dataset(tokenizer, eval_ds)
    test_ds = _tokenize_dataset(tokenizer, test_ds)

    trainer = _build_trainer(cfg, model, tokenizer, train_ds, eval_ds, "models/TC")
    trainer.train()

    print("\n[MERGE] Merging LoRA adapter with base model...")
    model = model.merge_and_unload()

    print("\n[TEST] Evaluating on test set...")
    test_results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_ds,
        device=cfg["device"],
        id2label=cfg["task_configs"]["TC"]["id2label"],
    )

    save_dir = "models/TC_merged"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    tc_id2label = cfg["task_configs"]["TC"]["id2label"]
    label_names = ", ".join(
        [tc_id2label[str(i)] for i in sorted([int(k) for k in tc_id2label.keys()])]
    )

    dataset_info = {
        "Train samples": f"{len(train_ds)}",
        "Valid samples": f"{len(eval_ds)}",
        "Test samples": f"{len(test_ds)}",
        "Number of labels": num_labels,
        "Labels": label_names,
    }

    usage_example = f"""from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "{cfg['huggingface_repos']['TC']}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = "오늘 날씨가 정말 좋네요"
result = classifier(text)
print(result)
"""

    _upload_final_model(cfg, "TC", save_dir, test_results, dataset_info, usage_example)
    return trainer


# ============================================================
# Pipeline: SD → SC_normal → SC_sarc
# ============================================================

def train_SD_SC(cfg: dict):
    """SD + SC 파이프라인 전체 학습"""
    print("\n========== Train Pipeline: SD → SC_normal → SC_sarc ==========\n")

    print("\n[STEP 1/4] Training Sarcasm Detector (SD)...")
    train_SD(cfg)

    print("\n[STEP 2/4] Training Normal Sentiment Classifier (SC_normal)...")
    train_SC_normal(cfg)

    print("\n[STEP 3/4] Training Sarcastic Sentiment Classifier (SC_sarc)...")
    train_SC_sarc(cfg)

    # 통합 테스트
    print("\n[STEP 4/4] Integrated Test: Evaluating combined performance...")
    print("=" * 70)

    device = cfg["device"]

    # 테스트 데이터 로드
    test_ds = _load_data("SC", "test", None)

    # SC_normal 통합 테스트
    print(f"\n[SC_NORMAL] Loading model from models/SC_normal_merged...")
    tokenizer_norm = AutoTokenizer.from_pretrained("models/SC_normal_merged")
    model_norm = AutoModelForSequenceClassification.from_pretrained("models/SC_normal_merged")
    test_tok = _tokenize_dataset(tokenizer_norm, test_ds)

    test_results_norm = evaluate_model(
        model=model_norm,
        tokenizer=tokenizer_norm,
        test_dataset=test_tok,
        device=device,
        id2label=cfg["task_configs"]["SC"]["id2label"],
    )

    print(f"\n[SC_NORMAL] Integrated Test Results:")
    print(f"  - Accuracy: {test_results_norm['accuracy']:.4f}")
    print(f"  - F1 (macro): {test_results_norm['f1_macro']:.4f}")

    # SC_sarc 통합 테스트
    print(f"\n[SC_SARC] Loading model from models/SC_sarc_merged...")
    tokenizer_sarc = AutoTokenizer.from_pretrained("models/SC_sarc_merged")
    model_sarc = AutoModelForSequenceClassification.from_pretrained("models/SC_sarc_merged")
    test_tok = _tokenize_dataset(tokenizer_sarc, test_ds)

    test_results_sarc = evaluate_model(
        model=model_sarc,
        tokenizer=tokenizer_sarc,
        test_dataset=test_tok,
        device=device,
        id2label=cfg["task_configs"]["SC"]["id2label"],
    )

    print(f"\n[SC_SARC] Integrated Test Results:")
    print(f"  - Accuracy: {test_results_sarc['accuracy']:.4f}")
    print(f"  - F1 (macro): {test_results_sarc['f1_macro']:.4f}")

    # 최종 요약
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - SD_SC Pipeline")
    print("=" * 70)

    print("\n[1] Sarcasm Detector (SD)")
    print("    - Check models/SD_merged for results")

    print("\n[2] Normal Sentiment Classifier (SC_normal)")
    print(f"    - Integrated Test Accuracy: {test_results_norm['accuracy']:.4f}")
    print(f"    - Integrated Test F1: {test_results_norm['f1_macro']:.4f}")

    print("\n[3] Sarcastic Sentiment Classifier (SC_sarc)")
    print(f"    - Integrated Test Accuracy: {test_results_sarc['accuracy']:.4f}")
    print(f"    - Integrated Test F1: {test_results_sarc['f1_macro']:.4f}")

    print("\n" + "=" * 70)
    print("[Pipeline Completed] All models trained, tested, and uploaded!")
    print("=" * 70 + "\n")
