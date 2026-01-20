"""모델 빌드 모듈"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def build_model_with_lora(
    base_model_name: str,
    num_labels: int,
    lora_cfg: dict,
    cfg: dict,
    task_type: str = "SC"
):
    """
    LoRA가 적용된 분류 모델 생성

    Args:
        base_model_name: 사전 학습된 모델 이름
        num_labels: 분류할 클래스 수
        lora_cfg: LoRA 설정 (train_config에서 추출)
        cfg: 전체 설정 (task_configs 포함)
        task_type: 작업 유형 (SC, SD, TC)

    Returns:
        (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels
    )

    # LoRA 적용
    lora = LoraConfig(
        r=lora_cfg["lora_r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora)

    # 라벨 매핑 설정
    task_config = cfg["task_configs"][task_type]
    id2label = task_config["id2label"]
    label2id = task_config["label2id"]

    model.config.id2label = {int(i): v for i, v in id2label.items()}
    model.config.label2id = {k: int(v) for k, v in label2id.items()}
    model.config.architectures = cfg["architectures"]

    print(f"\n[CONFIG] Task: {task_type}")
    print("[CONFIG] id2label:", model.config.id2label)
    print("[CONFIG] architectures:", model.config.architectures)

    return tokenizer, model
