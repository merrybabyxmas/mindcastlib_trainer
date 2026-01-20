"""평가 및 모델 카드 생성 모듈"""
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, tokenizer, test_dataset, device, id2label=None) -> dict:
    """
    모델을 test dataset으로 평가하고 성능 지표 반환

    Args:
        model: 학습된 모델
        tokenizer: 토크나이저
        test_dataset: 테스트 데이터셋
        device: 디바이스
        id2label: 라벨 매핑 딕셔너리

    Returns:
        dict: accuracy, f1_macro, f1_weighted, confusion_matrix, classification_report
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []

    print(f"\n[EVAL] Starting evaluation on {len(test_dataset)} samples...")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=data_collator,
        shuffle=False
    )

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    labels = None
    target_names = None
    if id2label:
        labels = sorted(id2label.keys())
        target_names = [id2label[i] for i in labels]

    report = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print(f"\n[EVAL] Accuracy: {accuracy:.4f}")
    print(f"[EVAL] F1 (macro): {f1_macro:.4f}")
    print(f"[EVAL] F1 (weighted): {f1_weighted:.4f}")
    print(f"\n[EVAL] Confusion Matrix:\n{cm}")
    print(f"\n[EVAL] Classification Report:\n{report}")

    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def generate_model_card(
    model_name: str,
    task_description: str,
    cfg: dict,
    test_results: dict,
    dataset_info: dict,
    usage_example: str
) -> str:
    """
    HuggingFace 모델 카드 자동 생성

    Args:
        model_name: 모델 이름
        task_description: 태스크 설명
        cfg: 설정
        test_results: evaluate_model 결과
        dataset_info: 데이터셋 정보
        usage_example: 사용 예제 코드

    Returns:
        str: README.md 내용
    """
    cm = np.array(test_results['confusion_matrix'])
    cm_str = "```\n" + str(cm) + "\n```"

    train_cfg = cfg['train_config']
    hyperparams = f"""
| Hyperparameter | Value |
|---|---|
| Base Model | `{train_cfg['base_model_name']}` |
| Batch Size | {train_cfg['batch_size']} |
| Epochs | {train_cfg['epochs']} |
| Learning Rate | {train_cfg['learning_rate']} |
| Warmup Ratio | {train_cfg['warmup_ratio']} |
| Weight Decay | {train_cfg['weight_decay']} |
| LoRA r | {train_cfg['lora_r']} |
| LoRA alpha | {train_cfg['lora_alpha']} |
| LoRA dropout | {train_cfg['lora_dropout']} |
"""

    dataset_str = ""
    for key, value in dataset_info.items():
        dataset_str += f"- **{key}**: {value}\n"

    model_card = f"""---
language: ko
license: apache-2.0
tags:
- text-classification
- korean
- emotion-classification
- sentiment-analysis
datasets:
- custom
metrics:
- accuracy
- f1
widget:
- text: "오늘 정말 기분이 좋아!"
---

# {model_name}

## Model Description

{task_description}

이 모델은 LoRA (Low-Rank Adaptation)를 사용하여 효율적으로 파인튜닝되었으며, 최종적으로 base model과 merge되어 배포되었습니다.

**Training Date**: {datetime.now().strftime("%Y-%m-%d")}

## Performance

### Test Set Results

| Metric | Score |
|---|---|
| **Accuracy** | **{test_results['accuracy']:.4f}** |
| **F1 Score (Macro)** | **{test_results['f1_macro']:.4f}** |
| **F1 Score (Weighted)** | **{test_results['f1_weighted']:.4f}** |

### Confusion Matrix

{cm_str}

### Detailed Classification Report

```
{test_results['classification_report']}
```

## Training Details

### Hyperparameters

{hyperparams}

### Training Data

{dataset_str}

## Usage

### Installation

```bash
pip install transformers torch
```

### Quick Start

```python
{usage_example}
```

## Model Architecture

- **Base Model**: {train_cfg['base_model_name']}
- **Task**: Sequence Classification
- **Number of Labels**: {cfg.get('task_config', {}).get('num_labels', 'N/A')}

## Citation

If you use this model, please cite:

```bibtex
@misc{{mindcast-model,
  author = {{Mindcast Team}},
  title = {{{model_name}}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{cfg['huggingface_repos'].get('SC_only', 'model-repo')}}}}},
}}
```

## Contact

For questions or feedback, please open an issue on the model repository.

---

*This model card was automatically generated.*
"""

    return model_card
