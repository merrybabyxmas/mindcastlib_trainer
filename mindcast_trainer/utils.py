"""유틸리티 함수 모듈"""
import evaluate


def get_metric(metric_name: str):
    """평가 메트릭 로드"""
    if metric_name in ("macro_f1", "f1"):
        return evaluate.load("f1")
    elif metric_name == "accuracy":
        return evaluate.load("accuracy")
    else:
        return evaluate.load(metric_name)
