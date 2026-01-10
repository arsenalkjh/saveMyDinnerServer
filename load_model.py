from paddleocr import PaddleOCR
from ultralytics.models.sam import SAM3SemanticPredictor


def load_ocr_engine():
    OCR_ENGINE = PaddleOCR(lang="korean", use_angle_cls=True)
    return OCR_ENGINE


def load_sam_model():
    overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=str("MY WEIGHTS PATH"),
    half=True,  # Use FP16 for faster inference
    save=True,
    )
    SAM_MODEL = SAM3SemanticPredictor(overrides=overrides)
    return SAM_MODEL

