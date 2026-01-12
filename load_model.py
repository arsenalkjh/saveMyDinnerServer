# from paddleocr import PaddleOCR
from ultralytics.models.sam import SAM3SemanticPredictor
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights" / "sam3.pt"

def load_ocr_engine():
    OCR_ENGINE = PaddleOCR(lang="korean", use_angle_cls=True,device = "gpu")
    return OCR_ENGINE


def load_sam_model():
    overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model=str(WEIGHTS_DIR),
    half=True,  # Use FP16 for faster inference
    save=True,
    )
    SAM_MODEL = SAM3SemanticPredictor(overrides=overrides)
    return SAM_MODEL

def load_varco_ocr():
    model_name = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
)
    processor = AutoProcessor.from_pretrained(model_name)
    return model , processor