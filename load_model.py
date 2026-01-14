# from paddleocr import PaddleOCR
from ultralytics.models.sam import SAM3SemanticPredictor
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from pathlib import Path
import clip

# Patch for SimpleTokenizer issue in SAM3
try:
    # Try to patch the internal tokenizer if it exists
    # Based on traceback and search results, the class might be in tokenizer_ve
    from ultralytics.models.sam.sam3.tokenizer_ve import SimpleTokenizer as UltralyticsTokenizer
    
    def tokenizer_call(self, texts, context_length=77, truncate=False):
            return clip.tokenize(texts, context_length=context_length, truncate=truncate)
            
    UltralyticsTokenizer.__call__ = tokenizer_call
    print("✅ Patched Ultralytics SimpleTokenizer")
except ImportError:
    pass

try:
    # Also patch clip's SimpleTokenizer just in case
    from clip.simple_tokenizer import SimpleTokenizer
    
    def tokenizer_call_clip(self, texts, context_length=77, truncate=False):
            return clip.tokenize(texts, context_length=context_length, truncate=truncate)
            
    if not hasattr(SimpleTokenizer, '__call__'):
            SimpleTokenizer.__call__ = tokenizer_call_clip
            print("✅ Patched CLIP SimpleTokenizer")
except ImportError:
    print("⚠️ Could not import clip to patch tokenizer")



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
