from pathlib import Path

# from paddleocr import PaddleOCR
from ultralytics.models.sam import SAM3SemanticPredictor

from ocr.ocr_inference import run_ocr_with_rotations
from ocr.detect_ingredients import run_sam
from ocr.qwen_model import postprocessing_with_vlm,clean_ocr_with_llm
from ocr.varco_ocr import run_varco_ocr

def _dedupe_keep_order(items):
    """리스트 아이템 중복제거 함수"""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def run_ocr_pipeline(
        # ocr_engine,
        varco_ocr_model,
        varco_ocr_processor,
        sam_model,
        image_path,
        model_name_llm = "Qwen/Qwen3-1.7B",
        model_name_vlm = "Qwen/Qwen3-VL-2B-Instruct",
        # rotations = [0,90,180,270]
):
    print("step1 OCR")
    # raw_texts = run_ocr_with_rotations(ocr_engine, image_path, rotations=rotations)
    raw_texts = run_varco_ocr(image_path = image_path , model = varco_ocr_model , processor = varco_ocr_processor)
    print(raw_texts)

    print("Step2 TEXT LLM 후처리")
    llm_items = clean_ocr_with_llm(raw_texts, model_name = model_name_llm)
    if llm_items is None:
        llm_items = []
    print(llm_items)

    print("Step3 SAM_MODEL")
    sam_result = run_sam(image_path=image_path, model=sam_model)
    print(sam_result)

    print("Step4 VLM 후처리")
    vlm_items = postprocessing_with_vlm(sam_item=sam_result, model_name=model_name_vlm)
    
    final_result = [*llm_items , *vlm_items]

    cleaned_list = [item.strip().lower() for item in final_result]

    return _dedupe_keep_order(cleaned_list)

