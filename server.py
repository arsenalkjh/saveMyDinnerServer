from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import shutil
from contextlib import asynccontextmanager

from ocr.main import run_ocr_pipeline




from load_model import load_ocr_engine
from load_model import load_sam_model
from load_model import load_varco_ocr

ocr_engine = None
sam3_model = None
varco_ocr_model = None
varco_ocr_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_engine, sam3_model, varco_ocr_model , varco_ocr_processor

    # print("🚀 Loading OCR engine...")
    # ocr_engine = load_ocr_engine()

    print("🚀 Loading Varco OCR")

    varco_ocr_model ,varco_ocr_processor = load_varco_ocr()
    print("🧠 Loading SAM3 model...")
    sam3_model = load_sam_model()

    print("✅ All models loaded")
    yield

    print("🧹 Server shutting down...")

app = FastAPI(lifespan=lifespan)



@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr_api(file: UploadFile = File(...)):
    image_path = f"/tmp/{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_ocr_pipeline(
        # ocr_engine = ocr_engine, 
        varco_ocr_model = varco_ocr_model,
        varco_ocr_processor = varco_ocr_processor,
        sam_model = sam3_model,
        image_path = image_path,
    )
    print({"result": result}) 
    return {"result": result}

