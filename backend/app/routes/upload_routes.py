import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR
from app.services.file_parser import extract_text

router = APIRouter()
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/resume")
async def upload_resume(file: UploadFile = File(...)):
    lower = file.filename.lower()
    if not lower.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, or TXT files are supported.")
    safe_name = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        text = extract_text(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not parse resume: {exc}")
    return {"filename": safe_name, "text": text, "characters": len(text)}
