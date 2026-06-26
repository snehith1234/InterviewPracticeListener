from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.config import DEFAULT_MODEL
from app.services.coach_service import build_profile, detect_question, generate_answer, generate_answer_stream, detect_and_answer_stream, quick_short_answer_stream, evaluate_user_answer
from app.services.llm_client import responses_text

router = APIRouter()

class ProfileRequest(BaseModel):
    role: str
    job_description: str
    resume_text: str
    company_context: str = ""
    model: Optional[str] = None

class DetectRequest(BaseModel):
    transcript: str
    model: Optional[str] = None

class AnswerRequest(BaseModel):
    role: str
    job_description: str
    resume_text: str
    company_context: str = ""
    profile: dict = {}
    question: str
    mode: str = "practice"
    model: Optional[str] = None

class QuickAnswerRequest(BaseModel):
    role: str
    job_description: str
    resume_text: str
    company_context: str = ""
    profile: dict = {}
    transcript: str
    mode: str = "practice"
    model: Optional[str] = None

class EvaluateRequest(BaseModel):
    role: str
    job_description: str
    profile: dict = {}
    question: str
    user_answer: str
    model: Optional[str] = None

class TestRequest(BaseModel):
    model: Optional[str] = None


def _key(header_key: Optional[str]) -> Optional[str]:
    return header_key.strip() if header_key else None

@router.post("/profile")
def profile(req: ProfileRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    if not req.resume_text.strip() or not req.job_description.strip():
        raise HTTPException(status_code=400, detail="Resume text and job description are required.")
    return build_profile(req.role, req.job_description, req.resume_text, req.company_context, _key(x_openai_api_key), req.model or DEFAULT_MODEL)

@router.post("/detect-question")
def detect(req: DetectRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required.")
    return detect_question(req.transcript, _key(x_openai_api_key), req.model or DEFAULT_MODEL)

@router.post("/answer")
def answer(req: AnswerRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    content = generate_answer(req.role, req.job_description, req.resume_text, req.company_context, req.profile, req.question, req.mode, _key(x_openai_api_key), req.model or DEFAULT_MODEL)
    return {"answer": content}

@router.post("/answer-stream")
def answer_stream(req: AnswerRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    """Stream answer tokens via Server-Sent Events for low-latency display."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    gen = generate_answer_stream(req.role, req.job_description, req.resume_text, req.company_context, req.profile, req.question, req.mode, _key(x_openai_api_key), req.model or DEFAULT_MODEL)
    def event_stream():
        for chunk in gen:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/quick-answer")
def quick_answer(req: QuickAnswerRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    """Combined detect+answer in one streamed call. Skips separate detect round-trip."""
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required.")
    gen = detect_and_answer_stream(req.role, req.job_description, req.resume_text, req.company_context, req.profile, req.transcript, req.mode, _key(x_openai_api_key), req.model or DEFAULT_MODEL)
    def event_stream():
        for chunk in gen:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/quick-short")
def quick_short(req: QuickAnswerRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    """Ultra-fast 2-sentence answer for immediate display while full answer loads."""
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is required.")
    gen = quick_short_answer_stream(req.role, req.job_description, req.resume_text, req.company_context, req.profile, req.transcript, _key(x_openai_api_key), req.model or DEFAULT_MODEL)
    def event_stream():
        for chunk in gen:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/evaluate")
def evaluate(req: EvaluateRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    if not req.question.strip() or not req.user_answer.strip():
        raise HTTPException(status_code=400, detail="Question and user answer are required.")
    content = evaluate_user_answer(req.question, req.user_answer, req.role, req.job_description, req.profile, _key(x_openai_api_key), req.model or DEFAULT_MODEL)
    return {"feedback": content}

@router.post("/test-llm")
def test_llm(req: TestRequest, x_openai_api_key: Optional[str] = Header(default=None)):
    try:
        text = responses_text(
            "Say: Model connection successful.",
            "You are a connection test. Reply in one short sentence.",
            api_key=_key(x_openai_api_key),
            model=req.model or DEFAULT_MODEL,
            kind="test",
        )
        return {"ok": True, "model": req.model or DEFAULT_MODEL, "message": text}
    except Exception as exc:
        return {"ok": False, "model": req.model or DEFAULT_MODEL, "error": str(exc)}
