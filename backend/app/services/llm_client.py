import json
import os
import re
from typing import Any, Generator, Optional
from openai import OpenAI
from app.config import DEFAULT_MODEL, SERVER_OPENAI_API_KEY, ALLOW_CLIENT_API_KEY


def resolve_api_key(client_key: Optional[str]) -> str:
    if client_key and ALLOW_CLIENT_API_KEY:
        return client_key.strip()
    return SERVER_OPENAI_API_KEY


def has_api_key(client_key: Optional[str]) -> bool:
    return bool(resolve_api_key(client_key))


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _mock_text(prompt: str, kind: str = "answer") -> str:
    if kind == "test":
        return "Model connection mock success. Add an OpenAI API key for live responses."
    if kind == "profile":
        return json.dumps({
            "candidate_summary": "Candidate profile summary will be generated here when an API key is provided.",
            "key_skills": ["Cloud", "DevOps", "Automation", "Troubleshooting"],
            "project_examples": ["Recent project from resume context"],
            "job_focus": ["Role-specific technical skills", "production support", "communication"],
            "answer_style_guidance": "Use concise STAR-style answers with tools, issue, action, and result."
        })
    if kind == "detect":
        return json.dumps({
            "is_interview_question": True,
            "clean_question": "Can you explain your recent project and your responsibilities?",
            "question_type": "project",
            "topic": "recent project",
            "difficulty": "medium"
        })
    return """# 30-Second Version\nI would answer by connecting the question to my real project experience using Tool + Project + Issue + Action + Result.\n\n# Real-Time Example\nIn my recent project, I handled production issues by checking logs, isolating the component, applying a fix, and documenting root cause.\n\n# Strong Answer\nI focus on what I personally owned, how I troubleshot the issue, and what business impact it had.\n\n# Key Points to Mention\n- Tools from resume and JD\n- Specific project responsibility\n- Troubleshooting steps\n- Team communication\n- Business impact\n\n# Possible Follow-Up Questions\n- What exact tool or command did you use?\n- How did you confirm the issue was resolved?\n- What did you do to prevent recurrence?"""


def responses_text(prompt: str, system: str, api_key: Optional[str] = None, model: Optional[str] = None, kind: str = "answer") -> str:
    key = resolve_api_key(api_key)
    if not key:
        return _mock_text(prompt, kind)
    selected_model = model or DEFAULT_MODEL
    client = _client(key)
    resp = client.responses.create(
        model=selected_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text


def responses_stream(prompt: str, system: str, api_key: Optional[str] = None, model: Optional[str] = None, kind: str = "answer") -> Generator[str, None, None]:
    """Stream response tokens as a generator. Falls back to mock if no key."""
    key = resolve_api_key(api_key)
    if not key:
        # Yield mock text in chunks to simulate streaming
        mock = _mock_text(prompt, kind)
        for i in range(0, len(mock), 20):
            yield mock[i:i+20]
        return
    selected_model = model or DEFAULT_MODEL
    client = _client(key)
    stream = client.responses.create(
        model=selected_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta
    stream.close()


def responses_json(prompt: str, system: str, api_key: Optional[str] = None, model: Optional[str] = None, kind: str = "profile") -> dict[str, Any]:
    text = responses_text(prompt, system + "\nReturn valid JSON only.", api_key, model, kind=kind)
    return extract_json(text)


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"raw": text, "parse_warning": "Could not parse strict JSON."}
