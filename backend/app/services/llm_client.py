import json
import os
import re
from typing import Any, Optional
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
    return """Direct answer: I would answer by connecting the question to my real project experience, the tools I used, the issue I handled, the action I took, and the result/business impact.\n\nDetailed answer:\nIn my recent project, I worked on role-relevant responsibilities using the tools listed in my resume and the job description. When facing production or delivery issues, I first clarified the impact, checked logs/metrics/pipeline or system status, isolated the failing component, coordinated with developers/QA/business teams, and applied the safest fix or rollback. After resolution, I documented the root cause and added preventive improvements such as better monitoring, validation, automation, or runbooks.\n\nShort answer to say:\nI usually explain it using Tool + Project + Issue + Action + Result. I focus on what I personally owned, how I troubleshot the issue, and what business impact it had.\n\nKey points to mention:\n- Tools from resume and JD\n- Specific project responsibility\n- Troubleshooting steps\n- Team communication\n- Business impact\n\nPossible follow-ups:\n- What exact tool or command did you use?\n- How did you confirm the issue was resolved?\n- What did you do to prevent recurrence?"""


def responses_text(prompt: str, system: str, api_key: Optional[str] = None, model: Optional[str] = None, kind: str = "answer") -> str:
    key = resolve_api_key(api_key)
    if not key:
        return _mock_text(prompt, kind)
    selected_model = model or DEFAULT_MODEL
    client = _client(key)
    # Use Responses API. Keep params conservative for compatibility.
    resp = client.responses.create(
        model=selected_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text


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
