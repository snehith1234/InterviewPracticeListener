from app.services.llm_client import responses_json, responses_text


def build_profile(role: str, job_description: str, resume_text: str, company_context: str, api_key: str | None, model: str | None) -> dict:
    prompt = f"""
Analyze this candidate context for interview answer coaching.

Role/title:
{role}

Job description:
{job_description}

Company/domain/context:
{company_context or 'Not provided'}

Candidate resume:
{resume_text[:20000]}

Return JSON with:
- candidate_summary
- key_skills
- role_requirements
- matched_skills
- missing_or_weak_areas
- project_examples
- domain_context
- answer_style_guidance
- safe_assumptions
"""
    return responses_json(
        prompt,
        system="You are a technical interview coach. Analyze resume/JD and create a concise candidate profile for answer generation. Do not invent unsupported experience.",
        api_key=api_key,
        model=model,
        kind="profile",
    )


def detect_question(transcript: str, api_key: str | None, model: str | None) -> dict:
    prompt = f"""
From this transcript, detect the latest clear interview question.

Transcript:
{transcript[-6000:]}

Return JSON:
{{
  "is_interview_question": true/false,
  "clean_question": "latest clear question only",
  "question_type": "intro|project|technical|scenario|behavioral|closing|other",
  "topic": "short topic",
  "difficulty": "easy|medium|hard",
  "confidence": 0.0
}}
"""
    return responses_json(
        prompt,
        system="You extract interviewer questions from transcripts. Return JSON only.",
        api_key=api_key,
        model=model,
        kind="detect",
    )


def generate_answer(role: str, job_description: str, resume_text: str, company_context: str, profile: dict, question: str, mode: str, api_key: str | None, model: str | None) -> str:
    prompt = f"""
The user is practicing for an interview. Generate a resume/JD-aligned answer for the detected interview question.

IMPORTANT ETHICAL BOUNDARY:
This is for mock interviews, practice sessions, or situations where AI assistance is allowed. Do not frame this as secret real-interview cheating.

Role/title:
{role}

Job description:
{job_description[:12000]}

Company/domain/context:
{company_context or 'Not provided'}

Candidate profile analysis:
{profile}

Resume text:
{resume_text[:18000]}

Interview question:
{question}

Mode:
{mode}

Rules:
1. Start with a direct answer.
2. Align to resume and JD.
3. Do not invent unsupported experience. If something is assumed, phrase it as a reasonable way to answer.
4. Use practical project language.
5. Mention tools where relevant.
6. Include production/project context and business impact.
7. For senior roles, use Tool + Project + Issue + Action + Result.
8. Avoid overly textbook language.
9. Keep it interview-speak: natural, confident, not too long.

Return in this format:
# Strong Answer

# 30-Second Version

# Key Points to Mention

# Resume/JD Alignment

# Possible Follow-Up Questions

# Follow-Up Answer Hints
"""
    return responses_text(
        prompt,
        system="You are an interview answer coach for practice sessions. Generate strong but truthful candidate answers.",
        api_key=api_key,
        model=model,
        kind="answer",
    )


def evaluate_user_answer(question: str, user_answer: str, role: str, job_description: str, profile: dict, api_key: str | None, model: str | None) -> str:
    prompt = f"""
Evaluate the user's practice answer and rewrite it stronger.

Role:
{role}

Job description:
{job_description[:12000]}

Candidate profile:
{profile}

Question:
{question}

User answer:
{user_answer}

Return:
# Score
Give score out of 10.

# What Was Good

# What Was Missing

# Stronger Version

# Short Version to Memorize

# Next Follow-Up to Practice
"""
    return responses_text(
        prompt,
        system="You are a technical interview coach. Give practical feedback and a better answer.",
        api_key=api_key,
        model=model,
        kind="answer",
    )
