from typing import Generator
from app.services.llm_client import responses_json, responses_text, responses_stream


def build_profile(role: str, job_description: str, resume_text: str, company_context: str, additional_context: str, api_key: str | None, model: str | None) -> dict:
    additional_block = ""
    if additional_context and additional_context.strip():
        additional_block = f"""

Additional context provided by the candidate (work samples, project details, certifications, portfolio notes, etc.):
{additional_context[:20000]}
"""
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
{additional_block}
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


def generate_answer(role: str, job_description: str, resume_text: str, company_context: str, additional_context: str, profile: dict, question: str, mode: str, api_key: str | None, model: str | None) -> str:
    prompt = _build_answer_prompt(role, job_description, resume_text, company_context, additional_context, profile, question, mode)
    return responses_text(
        prompt,
        system="You are an expert interview answer coach. Generate answers that sound like a real person talking in an interview — natural, conversational, and confident. Never sound robotic or keyword-stuffed. Use plain language over corporate jargon.",
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
(Bullet list)

# What Was Missing
(Bullet list)

# Stronger Version
(Rewrite as natural conversational talking points — each "- " line is one thought the candidate would say out loud. Use natural connectors like "So...", "Then...", "For example...". Sound like a real person, not a keyword-stuffed bot.)

# Short Version to Memorize
(3-4 key sentences written naturally — something you could actually memorize and say.)

# Next Follow-Up to Practice
"""
    return responses_text(
        prompt,
        system="You are a technical interview coach. Give practical feedback and a better answer.",
        api_key=api_key,
        model=model,
        kind="answer",
    )


def _build_answer_prompt(role: str, job_description: str, resume_text: str, company_context: str, additional_context: str, profile: dict, question: str, mode: str) -> str:
    """Build the answer prompt, trimming context if profile exists."""
    additional_block = ""
    if additional_context and additional_context.strip():
        additional_block = f"""

IMPORTANT — Additional context provided by the candidate (their actual domain experience, work details, project notes, etc.). Use this to ground your answer accurately. Do NOT contradict this context:
{additional_context[:15000]}
"""
    if profile and profile.get("candidate_summary"):
        # Profile exists — use compact context instead of full resume+JD
        context_block = f"""
Candidate profile analysis:
{profile}

Role/title:
{role}

Key JD requirements (summarized from profile):
{profile.get('role_requirements', job_description[:4000])}

Company/domain/context:
{company_context or 'Not provided'}
{additional_block}"""
    else:
        # No profile — use full text (first-time flow)
        context_block = f"""
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
{additional_block}"""
    return f"""
The user is practicing for an interview. Generate a resume/JD-aligned answer for the detected interview question.

IMPORTANT ETHICAL BOUNDARY:
This is for mock interviews, practice sessions, or situations where AI assistance is allowed. Do not frame this as secret real-interview cheating.

SPEECH-TO-TEXT NOTE:
The question may come from voice transcription which often garbles technical terms. Interpret intelligently based on context:
- "our apps" / "are apps" likely means "rApps" (O-RAN)
- "ex app" / "X app" likely means "xApp" (O-RAN)
- "oh ran" / "o ran" means "O-RAN"
- "jane B" / "gene B" means "gNB"
- "cube control" / "cube CTL" means "kubectl"
- "terrace form" / "terraform" means "Terraform"
- "answer ball" / "answerable" means "Ansible"
- "doctor" in DevOps context means "Docker"
- "easy to" / "EC to" means "EC2"
- "see I see D" / "CICD" means "CI/CD"
- "AWS three" / "S three" means "S3"
- "lam da" means "Lambda"
Use the role, JD, and domain context to infer the correct technical term when transcription is ambiguous.

{context_block}

Interview question:
{question}

Mode:
{mode}

CRITICAL OUTPUT FORMAT — NATURAL CONVERSATIONAL TALKING POINTS:
Write every answer section as the candidate would ACTUALLY SPEAK in a real interview — natural, conversational, human.
- Each line starts with "- " and is ONE thought the person would say.
- Use natural connectors between points: "So what I usually do is...", "Then I...", "Another important thing I check is...", "For example, in one case I...", "What that gave us was..."
- Do NOT start every line with "I" — vary sentence openings naturally.
- Do NOT cram tools, metrics, and buzzwords into every single sentence. Spread them naturally across the answer.
- Do NOT use corporate jargon like "leveraging", "streamlining", "spearheading", "facilitating". Use plain language: "using", "improving", "leading", "helping".
- Do NOT list your skills or tools — weave them into the story of what you did.
- Sound like a real person explaining their work to a peer, not like a resume bullet point.
- A good test: read each line out loud. If it sounds robotic or rehearsed, rewrite it.

BAD example (robotic, keyword-stuffed):
- "I lead my QA teams to focus on thorough backend validation alongside UI and API tests to ensure full coverage of data consistency."
- "Leveraging Agile methodology and tools like Jira, Jenkins, and SQL Developer, I manage and track these validations efficiently while mentoring the team on best practices."

GOOD example (natural, conversational):
- "When I test a stored procedure, I first understand what tables it's supposed to read or modify and what the expected business result is."
- "Then I usually capture the data before execution, run the procedure with different test inputs, and query the affected tables afterward to make sure everything happened correctly."
- "Another important thing I test is transaction handling — for example, I'll intentionally provide invalid data and verify that the transaction rolls back completely."

Rules:
1. Start with a direct, natural answer to the question.
2. Align to resume and JD but do it subtly — don't announce alignment.
3. Do not invent unsupported experience. If something is assumed, phrase it as a reasonable way to answer.
4. Use practical, everyday project language — the way you'd explain it to a colleague.
5. Mention specific tools and techniques where natural, but don't force them into every sentence.
6. Include real-world context and outcomes when they add value, not as checkbox items.
7. Keep it conversational: natural, confident, like you're talking to the interviewer across a table.
8. DOMAIN EXPERTISE: If the company/role is in a specific domain, use domain terminology naturally — don't list it, weave it into your story.

Return in this format:
# 30-Second Version
(3-5 bullet points — conversational, as if giving a quick verbal summary)

# Real-Time Example
(A concrete story from the candidate's experience told naturally. Include what happened, what you did, and what the result was — but tell it like a story, not a checklist.)

# Strong Answer
(The full answer as natural conversational talking points. Each "- " line is one thought you'd speak. Use connectors between points. This should sound like a real person talking.)

# Key Points to Mention
(Short bullet reminders)

# Resume/JD Alignment
(Bullet list)

# Possible Follow-Up Questions
(Bullet list)

# Follow-Up Answer Hints
(Conversational talking points for each follow-up)
"""


def generate_answer_stream(role: str, job_description: str, resume_text: str, company_context: str, additional_context: str, profile: dict, question: str, mode: str, api_key: str | None, model: str | None) -> Generator[str, None, None]:
    """Stream answer tokens for low-latency perceived response."""
    prompt = _build_answer_prompt(role, job_description, resume_text, company_context, additional_context, profile, question, mode)
    return responses_stream(
        prompt,
        system="You are an expert interview answer coach. Generate answers that sound like a real person talking in an interview — natural, conversational, and confident. Never sound robotic or keyword-stuffed. Use plain language over corporate jargon.",
        api_key=api_key,
        model=model,
        kind="answer",
    )


def detect_and_answer_stream(role: str, job_description: str, resume_text: str, company_context: str, additional_context: str, profile: dict, transcript: str, mode: str, api_key: str | None, model: str | None) -> Generator[str, None, None]:
    """Combined: detect question from transcript AND generate answer in one LLM call (streamed)."""
    additional_block = ""
    if additional_context and additional_context.strip():
        additional_block = f"""

IMPORTANT — Additional context provided by the candidate (their actual domain experience, work details, project notes, etc.). Use this to ground your answer accurately. Do NOT contradict this context:
{additional_context[:15000]}
"""
    if profile and profile.get("candidate_summary"):
        context_block = f"""
Candidate profile analysis:
{profile}

Role/title:
{role}

Key JD requirements:
{profile.get('role_requirements', job_description[:4000])}

Company/domain/context:
{company_context or 'Not provided'}
{additional_block}"""
    else:
        context_block = f"""
Role/title:
{role}

Job description:
{job_description[:12000]}

Company/domain/context:
{company_context or 'Not provided'}

Resume text:
{resume_text[:18000]}
{additional_block}"""
    prompt = f"""
The user is in a mock interview practice session. Below is a transcript from the conversation. Your job:
1. Identify the latest clear interview question from the transcript.
2. Generate a strong practice answer aligned to the candidate's context.

IMPORTANT ETHICAL BOUNDARY:
This is for mock interviews, practice sessions, or situations where AI assistance is allowed.

SPEECH-TO-TEXT NOTE:
The transcript comes from voice recognition which often garbles technical terms. Interpret intelligently based on context:
- "our apps" / "are apps" likely means "rApps" (O-RAN)
- "ex app" / "X app" likely means "xApp" (O-RAN)
- "oh ran" / "o ran" means "O-RAN"
- "jane B" / "gene B" means "gNB"
- "cube control" / "cube CTL" means "kubectl"
- "terrace form" means "Terraform"
- "answer ball" / "answerable" means "Ansible"
- "doctor" in DevOps context means "Docker"
- "easy to" / "EC to" means "EC2"
- "see I see D" means "CI/CD"
- "AWS three" / "S three" means "S3"
- "lam da" means "Lambda"
Use the role, JD, and domain context to infer the correct technical term when transcription is ambiguous.

{context_block}

Transcript (latest portion):
{transcript[-6000:]}

Mode:
{mode}

Rules:
1. Start by stating the detected question clearly.
2. Then provide the answer the way a real person would speak it — natural, conversational, confident.
3. Do not invent unsupported experience.
4. Use everyday project language, not corporate jargon.
5. Mention tools and techniques naturally where they fit — don't force them into every sentence.
6. Include real-world context and outcomes when they add value.
7. Sound like a person explaining their work to a peer, not like a resume.
8. If the role is in a specific domain, weave domain knowledge naturally into the story.

CRITICAL OUTPUT FORMAT — NATURAL CONVERSATIONAL TALKING POINTS:
Write every answer section as the candidate would ACTUALLY SPEAK — natural, human, conversational.
- Each line starts with "- " and is ONE thought.
- Use natural connectors: "So what I did was...", "Then I...", "Another thing I noticed was...", "For example..."
- Do NOT start every line with "I" — vary sentence openings.
- Do NOT cram tools and buzzwords into every sentence.
- Do NOT use jargon like "leveraging", "streamlining", "spearheading". Use plain language.
- Sound like a real person, not a keyword-stuffed bot.

Return in this format:
# Detected Question
(The clear interview question you identified)

# 30-Second Version
(3-5 conversational talking points)

# Real-Time Example
(A concrete story told naturally — what happened, what you did, what the result was.)

# Strong Answer
(Full answer as natural conversational talking points. Each "- " line is one spoken thought with natural connectors.)

# Key Points to Mention
(Short bullet reminders)

# Possible Follow-Up Questions
(Bullet list)

# Follow-Up Answer Hints
(Conversational talking points for each follow-up)
"""
    return responses_stream(
        prompt,
        system="You are an expert interview answer coach. First identify the question from the transcript, then generate a natural, conversational practice answer. Sound like a real person talking, not a keyword-stuffed bot.",
        api_key=api_key,
        model=model,
        kind="answer",
    )


def quick_short_answer_stream(role: str, job_description: str, resume_text: str, company_context: str, additional_context: str, profile: dict, transcript: str, api_key: str | None, model: str | None) -> Generator[str, None, None]:
    """Ultra-fast first response: detect question + give ONLY a 2-sentence answer. Streams immediately."""
    additional_hint = ""
    if additional_context and additional_context.strip():
        additional_hint = f" Candidate's additional context: {additional_context[:3000]}."
    if profile and profile.get("candidate_summary"):
        context_block = f"Profile: {profile.get('candidate_summary', '')}. Skills: {', '.join(profile.get('key_skills', [])[:8])}. Style: {profile.get('answer_style_guidance', '')}.{additional_hint}"
    else:
        context_block = f"Role: {role}. Key JD: {job_description[:2000]}. Resume highlights: {resume_text[:2000]}.{additional_hint}"

    prompt = f"""
From this transcript, identify the interview question and give a SHORT 2-3 sentence answer the candidate can say immediately.

Note: Transcript is from voice recognition — interpret garbled terms using domain context (e.g., "our apps" = "rApps", "ex app" = "xApp", "oh ran" = "O-RAN", "terrace form" = "Terraform", "cube control" = "kubectl", "doctor" = "Docker" in DevOps).

Context: {context_block}

Transcript: {transcript[-3000:]}

Reply in EXACTLY this format (nothing else):
**Q:** [the detected question — use correct technical terms even if transcript garbled them]

**Quick Answer:**

[2-3 sentence answer they can start speaking right now]
"""
    return responses_stream(
        prompt,
        system="You give ultra-short interview answers. Be direct, no fluff. 2-3 sentences max.",
        api_key=api_key,
        model=model,
        kind="answer",
    )
