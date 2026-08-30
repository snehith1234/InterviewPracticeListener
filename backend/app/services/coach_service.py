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
(Rewrite as natural short paragraphs — 2-4 sentences each, grouped by thought. Should sound like a real person talking in an interview, not a keyword-stuffed script. Use natural transitions between paragraphs.)

# Short Version to Memorize
(3-4 sentences written naturally — something you could actually say from memory.)

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

CRITICAL OUTPUT FORMAT — NATURAL SPOKEN ANSWER:
Write every answer section as the candidate would ACTUALLY SPEAK in a real interview — natural, conversational, human.
Format as SHORT PARAGRAPHS (2-4 sentences each), not as single-line bullet points. Each paragraph is one coherent thought or mini-topic the person would say together before pausing. This is how people actually speak — they group related ideas, not list them one per line.

STYLE RULES:
- Use natural transitions between paragraphs: "In my recent projects...", "From a Test Lead perspective...", "For example...", "Most of my experience has been in...", "So I believe..."
- Do NOT map each paragraph to one JD keyword. Group related ideas naturally.
- Do NOT start every paragraph with "I" — vary openings.
- Do NOT cram tools, metrics, and buzzwords into every sentence. Mention them where they naturally fit.
- Do NOT use corporate jargon like "leveraging", "streamlining", "spearheading", "facilitating", "ensuring full coverage". Use plain language.
- Do NOT end with a neat summary sentence that packages all your skills. Just stop naturally.
- Sound like a real person explaining their work to a colleague, not like a resume.
- A good test: read it out loud. If it sounds like a rehearsed elevator pitch, rewrite it.

BAD example (scripted, keyword-per-bullet):
- "Working with Selenium WebDriver paired with Java has been my main approach to automation."
- "I also make sure my automation fits smoothly into the CI/CD pipeline."
- "Beyond just writing tests, I've managed small QA teams, mentored juniors, and planned test strategies."
- "So for a QA Test Lead role like this, I bring a solid foundation in automation, team leadership, and Agile experience."

GOOD example (natural speech, grouped thoughts):
I have over 13 years of experience in software testing, and over the last several years I've been working extensively on test automation, mainly using Selenium WebDriver with Java.

In my recent projects, I've been responsible not only for developing automation scripts but also for maintaining the framework and making sure the tests are stable and reusable. For example, I use reusable page components, proper synchronization and waits, and common utilities so that we don't end up with a lot of flaky tests.

From a Test Lead perspective, my role goes beyond automation. I've worked on test planning, reviewing coverage, coordinating testing activities within the sprint, tracking defects, and mentoring other QA team members.

Most of my recent domain experience has been in healthcare and e-commerce, but the QA processes are very similar. So I believe that combination of hands-on experience along with QA leadership fits well with this role.

Rules:
1. Start with a direct, natural answer to the question.
2. Align to resume and JD subtly — don't announce alignment.
3. Do not invent unsupported experience.
4. Use practical, everyday project language.
5. Mention specific tools and techniques where natural, but don't force them.
6. Include real-world context and outcomes when they add value, not as checkbox items.
7. Keep it conversational — like you're talking to the interviewer across a table.
8. If the role is in a specific domain, weave domain knowledge naturally into your story.

Return in this format:
# 30-Second Version
(A short 3-4 sentence spoken summary — as if someone asked "give me the quick version")

# Real-Time Example
(A concrete story from the candidate's experience told naturally in short paragraphs. What happened, what you did, what the result was — told like a story, not a checklist.)

# Strong Answer
(The full answer written as natural short paragraphs — 2-4 sentences each, grouped by thought. This should read like a transcript of a real person speaking confidently in an interview.)

# Key Points to Mention
(Short bullet reminders)

# Resume/JD Alignment
(Bullet list)

# Possible Follow-Up Questions
(Bullet list)

# Follow-Up Answer Hints
(Short natural paragraphs for each follow-up)
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

CRITICAL OUTPUT FORMAT — NATURAL SPOKEN ANSWER:
Write every answer section as SHORT PARAGRAPHS (2-4 sentences each), not single-line bullets. Each paragraph is one coherent thought the person would say together. This is how people actually speak.
- Use natural transitions: "In my recent projects...", "For example...", "From a leadership perspective..."
- Do NOT map each paragraph to one JD keyword.
- Do NOT cram tools and buzzwords into every sentence.
- Do NOT use jargon like "leveraging", "streamlining", "spearheading". Use plain language.
- Do NOT end with a neat summary that packages all skills. Just stop naturally.
- Sound like a real person, not a keyword-stuffed bot.

Return in this format:
# Detected Question
(The clear interview question you identified)

# 30-Second Version
(A short 3-4 sentence spoken summary)

# Real-Time Example
(A concrete story told naturally in short paragraphs.)

# Strong Answer
(Full answer as natural short paragraphs — 2-4 sentences each, grouped by thought. Should read like a transcript of a real person speaking.)

# Key Points to Mention
(Short bullet reminders)

# Possible Follow-Up Questions
(Bullet list)

# Follow-Up Answer Hints
(Short natural paragraphs for each follow-up)
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
