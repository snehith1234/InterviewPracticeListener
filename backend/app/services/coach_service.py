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
        system="You generate interview answers that sound like a real person speaking naturally. Use simple spoken English. Never sound like AI output or a job description. Prefer specific examples over broad claims.",
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
        system="You are a technical interview coach. Give practical feedback. When rewriting answers, write them as natural spoken English — something the candidate could say aloud in an interview, not polished written text.",
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
Write the answer as if an experienced professional is speaking live in an interview. It must sound natural and conversational — not like it was written by AI or copied from a job description.
Format as SHORT PARAGRAPHS (2-4 sentences each). Each paragraph is one coherent thought the person would say before pausing. This is how people actually talk.

STYLE RULES:
- Use simple spoken English instead of polished corporate language.
- Keep answers focused on the exact question being asked.
- Do NOT try to include every tool, technology, domain, and leadership skill in every answer.
- Avoid unnecessary phrases such as "I'm confident I can transition effectively," "the fundamentals are universal," "I see a great match," "I've built my career around," "my experience isn't limited to," or similar generic interview statements.
- Use shorter sentences and a natural flow, like someone explaining their actual experience.
- Prefer specific examples of what the candidate did instead of broad claims about skills.
- It is okay for the answer to sound slightly imperfect or conversational. It should NOT sound memorized.
- If the candidate has not worked with a particular technology or domain, say that clearly and briefly instead of trying to compensate with a long explanation.
- Do NOT force job-description keywords into the answer unless they naturally relate to the question.
- Do NOT name-drop tools from the JD that the candidate has NOT actually used.
- Keep most answers around 45-90 seconds of speaking time unless the question requires a detailed example.
- For technical questions, explain the actual approach step by step rather than giving textbook definitions.
- For leadership questions, focus on what was personally done with the team, decisions made, problems handled, and outcomes.
- When appropriate, use phrases that people naturally use while speaking: "Usually what I do is...", "In my last project...", "One example would be...", "The first thing I check is...", "We ran into this issue once..."
- Do NOT end every answer with a summary of why the candidate is a good fit for the role. Only connect it back to the role when it naturally makes sense.
- Most importantly, write the answer as something the candidate could comfortably say aloud in an interview, not as something they would submit in writing.

BAD example (polished, scripted, JD-keyword matching):
"Although I haven't worked directly in banking, the fundamentals of testing remain quite consistent. I'm confident that my strong foundation in automation, SQL and backend validation, Agile collaboration, and team leadership will allow me to contribute effectively and ramp up quickly in a banking environment. I'm also fully open to learning new tools and processes specific to your environment, whether that's Playwright, Litmus, or Micro Focus Octane."

GOOD example (natural, honest, spoken):
I have over 13 years of experience in software testing, and a large part of my recent work has been focused on automation. I've mainly worked with Selenium WebDriver and Java for UI testing, REST Assured for API automation, and SQL for backend and data validation.

In my projects, I've been involved throughout the testing lifecycle. I work with the team to understand requirements, identify the important test scenarios, prepare the test approach, execute testing, track defects, and support release validation. I've also integrated automation suites with Jenkins so that regression tests can run as part of the CI/CD process and give the team faster feedback.

Along with the hands-on testing, I've also taken on team responsibilities. I've coordinated testing activities during sprints, reviewed test cases and automation scripts, helped with defect triage, and mentored junior QA engineers when they needed support.

Most of my domain experience has been in healthcare, so I'm used to working with complex workflows and data that needs to be validated very carefully. I haven't worked directly in banking, so that would be a new domain for me, but I'm comfortable learning new business processes and understanding how the application works.

For this Test Lead role, I think my combination of hands-on automation experience and experience coordinating QA activities would allow me to contribute both technically and from a team leadership perspective.

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
        system="You generate interview answers that sound like a real person speaking naturally. Use simple spoken English. Never sound like AI output or a job description. Prefer specific examples over broad claims.",
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
2. Then provide the answer as natural spoken English — like an experienced professional talking live in an interview.
3. Do not invent unsupported experience.
4. If the candidate hasn't worked with a technology or domain, say that clearly and briefly.
5. Do not force JD keywords into the answer unless they naturally relate to the question.

CRITICAL OUTPUT FORMAT — NATURAL SPOKEN ANSWER:
Write every answer section as SHORT PARAGRAPHS (2-4 sentences each). Each paragraph is one coherent thought the person would say together.
- Use simple spoken English instead of polished corporate language.
- Keep answers focused on the exact question being asked.
- Do NOT try to include every tool, technology, domain, and leadership skill in every answer.
- Avoid phrases like "I'm confident I can transition effectively," "the fundamentals are universal," "I see a great match."
- Use shorter sentences and natural flow. Prefer specific examples over broad claims.
- Use natural spoken phrases: "Usually what I do is...", "In my last project...", "One example would be...", "The first thing I check is..."
- Do NOT end with a summary of why the candidate is a good fit. Only connect back to the role when natural.
- Do NOT name-drop tools the candidate hasn't used.
- Be honest about gaps — say them plainly.
- Write something the candidate could comfortably say aloud, not submit in writing.

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
        system="First identify the question from the transcript, then generate a natural spoken answer. Use simple English. Sound like a real person, not AI output. Prefer specific examples over broad claims.",
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
