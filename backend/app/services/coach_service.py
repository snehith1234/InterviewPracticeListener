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
(One sentence per line, each starting with "- ". Natural spoken sentences — something the candidate could say aloud. Do not shorten or rewrite the content, just display each sentence on its own line.)

# Short Version to Memorize
(One sentence per line, each starting with "- ". 3-4 key sentences written naturally.)

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

DISPLAY FORMAT — ONE SENTENCE PER POINT:
Do NOT display answers as paragraphs. Break every answer into short, easy-to-read points.
Each point must:
- Contain one complete sentence or one clear thought.
- End with a full stop.
- Appear on a separate line, starting with "- ".
- Keep the natural spoken flow of the answer.
- NOT shorten or rewrite the answer just for formatting.

For example, instead of:
"Sure. I have over 13 years of experience in software testing, and for about the last seven years I've been working mainly on automation. Most of my recent experience has been in healthcare."

Display as:
- Sure.
- I have over 13 years of experience in software testing, and for about the last seven years I've been working mainly on automation using Selenium WebDriver with Java.
- Most of my recent experience has been in healthcare, working on applications related to member enrollment and claims processing.

Apply this sentence-by-sentence point format to every answer section (30-Second Version, Real-Time Example, Strong Answer, Follow-Up Answer Hints).

STYLE RULES:
- STOP TRYING TO MAXIMIZE KEYWORD COVERAGE. Prioritize believable spoken answers over matching every line of the job description. A real person in an interview does not mention Selenium + REST Assured + SQL + TestNG + Cucumber + Jenkins + Agile + leadership + stakeholders + banking all in one answer.
- Instead, give ONE OR TWO small concrete examples from actual project work. For example: "In my healthcare projects, I worked on member enrollment and claims processing, where we had to validate not only the UI but also whether the correct data was being updated in the database." That makes the candidate sound like they've actually done the work.
- Use simple spoken English instead of polished corporate language.
- Keep answers focused on the exact question being asked.
- Avoid unnecessary phrases such as "I'm confident I can transition effectively," "the fundamentals are universal," "I see a great match," "I've built my career around," "my experience isn't limited to," "Besides my technical skills," "One key part of my work is," "This approach caught," "has prepared me well," "I want to be clear that," "I'm confident that with some ramp-up," "I'm ready to apply my skills," "This gave me confidence that," "I see many parallels," "contribute effectively," "I'm confident that my technical skills and," "deliver good quality results," "keep the testing aligned with development," or similar generic interview statements. Also avoid overly formal phrases like "triage," "facilitating," "ensuring," "precision and compliance," "planning test coverage with the team" — use simpler alternatives like "working through issues," "helping with," "making sure," "getting things right," "working on what needs to be tested for a release."
- Do NOT add narrated outcome statements like "This helped catch several data issues early," "which helped speed up our release cycles," "This helped reduce manual testing effort and catch defects faster." Real people in interviews don't narrate the impact of every action. Only mention an outcome if it was a specific, memorable event the candidate could describe in detail if asked.
- Do NOT add specific details the candidate might not be able to back up in follow-ups. For example, don't say "multiple databases" unless the resume or context confirms that. Don't say "assigned tasks based on their strengths" unless the candidate can give a real example. Specific details are only strong when the candidate can comfortably answer follow-up questions about them.
- Do NOT bridge domains mid-answer with phrases like "which is critical in healthcare and would be just as important in banking" or "I understand that accuracy and data integrity are crucial in this domain." These sound like prepared transitions. If there's a domain gap, state it simply at the end and move on.
- Be PRECISE about domain experience. Do NOT say "most of my experience is in X" if only the recent experience is in X. Say "my recent experience is in X" or "I've been working in X for the last few years." Match what the resume and additional context actually say about the timeline.
- Use shorter sentences and a natural flow, like someone explaining their actual experience.
- Prefer specific examples of what the candidate did instead of broad claims about skills.
- It is okay for the answer to sound slightly imperfect or conversational. It should NOT sound memorized.
- If the candidate has not worked with a particular technology or domain, say that clearly and briefly instead of trying to compensate with a long explanation.
- Do NOT force job-description keywords into the answer unless they naturally relate to the question.
- Do NOT name-drop tools from the JD that the candidate has NOT actually used.
- Do NOT end with a sentence that lists multiple skill categories ("independently or as part of a team and communicating clearly with developers, managers, and business stakeholders"). That sounds pulled from a JD.
- Keep most answers around 45-90 seconds of speaking time unless the question requires a detailed example.
- For technical questions, explain the actual approach step by step rather than giving textbook definitions.
- For leadership questions, focus on what was personally done with the team, decisions made, problems handled, and outcomes.
- When appropriate, use phrases that people naturally use while speaking: "Usually what I do is...", "In my last project...", "One example would be...", "The first thing I check is...", "We ran into this issue once...", "For example, in my healthcare projects..."
- Do NOT end every answer with a summary of why the candidate is a good fit for the role. Only connect it back to the role when it naturally makes sense.
- Most importantly, write the answer as something the candidate could comfortably say aloud in an interview, not as something they would submit in writing.

BAD example (domain bridging, narrated outcomes, unverifiable claims):
"I write SQL queries to verify that the data is correctly inserted or updated in the backend, which is critical in healthcare and would be just as important in banking. This helped reduce manual testing effort and catch defects faster. I assigned tasks to team members based on their strengths. I understand that accuracy and data integrity are crucial in this domain. I'm comfortable applying the solid automation and leadership experience I have to deliver good quality results."

GOOD example (natural, concrete, sentence-per-point format):
- Sure.
- I have over 13 years of experience in software testing, and for about the last seven years I've been working mainly on automation using Selenium WebDriver with Java.
- Most of my recent experience has been in healthcare, working on applications related to member enrollment and claims processing.
- When I automate a workflow, I usually don't stop with checking the UI.
- I also use SQL to verify what happened in the backend.
- For example, in one claims project, I automated the process of submitting a claim through the application and then checked the database to make sure the claim record and status were updated correctly.
- We also ran our automation through Jenkins whenever we received a new build, so we could identify failures without waiting for everything to be tested manually.
- Over time, I also started handling more lead responsibilities.
- I worked with a small QA team, divided the testing work, reviewed automation scripts, and helped team members when they had issues with their tests or environments.
- I was also involved in sprint planning and daily meetings with the development team.
- I haven't worked directly in banking, so I would need to learn the specific business flows.
- But I'm already used to applications where data accuracy and careful testing are very important, and I'm comfortable learning a new domain.

Return in this format:
# 30-Second Version
(One sentence per line, each starting with "- ". A quick 3-5 sentence spoken summary.)

# Real-Time Example
(One sentence per line, each starting with "- ". A concrete story from the candidate's experience told naturally.)

# Strong Answer
(One sentence per line, each starting with "- ". The full answer as natural spoken sentences. Keep the conversational flow — do not shorten or rewrite just for formatting.)

# Key Points to Mention
(Short bullet reminders)

# Resume/JD Alignment
(Bullet list)

# Possible Follow-Up Questions
(Bullet list)

# Follow-Up Answer Hints
(One sentence per line, each starting with "- ". Natural spoken sentences for each follow-up.)
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


def detect_and_answer_stream(role: str, job_description: str, resume_text: str, company_context: str, additional_context: str, profile: dict, transcript: str, quick_answer: str, mode: str, api_key: str | None, model: str | None) -> Generator[str, None, None]:
    """Combined: detect question from transcript AND generate answer in one LLM call (streamed)."""
    additional_block = ""
    if additional_context and additional_context.strip():
        additional_block = f"""

IMPORTANT — Additional context provided by the candidate (their actual domain experience, work details, project notes, etc.). Use this to ground your answer accurately. Do NOT contradict this context:
{additional_context[:15000]}
"""

    quick_answer_block = ""
    if quick_answer and quick_answer.strip():
        quick_answer_block = f"""

CRITICAL — CONTINUE FROM THIS QUICK ANSWER:
The candidate may already be speaking the Quick Answer shown below. You MUST build the entire answer around it. Preserve the same story, technical approach, example, and sequence. Do NOT contradict it or restart with a different answer.

Quick Answer already given:
{quick_answer}

All sections must be consistent with this Quick Answer:
- 30-Second Version = Quick Answer ideas expanded to 3-5 sentences.
- Strong Answer = Quick Answer ideas expanded to 7-8 sentences with more detail. The first 2 sentences of Strong Answer must match the Quick Answer.
- Real-Time Example = Expand the SAME example/scenario from the Quick Answer with more detail. Do NOT introduce a different project or different technical solution.
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
{additional_block}{quick_answer_block}"""
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
{additional_block}{quick_answer_block}"""
    prompt = f"""
The user is in a mock interview practice session. Below is a transcript from the conversation. Your job:
1. Identify the latest clear interview question from the transcript.
2. Generate a strong practice answer aligned to the candidate's context.
{"3. IMPORTANT: A Quick Answer has already been given to the candidate. ALL sections must continue from that same answer. Do NOT start a different story or approach." if quick_answer_block else ""}

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
Display every answer as ONE SENTENCE PER POINT — each line starts with "- ", contains one complete thought, and ends with a full stop. Do NOT write paragraphs. Keep the natural spoken flow.
- STOP TRYING TO MAXIMIZE KEYWORD COVERAGE. Prioritize believable spoken answers over matching every line of the job description. Give one or two small concrete examples from actual work instead of listing every tool and skill.
- Use simple spoken English instead of polished corporate language.
- Keep answers focused on the exact question being asked.
- Avoid phrases like "I'm confident I can transition effectively," "the fundamentals are universal," "I see a great match," "Besides my technical skills."
- Use shorter sentences and natural flow. Prefer specific examples over broad claims.
- Use natural spoken phrases: "Usually what I do is...", "In my last project...", "One example would be...", "The first thing I check is...", "For example, in my healthcare projects..."
- Do NOT end with a summary of why the candidate is a good fit. Only connect back to the role when natural.
- Do NOT name-drop tools the candidate hasn't used. Do NOT end with a sentence listing multiple skill categories.
- Do NOT bridge domains mid-answer. If there's a domain gap, state it simply at the end.
- Be honest about gaps — say them plainly.
- Write something the candidate could comfortably say aloud, not submit in writing.

Return in this format:
# Detected Question
(The clear interview question you identified)

# 30-Second Version
(One sentence per line, each starting with "- ". Quick 3-5 sentence spoken summary.)

# Real-Time Example
(One sentence per line, each starting with "- ". A concrete story told naturally.)

# Strong Answer
(One sentence per line, each starting with "- ". Full answer as natural spoken sentences.)

# Key Points to Mention
(Short bullet reminders)

# Possible Follow-Up Questions
(Bullet list)

# Follow-Up Answer Hints
(One sentence per line, each starting with "- ".)
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
    if profile and profile.get("candidate_summary"):
        context_block = f"Profile: {profile.get('candidate_summary', '')}. Skills: {', '.join(profile.get('key_skills', [])[:5])}."
    else:
        context_block = f"Role: {role}. Resume: {resume_text[:800]}."

    prompt = f"""Question from transcript, then 2-sentence answer.

Context: {context_block}

Transcript: {transcript[-1500:]}

Reply ONLY:
**Q:** [question]
**A:** [2 sentences max]
"""
    return responses_stream(
        prompt,
        system="Ultra-short interview answers. 2 sentences max. No fluff.",
        api_key=api_key,
        model=model,
        kind="quick",
        max_output_tokens=150,
    )
