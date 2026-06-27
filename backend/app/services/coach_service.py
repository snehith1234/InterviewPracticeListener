from typing import Generator
from app.services.llm_client import responses_json, responses_text, responses_stream


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
    prompt = _build_answer_prompt(role, job_description, resume_text, company_context, profile, question, mode)
    return responses_text(
        prompt,
        system="You are an expert interview answer coach with deep knowledge across technical domains (DevOps, cloud, AI/ML, data science, software engineering) AND business domains (banking, telecom, healthcare, e-commerce, insurance, manufacturing, etc.). Generate strong, truthful, domain-aware candidate answers for practice sessions.",
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


def _build_answer_prompt(role: str, job_description: str, resume_text: str, company_context: str, profile: dict, question: str, mode: str) -> str:
    """Build the answer prompt, trimming context if profile exists."""
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
"""
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
"""
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

Rules:
1. Start with a direct answer.
2. Align to resume and JD.
3. Do not invent unsupported experience. If something is assumed, phrase it as a reasonable way to answer.
4. Use practical project language.
5. Mention specific tools, metric names, commands, or thresholds where relevant — not generic descriptions.
6. Include production/project context and measurable business impact (time saved, incidents prevented, uptime improved, deployment frequency, etc.).
7. For senior roles, use Tool + Project + Issue + Action + Result + Impact.
8. Avoid overly textbook language.
9. Keep it interview-speak: natural, confident, not too long.
10. In Real-Time Example, always include: what the situation was, what you specifically did (with tool/command names), and the measurable outcome or impact.
11. DOMAIN EXPERTISE: If the company/role is in a specific domain (banking, telecom, healthcare, e-commerce, insurance, etc.), weave in domain-specific terminology, regulations, workflows, and concerns naturally. For example:
    - Banking/Finance: mention PCI-DSS, SOX compliance, transaction integrity, core banking systems, payment gateways, fraud detection, regulatory reporting, settlement cycles.
    - Telecom: mention OSS/BSS, network provisioning, CDRs, SLA management, 5G/LTE, service assurance, mediation, billing systems.
    - Healthcare: mention HIPAA, HL7/FHIR, EHR/EMR systems, patient data privacy, clinical workflows.
    - E-commerce: mention cart abandonment, payment processing, catalog services, peak traffic handling, recommendation engines.
    - Insurance: mention claims processing, underwriting, actuarial systems, policy administration.
    Show you understand the business context, not just the tech.

Return in this format:
# 30-Second Version

# Real-Time Example
(A concrete example from the candidate's resume/experience. MUST include: specific tools/commands used, the action taken, and the measurable impact — e.g. "reduced deployment time from 45min to 8min", "cut incident response from 30min to under 5min", "achieved 99.9% uptime". If no direct experience, build a realistic scenario with specifics the candidate could credibly claim. Use domain-specific context when applicable.)

# Strong Answer

# Key Points to Mention

# Resume/JD Alignment

# Possible Follow-Up Questions

# Follow-Up Answer Hints
"""


def generate_answer_stream(role: str, job_description: str, resume_text: str, company_context: str, profile: dict, question: str, mode: str, api_key: str | None, model: str | None) -> Generator[str, None, None]:
    """Stream answer tokens for low-latency perceived response."""
    prompt = _build_answer_prompt(role, job_description, resume_text, company_context, profile, question, mode)
    return responses_stream(
        prompt,
        system="You are an expert interview answer coach with deep knowledge across technical domains (DevOps, cloud, AI/ML, data science, software engineering) AND business domains (banking, telecom, healthcare, e-commerce, insurance, manufacturing, etc.). Generate strong, truthful, domain-aware candidate answers for practice sessions.",
        api_key=api_key,
        model=model,
        kind="answer",
    )


def detect_and_answer_stream(role: str, job_description: str, resume_text: str, company_context: str, profile: dict, transcript: str, mode: str, api_key: str | None, model: str | None) -> Generator[str, None, None]:
    """Combined: detect question from transcript AND generate answer in one LLM call (streamed)."""
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
"""
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
"""
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
2. Then provide the answer using practical project language aligned to the candidate context.
3. Do not invent unsupported experience.
4. Mention specific tools, metric names, commands, or thresholds — not generic descriptions.
5. Include production/project context and measurable business impact (time saved, incidents prevented, uptime improved, deployment frequency, etc.).
6. For senior roles, use Tool + Project + Issue + Action + Result + Impact.
7. Keep it interview-speak: natural, confident, not too long.
8. In Real-Time Example, always include: specific tools/commands used, the action taken, and the measurable outcome.
9. DOMAIN EXPERTISE: If the company/role is in a specific domain, weave in domain-specific terminology, regulations, and business concerns naturally. Show you understand the business, not just the tech.

Return in this format:
# Detected Question
(The clear interview question you identified)

# 30-Second Version

# Real-Time Example
(A concrete example from the candidate's context. MUST include: specific tools/commands, the action taken, and measurable impact. Use domain-specific context when applicable.)

# Strong Answer

# Key Points to Mention

# Possible Follow-Up Questions

# Follow-Up Answer Hints
"""
    return responses_stream(
        prompt,
        system="You are an expert interview answer coach with deep knowledge across technical domains (DevOps, cloud, AI/ML, data science, software engineering, project management, product ownership) AND business domains (banking, telecom, healthcare, e-commerce, insurance, manufacturing). First identify the question from the transcript, then generate a strong, domain-aware practice answer.",
        api_key=api_key,
        model=model,
        kind="answer",
    )


def quick_short_answer_stream(role: str, job_description: str, resume_text: str, company_context: str, profile: dict, transcript: str, api_key: str | None, model: str | None) -> Generator[str, None, None]:
    """Ultra-fast first response: detect question + give ONLY a 2-sentence answer. Streams immediately."""
    if profile and profile.get("candidate_summary"):
        context_block = f"Profile: {profile.get('candidate_summary', '')}. Skills: {', '.join(profile.get('key_skills', [])[:8])}. Style: {profile.get('answer_style_guidance', '')}."
    else:
        context_block = f"Role: {role}. Key JD: {job_description[:2000]}. Resume highlights: {resume_text[:2000]}."

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
