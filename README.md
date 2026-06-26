# Interview Practice Listener

A safe interview-practice app that listens to or accepts mock interview questions and generates resume/JD-aware practice answers.

> Intended for mock interviews, practice sessions, recordings, or interviews where AI assistance is explicitly allowed. Do not use it secretly during real interviews or assessments.

## Features

- Upload resume as PDF, DOCX, or TXT
- Paste job description and company/domain context
- Analyze resume + JD into a candidate profile
- Enter interview question manually
- Use browser microphone speech recognition to capture question transcript
- Detect the latest clear interview question from transcript
- Generate strong practice answer aligned to resume + JD
- Optional: type your own answer and get feedback/rewrite
- Test selected model/API key
- Mock mode when no API key is provided
- Download Q&A history as markdown

## Project Structure

```text
interview-practice-listener/
  backend/
    app/
      main.py
      routes/
      services/
    requirements.txt
    .env.example
  frontend/
    src/
    package.json
```

## Run Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Open backend docs:

```text
http://localhost:8000/docs
```

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## API Key / Model

You can run without an API key in mock mode. For live model responses, either:

1. Add API key to `backend/.env`:

```env
OPENAI_API_KEY=your_key_here
DEFAULT_MODEL=gpt-4o-mini
ALLOW_CLIENT_API_KEY=false
```

or

2. Paste API key in the frontend for local testing only.

For hosted production, use server-side environment variables and hide the API key field.

## Notes on Audio

The MVP uses browser speech recognition, which captures microphone input. Capturing same-laptop meeting/system audio depends on browser/OS permissions and is intentionally not optimized for hidden real-interview use. For ethical use, use it in mock sessions, practice calls, recordings, or consent-based contexts.

## Safe Product Modes

- Practice Mode: question → suggested answer
- Post-Answer Feedback Mode: question + your answer → feedback and stronger answer
- Consent-Based Live Mode: only where all parties allow AI support
