import React, { useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import './style.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const MODEL_OPTIONS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1', 'gpt-5.4-mini', 'gpt-5.4', 'gpt-5.5'];

function App() {
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-4o-mini');
  const [role, setRole] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [companyContext, setCompanyContext] = useState('');
  const [resumeText, setResumeText] = useState('');
  const [resumeName, setResumeName] = useState('');
  const [profile, setProfile] = useState(null);
  const [manualQuestion, setManualQuestion] = useState('');
  const [transcript, setTranscript] = useState('');
  const [detected, setDetected] = useState(null);
  const [answer, setAnswer] = useState('');
  const [userAnswer, setUserAnswer] = useState('');
  const [feedback, setFeedback] = useState('');
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState('');
  const [status, setStatus] = useState('');
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);

  const headers = useMemo(() => {
    const h = { 'Content-Type': 'application/json' };
    if (apiKey.trim()) h['x-openai-api-key'] = apiKey.trim();
    return h;
  }, [apiKey]);

  async function uploadResume(file) {
    if (!file) return;
    setLoading('Uploading resume...');
    setStatus('');
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch(`${API_BASE}/upload/resume`, { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      setResumeText(data.text || '');
      setResumeName(data.filename || file.name);
      setStatus(`Resume parsed: ${data.characters} characters`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  async function testModel() {
    setLoading('Testing model...');
    setStatus('');
    try {
      const res = await fetch(`${API_BASE}/coach/test-llm`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ model })
      });
      const data = await res.json();
      setStatus(data.ok ? `✅ ${data.message}` : `❌ ${data.error}`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  async function analyzeProfile() {
    setLoading('Analyzing resume and JD...');
    setStatus('');
    try {
      const res = await fetch(`${API_BASE}/coach/profile`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ role, job_description: jobDescription, resume_text: resumeText, company_context: companyContext, model })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Profile analysis failed');
      setProfile(data);
      setStatus('✅ Candidate profile created');
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  async function detectQuestionFromTranscript() {
    setLoading('Detecting latest question...');
    setStatus('');
    try {
      const res = await fetch(`${API_BASE}/coach/detect-question`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ transcript, model })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Detection failed');
      setDetected(data);
      if (data.clean_question) setManualQuestion(data.clean_question);
      setStatus(data.is_interview_question ? '✅ Question detected' : 'No clear interview question detected');
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  async function generateAnswer() {
    const question = manualQuestion.trim() || detected?.clean_question || '';
    if (!question) {
      setStatus('Enter or detect an interview question first.');
      return;
    }
    await streamAnswer(question);
  }

  async function evaluateMyAnswer() {
    const question = manualQuestion.trim() || detected?.clean_question || '';
    if (!question || !userAnswer.trim()) {
      setStatus('Question and your answer are required for feedback.');
      return;
    }
    setLoading('Evaluating your answer...');
    setFeedback('');
    try {
      const res = await fetch(`${API_BASE}/coach/evaluate`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ role, job_description: jobDescription, profile: profile || {}, question, user_answer: userAnswer, model })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Evaluation failed');
      setFeedback(data.feedback);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setStatus('Speech recognition is not supported in this browser. Try Chrome, or paste the question manually.');
      return;
    }
    const recog = new SpeechRecognition();
    recog.continuous = true;
    recog.interimResults = true;
    recog.lang = 'en-US';
    recognitionRef.current = recog;
    recog.onresult = (event) => {
      let finalText = '';
      let interimText = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += text + ' ';
        else interimText += text;
      }
      if (finalText) setTranscript(prev => `${prev} ${finalText}`.trim());
      setStatus(interimText ? `Listening: ${interimText}` : 'Listening...');
    };
    recog.onerror = (e) => setStatus(`Speech recognition error: ${e.error}`);
    recog.onend = () => setListening(false);
    recog.start();
    setListening(true);
  }

  async function stopListening() {
    if (recognitionRef.current) recognitionRef.current.stop();
    setListening(false);
    setStatus('Detecting question and generating answer...');
    await quickAnswerStream();
  }

  async function quickAnswerStream() {
    setLoading('Generating answer...');
    setAnswer('');
    setDetected(null);
    try {
      const res = await fetch(`${API_BASE}/coach/quick-answer`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ role, job_description: jobDescription, resume_text: resumeText, company_context: companyContext, profile: profile || {}, transcript, mode: 'practice', model })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed');
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            fullText += data;
            setAnswer(fullText);
          }
        }
      }
      setStatus('');
      if (fullText) {
        const qMatch = fullText.match(/# Detected Question\s*\n([^\n#]+)/);
        const detectedQ = qMatch ? qMatch[1].trim() : '';
        if (detectedQ) setManualQuestion(detectedQ);
        setHistory(prev => [{ question: detectedQ || 'From transcript', answer: fullText, createdAt: new Date().toISOString() }, ...prev]);
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  async function detectAndGenerate() {
    setLoading('Detecting question...');
    setAnswer('');
    try {
      const res = await fetch(`${API_BASE}/coach/detect-question`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ transcript, model })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Detection failed');
      setDetected(data);
      if (data.clean_question) setManualQuestion(data.clean_question);
      if (!data.is_interview_question || !data.clean_question) {
        setStatus('No clear interview question detected.');
        setLoading('');
        return;
      }
      setStatus('✅ Question detected. Generating answer...');
      await streamAnswer(data.clean_question);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  async function streamAnswer(question) {
    setLoading('Generating answer...');
    setAnswer('');
    try {
      const res = await fetch(`${API_BASE}/coach/answer-stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ role, job_description: jobDescription, resume_text: resumeText, company_context: companyContext, profile: profile || {}, question, mode: 'practice', model })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Answer generation failed');
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            fullText += data;
            setAnswer(fullText);
          }
        }
      }
      setStatus('');
      setHistory(prev => [{ question, answer: fullText, createdAt: new Date().toISOString() }, ...prev]);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setLoading('');
    }
  }

  function downloadHistory() {
    const text = history.map((h, i) => `# Question ${history.length - i}\n${h.question}\n\n${h.answer}\n`).join('\n---\n');
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `interview-practice-answers-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return <div className="app">
    <header>
      <div>
        <h1>Interview Practice Listener</h1>
        <p>For mock interviews, practice sessions, or consent-based use only. Do not use secretly in real interviews.</p>
      </div>
      <div className="headerControls">
        <select value={model} onChange={e => setModel(e.target.value)}>{MODEL_OPTIONS.map(m => <option key={m}>{m}</option>)}</select>
        <input type="password" placeholder="Optional OpenAI API key" value={apiKey} onChange={e => setApiKey(e.target.value)} />
        <button onClick={testModel}>Test Model</button>
      </div>
    </header>

    <main className="grid">
      <section className="card context">
        <h2>1. Candidate Context</h2>
        <label>Role / Title</label>
        <input value={role} onChange={e => setRole(e.target.value)} placeholder="Senior DevOps Engineer" />
        <label>Company / Domain Context</label>
        <input value={companyContext} onChange={e => setCompanyContext(e.target.value)} placeholder="Banking, healthcare, telecom, e-commerce..." />
        <label>Job Description</label>
        <textarea value={jobDescription} onChange={e => setJobDescription(e.target.value)} placeholder="Paste job requirements here" />
        <label>Resume</label>
        <input type="file" accept=".pdf,.docx,.txt" onChange={e => uploadResume(e.target.files[0])} />
        {resumeName && <small>Loaded: {resumeName}</small>}
        <textarea value={resumeText} onChange={e => setResumeText(e.target.value)} placeholder="Or paste resume text here" />
        <button onClick={analyzeProfile}>Analyze Resume + JD</button>
        {profile && <details open><summary>Candidate Profile</summary><pre>{JSON.stringify(profile, null, 2)}</pre></details>}
      </section>

      <section className="card listener">
        <h2>2. Listen or Enter Question</h2>
        <div className="row">
          {!listening ? <button onClick={startListening}>Start Mic Listening</button> : <button className="danger" onClick={stopListening}>Stop Listening</button>}
          <button onClick={detectQuestionFromTranscript}>Detect Question</button>
        </div>
        <label>Live Transcript / Notes</label>
        <textarea className="big" value={transcript} onChange={e => setTranscript(e.target.value)} placeholder="Transcript will appear here, or paste interviewer question/context here" />
        {detected && <div className="detected"><b>Detected:</b> {detected.clean_question}<br/><small>{detected.question_type} • {detected.topic} • {detected.difficulty}</small></div>}
        <label>Interview Question</label>
        <textarea value={manualQuestion} onChange={e => setManualQuestion(e.target.value)} placeholder="Paste or confirm the interview question here" />
        <button className="primary" onClick={generateAnswer}>Generate Practice Answer</button>

        <h3>Optional: Evaluate My Own Answer</h3>
        <textarea value={userAnswer} onChange={e => setUserAnswer(e.target.value)} placeholder="Type your answer here, then get feedback" />
        <button onClick={evaluateMyAnswer}>Evaluate My Answer</button>
      </section>

      <section className="card output">
        <h2>3. Suggested Answer / Feedback</h2>
        {loading && <div className="loading">{loading}</div>}
        {status && <div className="status">{status}</div>}
        {answer && <div className="markdown"><ReactMarkdown>{answer}</ReactMarkdown></div>}
        {feedback && <><h2>Feedback on Your Answer</h2><div className="markdown"><ReactMarkdown>{feedback}</ReactMarkdown></div></>}
        <div className="row"><button onClick={downloadHistory} disabled={!history.length}>Download Q&A History</button><button onClick={() => {setAnswer(''); setFeedback('');}}>Clear Output</button></div>
      </section>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
