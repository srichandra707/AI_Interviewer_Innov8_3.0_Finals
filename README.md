# AI_Interviewer_Innov8_3.0_Finals
This repository contains the codebase for the final problem statement of Innov8_3.0 Hackathon - IIT Delhi 2025

# AI Coding Interview Platform

## Overview

This project is a **full-stack AI-powered coding interview platform** designed to simulate a real coding interview experience. It combines:

1. **Frontend** – A React app for coding, test case evaluation, chatting with an AI interviewer, and live code snapshots.  
2. **Voice & Transcription Layer** – Allows voice input to be transcribed into code or messages for the AI to process.  
3. **Backend & AI Agent** – Flask API that runs code, manages test cases, stores snapshots, and communicates with the **Agentic AI** for semantic analysis and hint generation.

---

## 1. Frontend: React App

**Location:** `frontend/src/App.jsx`  

The React frontend is **the main interface** for candidates:

- **Code Editor**: Uses [Monaco Editor](https://github.com/microsoft/monaco-editor) for Python and C++.  
- **Test Case Panel**: Displays visible and hidden test cases with pass/fail/TLE/error statuses.  
- **AI Interview Panel**:  
  - Shows AI questions and candidate responses in real-time.  
  - Polls backend every 2 seconds for new questions and every 3 seconds for hints.  
  - Allows candidates to chat with the AI for guidance or clarifications.  
- **Automatic Code Snapshots**: Sends the current code and language every 10 seconds to the backend for analysis or auditing.  
- **Run Code Button**: Sends code and test cases to the backend to execute and update results in the UI.  

**Frontend Styling:** TailwindCSS and MUI components for clean, responsive, split-screen layout: coding on the left, AI chat on the right.  

**Minimal Backend Reference:**  
The React app communicates with a Flask backend (`app.py`) which exposes REST endpoints for running code, AI messaging, hints, questions, and snapshots.

---

## 2. Voice Input & Live Transcription

**Location:** Integrated via backend and Deepgram API (`app.py`)  

- Users can **speak code or questions**, which is captured and **transcribed in real-time**.  
- Voice input is processed alongside typed input, enabling **hands-free interaction**.  
- Transcriptions are treated as code or chat messages and sent through the same endpoints as typed input:  
  - `/run_code` for code execution.  
  - `/ai` for AI interaction.  
- This ensures **voice-driven coding interviews** are as seamless as typing.  
- Backend handles **parallel transcription and analysis** using threading, keeping the UI responsive.

---

## 3. AI Agent: Agentic Hint-Giving

**Location:** `agent/agent.py`  

The **Agentic AI** is the **core intelligence** behind this platform. Its responsibilities:

1. **AST Structural Analysis**  
   - Checks the structure of candidate code against reference solutions.  
   - Detects logical patterns, loops, and functions to give accurate feedback.  

2. **Semantic AI Analysis (LLM)**  
   - Uses OpenAI or Google Gemini to understand **intent** and correctness beyond test cases.  
   - Evaluates code meaning and provides natural language hints or suggestions.  

3. **Hint Generation & Guidance**  
   - Generates hints when candidates are stuck.  
   - Combines **structural and semantic checks** for more precise guidance.  
   - Returns results to the frontend chat panel in real-time.  

4. **Test Case Aggregation**  
   - Combines outputs from AST analysis, semantic checks, and code execution.  
   - Sends a unified response back to the candidate showing:  
     - Testcase pass/fail  
     - AI hints  
     - Semantic feedback  

**Impact:** This layer ensures the platform is **truly agentic**, not just running code—it actively **guides candidates**, evaluates their approach, and provides **intelligent hints** in real-time.

---

## Project Structure
```
ai-coding-interview/
│
├── frontend/
│ ├── src/
│ │ ├── App.jsx # Main React frontend (coding + chat)
│ │ ├── index.jsx # Entry point
│ │ └── index.css # Tailwind imports
│ └── tailwind.css # Tailwind base styles
│
├── backend/
│ └── app.py # Flask API server (runs code, handles voice, snapshots)
│
├── agent/
│ └── agent.py # Agentic AI logic: AST + semantic + hint generation
│
├── requirements.txt # Python dependencies
└── README.md

```
---

## Requirements

```txt
Flask==2.3.3
websockets==11.0.3
sounddevice==0.4.9
numpy==1.26.5
deepgram-sdk==1.3.2
openai==1.30.0
google-generativeai==0.1.4
python-dotenv==1.0.1
asttokens==2.2.1
textwrap3==0.9.2
pywin32==305
requests==2.31.0
pandas==2.1.1
