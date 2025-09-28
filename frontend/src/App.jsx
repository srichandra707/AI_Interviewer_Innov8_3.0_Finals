// File: src/App.jsx
import React, { useState, useEffect, useRef } from 'react';
import Editor from '@monaco-editor/react';
import axios from 'axios';
import { Button, Card, CardContent, TextField, Typography, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

function App() {
  const [code, setCode] = useState(`// C++ Example
#include <iostream>
int main() {
    int a, b;
    std::cin >> a >> b;
    std::cout << a + b;
    return 0;
}`);
  const [language, setLanguage] = useState('cpp');
  const [output, setOutput] = useState('');
  const [testcases, setTestcases] = useState([]);
  const [chat, setChat] = useState([{ from: 'AI', message: 'Welcome to your AI interview!' }]);
  const chatEndRef = useRef(null);

  const hardcodedTestcases = [
    { input_data: "2 3", expected_output: "5", visible: true },
    { input_data: "10 20", expected_output: "30", visible: true },
    { input_data: "5 5", expected_output: "11", visible: false }
  ];

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat]);

  // API base URL
  const API_BASE_URL = 'http://127.0.0.1:5000/api';

  // Set up polling for new questions and hints
  useEffect(() => {
    // Set up polling for new questions
    const questionInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/question`);
        if (response.data.question) {
          setChat((prev) => [...prev, { from: 'AI', message: response.data.question }]);
        }
      } catch (error) {
        console.log('No new questions or server error');
      }
    }, 2000); // Poll every 2 seconds

    // Set up polling for hints
    const hintsInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/hints`);
        if (response.data.hints) {
          setChat((prev) => [...prev, { from: 'AI', message: `Hint: ${response.data.hints}` }]);
        }
      } catch (error) {
        console.log('No new hints or server error');
      }
    }, 3000); // Poll every 3 seconds

    return () => {
      clearInterval(questionInterval);
      clearInterval(hintsInterval);
    };
  }, []);

  // Function to send current code snapshot to backend
  const sendCodeSnapshot = async () => {
    try {
      await axios.post(`${API_BASE_URL}/snapshots`, {
        code: code,
        language: language,
        timestamp: Date.now()
      });
      console.log('Code snapshot sent successfully');
    } catch (error) {
      console.error('Failed to send code snapshot:', error);
    }
  };

  // Automatically send code snapshot every 10 seconds
  useEffect(() => {
    const snapshotInterval = setInterval(() => {
      sendCodeSnapshot();
    }, 10000); // Send snapshot every 10 seconds

    return () => clearInterval(snapshotInterval);
  }, [code, language]);

  const runCode = async () => {
    try {
      const res = await axios.post(`${API_BASE_URL}/run_code`, {
        language,
        code,
        testcases: hardcodedTestcases
      });
      if (res.data.results) {
        setTestcases(res.data.results.map((tc, idx) => ({
          id: idx + 1,
          input: tc.input_data,
          expected: hardcodedTestcases[idx].expected_output,
          visible: hardcodedTestcases[idx].visible,
          output: tc.output,
          status: tc.status
        })));
      }

      setOutput('Code executed. Check Testcases below for results.');

      const passedCount = res.data.results.filter(r => r.status === 'Accepted').length;
      setChat((prev) => [
        ...prev,
        { from: 'AI', message: `You passed ${passedCount} out of ${res.data.results.length} testcases.` }
      ]);
    } catch (err) {
      setOutput(err.response?.data?.error || 'Error running code');
    }
  };

  const sendMessage = async (message) => {
    if (!message) return;
    setChat((prev) => [...prev, { from: 'Candidate', message }]);
    try {
      const res = await axios.post(`${API_BASE_URL}/ai`, { message });
      setChat((prev) => [...prev, { from: 'AI', message: res.data.reply }]);
    } catch (err) {
      setChat((prev) => [...prev, { from: 'AI', message: 'AI failed to respond.' }]);
    }
  };

  return (
    <div className="flex min-h-screen">
      {/* Coding Panel */}
      <div className="flex-1 flex flex-col p-4">
        {/* Language Selector + Run Button */}
        <div className="flex space-x-2 items-center mb-2">
          <TextField
            select
            label="Language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            SelectProps={{ native: true }}
            variant="outlined"
            size="small"
          >
            <option value="python">python</option>
            <option value="cpp">C++</option>
          </TextField>
          <Button variant="contained" color="primary" onClick={runCode}>
            Run Code
          </Button>
        </div>

        {/* Editor: flex-grow to fill remaining space */}
        <div className="flex-1 mb-2">
          <Editor
            height="100%"
            language={language}
            value={code}
            onChange={setCode}
            theme="vs-dark"
          />
        </div>

        {/* Testcases Panel: stays at bottom */}
        <Card>
          <CardContent>
            <Typography variant="h6">Testcases</Typography>
            {testcases.map((tc) => (
              <Accordion
                key={tc.id}
                disabled={!tc.visible}
                TransitionProps={{ unmountOnExit: true }}
              >
                <AccordionSummary
                  expandIcon={tc.visible ? <ExpandMoreIcon /> : null}
                  aria-controls={`tc-content-${tc.id}`}
                  id={`tc-header-${tc.id}`}
                >
                  <span>
                    Testcase {tc.id} {tc.visible ? `(Visible)` : `(Hidden)`} -{" "}
                    {tc.status === "pass"
                      ? "✅ Pass"
                      : tc.status === "fail"
                      ? "❌ Fail"
                      : tc.status === "tle"
                      ? "⏱ TLE"
                      : tc.status === "error"
                      ? "⚠ Error"
                      : "⏳ Running"}
                  </span>
                </AccordionSummary>

                {tc.visible && (
                  <AccordionDetails>
                    <div className="max-h-48 overflow-auto space-y-2 p-2 border rounded bg-gray-50">
                      <div><strong>Input:</strong> <pre className="whitespace-pre-wrap">{tc.input}</pre></div>
                      <div><strong>Expected Output:</strong> <pre className="whitespace-pre-wrap">{tc.expected}</pre></div>
                      <div><strong>Your Output:</strong> <pre className="whitespace-pre-wrap">{tc.output}</pre></div>
                      <div><strong>Status:</strong> {tc.status}</div>
                      <div><strong>Test Passed:</strong> {tc.test_passed ? "✅ Yes" : "❌ No"}</div>
                    </div>
                  </AccordionDetails>
                )}
              </Accordion>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* AI Interview Panel */}
      <div className="w-1/3 border-l flex flex-col p-4 h-screen">
        <Typography variant="h6">AI Interviewer</Typography>
        <div className="flex-1 overflow-auto my-2 space-y-2 max-h-[calc(100vh-120px)]">
          {chat.map((msg, i) => (
            <div
              key={i}
              className={`p-2 rounded ${
                msg.from === 'AI' ? 'bg-gray-200 text-left' : 'bg-blue-200 text-right'
              }`}
            >
              <strong>{msg.from}: </strong>
              {msg.message}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
        <TextField
          placeholder="Type your question..."
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              sendMessage(e.target.value);
              e.target.value = '';
            }
          }}
        />
      </div>
    </div>
  );
}

export default App;