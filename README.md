# Voice AI Agent 

A voice-in, voice-out AI agent that listens to you, reasons with an LLM, decides whether it needs a tool, uses that tool, and replies out loud — all with free, open-source tools running locally. No paid APIs.

This isn't just a chatbot with a microphone bolted on. It's a full pipeline:

```
🎙️ Microphone Input
      ↓
📝 Speech-to-Text (Faster Whisper)
      ↓
🧠 LLM Reasoning (Llama 3.2 via Ollama)
      ↓
🔧 Tool Calling (Python functions, if needed)
      ↓
🔊 Text-to-Speech (Piper TTS)
      ↓
Spoken response
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **SoundDevice** | Records audio from your microphone |
| **Faster Whisper** | Speech-to-text (optimized version of OpenAI's Whisper) |
| **Ollama + Llama 3.2** | Local LLM for reasoning and deciding on tool use |
| **Piper TTS** | Converts the AI's text response into spoken audio |
| **Python `eval`/`datetime`** | Two simple example tools: a calculator and a clock |

100% free. 100% local. No OpenAI, no Google Cloud, no API keys.

---

## Requirements

- Windows, macOS, or Linux
- Python 3.9+
- A working microphone and speakers
- ~8 GB RAM recommended (CPU is fine; GPU speeds things up)

---

## Project Structure

```
voice_ai_agent/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── main.py
```

---

## Setup

### 1. Install Ollama and pull Llama 3.2
Install from [ollama.com](https://ollama.com/download), then:
```bash
ollama pull llama3.2
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Python dependencies
```bash
pip install faster-whisper sounddevice scipy ollama piper-tts
```

### 4. Download a Piper voice model
Piper needs a voice model file (`.onnx` + its `.json` config) to generate speech. Download one (e.g. `en_US-lessac-medium`) from the [Piper voices repository](https://github.com/rhasspy/piper/blob/master/VOICES.md) and place both files in your project folder. The code expects:
```python
PIPER_MODEL = "en_US-lessac-medium.onnx"
```
Update this path in `main.py` if you use a different voice.

### 5. Run it
```bash
python main.py
```
Speak when you see "Listening..." — the agent records for 5 seconds, transcribes what you said, thinks, (optionally) uses a tool, and speaks its answer. Say "exit," "quit," or "stop" to end.

---

## How It Works

### 1. Recording audio — `record_audio()`
Uses `sounddevice` to record from your mic at 16,000 Hz (a rate that works well with speech models) for a fixed 5-second window, then saves it as a `.wav` file. Fixed-duration recording is the simplest approach to start with; production systems typically use **Voice Activity Detection (VAD)** instead, which records only while you're actually speaking.

### 2. Speech-to-text — `transcribe_audio()`
Faster Whisper (an optimized version of OpenAI's Whisper) transcribes the audio file into text. The `"base"` model is a reasonable default for a normal laptop — `tiny` is fastest/least accurate, `large-v3` is most accurate/most demanding. `compute_type="int8"` keeps memory usage down so it runs comfortably on CPU.

### 3. The tools — `get_current_time()` and `calculate()`
Two intentionally simple example tools. In a real system, tools could hit databases, weather APIs, calendars, email, or internal company systems — the pattern stays the same regardless of what the tool actually does.

### 4. LLM reasoning — `ask_llm()`
The transcribed text is wrapped in a prompt that describes the available tools and instructs the model: if a tool is needed, respond with a specific JSON shape (`{"tool": ..., "argument": ...}`); otherwise just answer normally. **The LLM never executes anything itself** — it only decides what should happen next, in plain text.

### 5. Tool execution — `process_response()`
Tries to parse the LLM's reply as JSON. If it parses and contains a recognized tool name, the matching Python function actually runs and its result is returned. If parsing fails (i.e., the model just replied in plain language), that reply is passed through unchanged.

### 6. Text-to-speech — `speak()`
Piper TTS converts the final text response into a `.wav` file, which is then played back through your speakers with `sounddevice`.

### 7. The main loop — `run_voice_agent()`
Ties every step together: record → transcribe → reason → (maybe) call a tool → speak → repeat, until you say "exit," "quit," or "stop."

---

## A Note on Latency

Response time matters much more in voice apps than in text chat — a 5-second wait feels fine for a chatbot reply but frustrating for a spoken one. This version is intentionally simple and synchronous. Production voice agents typically stream speech recognition, LLM tokens, and TTS audio to cut down perceived latency.

---

## Customization Ideas

- Replace fixed 5-second recording with real Voice Activity Detection (VAD)
- Add more tools (weather, calendar, search, database lookups)
- Swap `llama3.2` for a larger/smaller Ollama model depending on your hardware
- Try a different Whisper model size (`small`, `medium`, `large-v3`) for better accuracy
- Add streaming so the agent starts speaking before the full response is generated
- Wrap the whole thing in LangChain or LangGraph once you're comfortable with the raw pipeline

---

## Credit

Based on the tutorial ["Build a Voice AI Agent From Scratch"](https://amanxai.com/2026/07/15/build-a-voice-ai-agent-from-scratch/) by Aman Kharwal.
