"""
Voice AI Agent — built from scratch, 100% free and open-source.

Pipeline:
  Microphone -> Speech-to-Text (Faster Whisper) -> LLM Reasoning (Llama 3.2 via Ollama)
  -> Tool Calling (Python functions) -> Text-to-Speech (Piper) -> Speaker

Run with: python main.py
See README.md for full setup steps.
"""

import json
import subprocess
from datetime import datetime

import ollama
import sounddevice as sd
from scipy.io.wavfile import write, read
from faster_whisper import WhisperModel

# --- Setup: Whisper model for speech-to-text ---
# "base" is a good starting point on a regular laptop. Options: tiny, base, small, medium, large-v3
whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

PIPER_MODEL = "en_US-lessac-medium.onnx"


# --- Step 1: Record audio from the microphone ---

def record_audio(filename="audio.wav", duration=5):
    sample_rate = 16000

    print("Listening...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    write(filename, sample_rate, audio)

    print("Recording complete.")


# --- Step 2: Convert speech to text ---

def transcribe_audio(filename="audio.wav"):
    segments, info = whisper_model.transcribe(filename)

    text = " ".join(
        segment.text for segment in segments
    )

    return text.strip()


# --- Step 3: Tools the agent can call ---

def get_current_time():
    return datetime.now().strftime("%I:%M %p")


def calculate(expression):
    try:
        return str(eval(expression))
    except Exception:
        return "Unable to calculate the expression."


tools_description = """
Available tools:

1. get_current_time
Use this when the user asks for the current time.

2. calculate
Use this for mathematical calculations.

If a tool is required, respond ONLY with JSON:

{
    "tool": "tool_name",
    "argument": "tool_argument"
}

If no tool is required, respond normally.
"""


# --- Step 4: Ask the LLM to reason about the request ---

def ask_llm(user_text):
    prompt = f"""
You are a helpful Voice AI Agent.

{tools_description}

User request:
{user_text}
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# --- Step 5: Execute a tool if the LLM asked for one ---

def process_response(response):
    try:
        tool_call = json.loads(response)

        tool_name = tool_call.get("tool")
        argument = tool_call.get("argument")

        if tool_name == "get_current_time":
            result = get_current_time()

        elif tool_name == "calculate":
            result = calculate(argument)

        else:
            return response

        return f"The result is {result}"

    except json.JSONDecodeError:
        return response


# --- Step 6: Convert the response to speech and play it ---

def speak(text):
    command = [
        "piper",
        "--model",
        PIPER_MODEL,
        "--output_file",
        "response.wav"
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        text=True
    )

    process.communicate(text)

    rate, audio = read("response.wav")
    sd.play(audio, rate)
    sd.wait()


# --- Step 7: Tie it all together ---

def run_voice_agent():
    print(" Voice AI Agent started. Say 'exit', 'quit', or 'stop' to end.")
    while True:
        record_audio()

        user_text = transcribe_audio()

        print("You:", user_text)

        if user_text.lower() in ["exit", "quit", "stop"]:
            print("Voice Agent stopped.")
            break

        llm_response = ask_llm(user_text)

        final_response = process_response(llm_response)

        print("Agent:", final_response)

        speak(final_response)


if __name__ == "__main__":
    run_voice_agent()
