import whisper
import numpy as np
import pyaudio
import wave
import tempfile
import os
import subprocess
from scipy.signal import resample

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "base"

MIC_INDEX = 3          # Your MV7+ device index (check with arecord -l)
RATE = 48000
TARGET_RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 4

PIPER_BIN = "/home/pi/piper/piper"
PIPER_MODEL = "/home/pi/piper_models/hi_IN-pratham-medium.onnx"

# =========================
# LOAD WHISPER
# =========================

print("⏳ Loading Whisper...")
model = whisper.load_model(MODEL_NAME)
print("✅ Whisper loaded\n")

# =========================
# SETUP AUDIO INPUT
# =========================

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt32,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=MIC_INDEX,
    frames_per_buffer=CHUNK
)

# =========================
# RECORD AUDIO
# =========================

def record_audio():
    print("🎤 Speak Hindi...")
    frames = []

    for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    raw_data = b''.join(frames)
    audio_np = np.frombuffer(raw_data, dtype=np.int32)

    # Normalize
    audio_np = audio_np.astype(np.float32)
    max_val = np.max(np.abs(audio_np))
    if max_val != 0:
        audio_np = audio_np / max_val

    # Resample to 16kHz
    number_of_samples = round(len(audio_np) * TARGET_RATE / RATE)
    audio_resampled = resample(audio_np, number_of_samples)

    # Save temporary WAV
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes((audio_resampled * 32767).astype(np.int16).tobytes())

    return temp_path

# =========================
# GENERATE LLM RESPONSE
# =========================

def generate_response(prompt):
    result = subprocess.run(
        ["ollama", "run", "gemma:2b"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

# =========================
# SPEAK USING PIPER
# =========================

def speak_with_piper(text):
    piper = subprocess.Popen(
        [
            PIPER_BIN,
            "--model",
            PIPER_MODEL,
            "--output-raw"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    audio_stream, _ = piper.communicate(text.encode())

    play = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
        stdin=subprocess.PIPE
    )

    play.communicate(audio_stream)

# =========================
# MAIN LOOP
# =========================

try:
    while True:
        temp_audio = record_audio()

        print("🧠 Transcribing...")
        result = model.transcribe(
            temp_audio,
            language="hi",
            fp16=False,
            temperature=0.0
        )

        os.remove(temp_audio)

        user_text = result["text"].strip()

        print("📝 You said:", user_text)

        if len(user_text) < 2:
            print("⚠ No valid speech detected\n")
            continue

        print("🤖 Generating response...")
        response = generate_response(user_text)

        print("💬 Assistant:", response)

        print("🔊 Speaking...")
        speak_with_piper(response)

        print("--------------------------------\n")

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()