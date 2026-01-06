import os
import time
from dotenv import load_dotenv
import whisperx
from whisperx.diarize import DiarizationPipeline
import torch

# 1. Load .env
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise RuntimeError("HF_TOKEN not found in environment")

# 2. Basic config
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_file = "english-male-female-interview-audio-twzglwtb_pt8ccYN8.mp3"

# -------------------------
# Start total timer
# -------------------------
t_total_start = time.time()

# 3. Load WhisperX ASR model
model = whisperx.load_model(
    "small",
    device=device,
    compute_type="float32"  # GTX 1650 safe
)

# 4. Load audio
audio = whisperx.load_audio(audio_file)

# -------------------------
# ASR timing
# -------------------------
t_asr_start = time.time()
result = model.transcribe(audio)
asr_time = time.time() - t_asr_start

# -------------------------
# Alignment timing
# -------------------------
t_align_start = time.time()
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"],
    device=device
)

result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device
)
align_time = time.time() - t_align_start

# -------------------------
# Diarization timing
# -------------------------
t_diar_start = time.time()
diarize_model = DiarizationPipeline(
    use_auth_token=hf_token,
    device=device
)

diarization = diarize_model(audio)
result = whisperx.assign_word_speakers(diarization, result)
diar_time = time.time() - t_diar_start

# -------------------------
# Total time
# -------------------------
total_time = time.time() - t_total_start

# -------------------------
# Print results
# -------------------------
print("\n===== TIMING =====")
print(f"ASR time        : {asr_time:.2f}s")
print(f"Alignment time  : {align_time:.2f}s")
print(f"Diarization time: {diar_time:.2f}s")
print(f"Total time      : {total_time:.2f}s")
print("==================\n")

# 10. Print transcript
for seg in result["segments"]:
    speaker = seg.get("speaker", "UNK")
    print(f"[{speaker}] {seg['text']}")
