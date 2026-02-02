![Orpheus-FASTAPI Banner](https://lex-au.github.io/Orpheus-FastAPI/Banner.png)

# 🚀 ORPHEUS-FASTAPI 2026 EDITION

[![GitHub](https://img.shields.io/github/license/Lex-au/Orpheus-FastAPI)](https://github.com/Lex-au/Orpheus-FastAPI/blob/main/LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-00a393.svg)](https://fastapi.tiangolo.com/)

> **Next-Generation Neural Text-to-Speech Engine** with OpenAI-compatible API, 8 distinct AI voices, emotion tags, and a stunning futuristic web interface. Optimized for RTX GPUs with out-of-the-box LM Studio API support.

[GitHub Repository](https://github.com/TheLocalLab/Orpheus-FastAPI-LMStudio)

---

## ✨ What's New in 2026 Edition

- 🎨 **Futuristic Cyberpunk UI** - Glassmorphism design with neon accents
- ⚡ **Async-First Architecture** - Non-blocking speech generation
- 🔌 **Dual API Endpoints** - Both `/v1/audio/speech` and `/v1/speech/audio`
- 📊 **Enhanced Monitoring** - Health checks, request IDs, and timing headers
- 🛡️ **Better Error Handling** - OpenAI-compatible error responses
- 🎯 **Input Validation** - Pydantic v2 models with strict validation

---

## 🎙️ Voice Demos

Listen to sample outputs with different voices and emotions:
- [Default Test Sample](https://lex-au.github.io/Orpheus-FastAPI/DefaultTest.mp3) - Standard neutral tone
- [Leah Happy Sample](https://lex-au.github.io/Orpheus-FastAPI/LeahHappy.mp3) - Cheerful, upbeat demo
- [Tara Sad Sample](https://lex-au.github.io/Orpheus-FastAPI/TaraSad.mp3) - Emotional, melancholic demo
- [Zac Contemplative Sample](https://lex-au.github.io/Orpheus-FastAPI/ZacContemplative.mp3) - Thoughtful, measured tone

---

## 🖥️ User Interface

![Web User Interface](https://lex-au.github.io/Orpheus-FastAPI/WebUI.png)

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🔗 **OpenAI API Compatible** | Drop-in replacement for `/v1/audio/speech` endpoint |
| 🎭 **8 Neural Voices** | Diverse voice profiles with unique characteristics |
| 💬 **Emotion Tags** | Add laughter, sighs, gasps, and more |
| ⚡ **Real-time Generation** | Optimized for RTX GPUs with CUDA acceleration |
| 🌐 **Modern Web UI** | Responsive interface with waveform visualization |
| 📡 **LM Studio Ready** | Works out-of-the-box with LM Studio Server API |

---

## 📁 Project Structure

```
Orpheus-FastAPI/
├── app.py                # FastAPI server with OpenAI-compatible endpoints
├── requirements.txt      # Python dependencies
├── static/               # Static assets (favicon, images)
├── outputs/              # Generated audio files
├── templates/            # Jinja2 HTML templates
│   └── tts.html          # Futuristic web UI
└── tts_engine/           # Core TTS functionality
    ├── __init__.py       # Package exports
    ├── inference.py      # Token generation & API handling
    └── speechpipe.py     # SNAC audio conversion pipeline
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (RTX series recommended)
- LM Studio or compatible LLM inference server

### Installation

```bash
# Clone the repository
git clone https://github.com/TheLocalLab/Orpheus-FastAPI-LMStudio.git
cd Orpheus-FastAPI-LMStudio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip3 install -r requirements.txt

# Create required directories
mkdir -p outputs static
```

### Start the Server

```bash
python app.py
```

Or with custom options:
```bash
uvicorn app:app --host 0.0.0.0 --port 5005 --reload
```

Access:
- 🌐 **Web Interface**: http://localhost:5005/
- 📚 **API Docs (Swagger)**: http://localhost:5005/docs
- 📖 **ReDoc**: http://localhost:5005/redoc

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/audio/speech` | OpenAI-compatible speech synthesis |
| `POST` | `/v1/speech/audio` | Alternative speech endpoint |
| `POST` | `/speak` | Legacy endpoint (returns JSON) |
| `GET` | `/v1/voices` | List available voices |
| `GET` | `/health` | Health check endpoint |

### Speech Synthesis

**OpenAI-Compatible Endpoint:**
```bash
curl http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus-2026",
    "input": "Hello world! This is a test of the Orpheus TTS system.",
    "voice": "tara",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

**Alternative Endpoint:**
```bash
curl http://localhost:5005/v1/speech/audio \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus-2026",
    "input": "This endpoint also works perfectly!",
    "voice": "leo"
  }' \
  --output speech.wav
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | *required* | Text to convert (max 8192 chars) |
| `model` | string | `"orpheus-2026"` | Model identifier |
| `voice` | string | `"tara"` | Voice selection |
| `response_format` | string | `"wav"` | Output format (only 'wav' supported) |
| `speed` | float | `1.0` | Speed factor (0.5-2.0) |

---

## 🎭 Available Voices

| Voice | Gender | Characteristics |
|-------|--------|-----------------|
| `tara` ⭐ | Female | Conversational, clear |
| `leah` | Female | Warm, gentle |
| `jess` | Female | Energetic, youthful |
| `leo` | Male | Authoritative, deep |
| `dan` | Male | Friendly, casual |
| `mia` | Female | Professional, articulate |
| `zac` | Male | Enthusiastic, dynamic |
| `zoe` | Female | Calm, soothing |

---

## 💫 Emotion Tags

Add expressive emotions to your speech:

```text
"Well, that's interesting <laugh> I hadn't thought of that before."
"Oh no <sigh> that's unfortunate news."
"What?! <gasp> I can't believe it!"
```

| Tag | Effect |
|-----|--------|
| `<laugh>` | Laughter |
| `<chuckle>` | Light chuckle |
| `<sigh>` | Sigh |
| `<gasp>` | Gasp |
| `<yawn>` | Yawn |
| `<groan>` | Groan |
| `<cough>` | Cough |
| `<sniffle>` | Sniffle |

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORPHEUS_API_URL` | `http://127.0.0.1:1234/v1/completions` | LLM inference endpoint |
| `ORPHEUS_API_TIMEOUT` | `120` | Request timeout (seconds) |

---

## 🖥️ LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download the **Orpheus-3b-0.1-ft-Q4_K_M-GGUF** model in the Discover tab
3. Load the model in the Developer Tab (starts API server automatically)

![LM Studio Orpheus Model](https://github.com/TheLocalLab/Orpheus-FastAPI-LMStudio/blob/8c9eb86b4c42a0ce60d8089c1b6a07d3635ae908/static/Lmstudio1.png)
![LM Studio Server API](https://github.com/TheLocalLab/Orpheus-FastAPI-LMStudio/blob/8c9eb86b4c42a0ce60d8089c1b6a07d3635ae908/static/Lmstudio.png)

---

## 🔗 Integration with OpenWebUI

Integrate with [OpenWebUI](https://github.com/open-webui/open-webui) for voice-enabled chat:

1. Start Orpheus-FASTAPI server
2. In OpenWebUI: **Admin Panel → Settings → Audio**
3. Change TTS from Web API to **OpenAI**
4. Set API Base URL to `http://localhost:5005`
5. API Key: `not-needed`
6. TTS Voice: `tara` (or any available voice)
7. TTS Model: `tts-1`

---

## 🛠️ Technical Details

### Architecture

This server acts as a frontend connecting to an external LLM inference server:

1. Text prompts are sent to the inference server
2. Generated tokens are converted to audio using SNAC codec
3. Optimized for RTX GPUs with CUDA acceleration

### Performance Optimizations

- Vectorized tensor operations
- Parallel processing with CUDA streams
- Efficient memory management
- Token and audio caching
- Optimized batch sizes

---

## 📜 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Credits

Originally created by [Lex-au](https://github.com/Lex-au/Orpheus-FastAPI)
Enhanced for 2026 Edition by the community.
