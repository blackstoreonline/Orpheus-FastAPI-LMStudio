# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ORPHEUS-FASTAPI 2026 EDITION                                                ║
# ║  Next-Generation Neural Text-to-Speech Engine                                ║
# ║  Originally by Lex-au • Enhanced for the Future                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import time
import uuid
import json
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Union, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, HTTPException, Depends, Query, Header, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from tts_engine import generate_speech_from_api, AVAILABLE_VOICES, DEFAULT_VOICE

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan context manager for startup/shutdown events."""
    # Startup
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  🚀 ORPHEUS-FASTAPI 2026 EDITION                             ║")
    print("║  Neural Voice Synthesis Engine Initializing...               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    yield
    # Shutdown
    print("⚡ Orpheus-FastAPI shutting down gracefully...")

# Create FastAPI app with modern configuration
app = FastAPI(
    title="Orpheus-FASTAPI 2026",
    description="""
## 🎙️ Next-Generation Neural Text-to-Speech Engine

**Orpheus-FASTAPI 2026** delivers state-of-the-art voice synthesis with:
- 🎭 8 Distinct Neural Voice Profiles
- 💫 Emotion Tags for Expressive Speech
- ⚡ Real-time Streaming Capability
- 🔌 OpenAI API v1 Compatible Endpoints

### Supported Endpoints
- `POST /v1/audio/speech` - OpenAI-compatible speech synthesis
- `POST /v1/speech/audio` - Alternative endpoint (same functionality)

### Voice Options
`tara` | `leah` | `jess` | `leo` | `dan` | `mia` | `zac` | `zoe`
    """,
    version="2026.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Speech Synthesis", "description": "Text-to-Speech generation endpoints"},
        {"name": "System", "description": "Health checks and system information"},
        {"name": "Web Interface", "description": "Browser-based TTS interface"},
    ]
)

# Add CORS middleware for modern API compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# STATIC FILES & TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

# Mount directories for serving files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# ═══════════════════════════════════════════════════════════════════════════════
# API MODELS (OpenAI v1 Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request model."""
    model: str = Field(
        default="orpheus-2026",
        description="The model to use for speech synthesis"
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="The text to synthesize into speech"
    )
    voice: str = Field(
        default=DEFAULT_VOICE,
        description="The voice to use for synthesis"
    )
    response_format: Literal["wav"] = Field(
        default="wav",
        description="The audio format for the output (currently only 'wav' is supported)"
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speed of speech (0.5 to 2.0)"
    )

    @field_validator('voice')
    @classmethod
    def validate_voice(cls, v):
        if v not in AVAILABLE_VOICES:
            raise ValueError(f"Voice must be one of: {', '.join(AVAILABLE_VOICES)}")
        return v

class SpeechResponse(BaseModel):
    """Response model for speech generation status."""
    id: str = Field(description="Unique request identifier")
    object: str = Field(default="audio.speech", description="Object type")
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used for synthesis")
    voice: str = Field(description="Voice used for synthesis")
    output_file: str = Field(description="Path to generated audio file")
    generation_time_ms: int = Field(description="Generation time in milliseconds")

class VoiceInfo(BaseModel):
    """Voice information model."""
    id: str
    name: str
    description: str
    gender: str
    preview_url: Optional[str] = None

class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""
    error: dict = Field(description="Error details")

class HealthResponse(BaseModel):
    """System health response."""
    status: str
    version: str
    timestamp: str
    uptime_seconds: float

# Track server start time
SERVER_START_TIME = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"speech-{uuid.uuid4().hex[:16]}"

def create_error_response(code: str, message: str, status_code: int = 400) -> JSONResponse:
    """Create an OpenAI-compatible error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code
            }
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check system health and status."""
    return HealthResponse(
        status="healthy",
        version="2026.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=round(time.time() - SERVER_START_TIME, 2)
    )

@app.get("/v1/voices", tags=["Speech Synthesis"])
async def list_voices():
    """List all available voices with their characteristics."""
    voice_details = {
        "tara": {"description": "Conversational and clear", "gender": "female"},
        "leah": {"description": "Warm and gentle", "gender": "female"},
        "jess": {"description": "Energetic and youthful", "gender": "female"},
        "leo": {"description": "Authoritative and deep", "gender": "male"},
        "dan": {"description": "Friendly and casual", "gender": "male"},
        "mia": {"description": "Professional and articulate", "gender": "female"},
        "zac": {"description": "Enthusiastic and dynamic", "gender": "male"},
        "zoe": {"description": "Calm and soothing", "gender": "female"},
    }
    
    voices = []
    for voice_id in AVAILABLE_VOICES:
        details = voice_details.get(voice_id, {"description": "Neural voice", "gender": "neutral"})
        voices.append(VoiceInfo(
            id=voice_id,
            name=voice_id.capitalize(),
            description=details["description"],
            gender=details["gender"]
        ))
    
    return {"voices": voices, "default": DEFAULT_VOICE}

# ═══════════════════════════════════════════════════════════════════════════════
# SPEECH SYNTHESIS ENDPOINTS (OpenAI v1 Compatible)
# ═══════════════════════════════════════════════════════════════════════════════

async def _generate_speech(request: SpeechRequest) -> tuple[str, float]:
    """Internal function to generate speech from request."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    start = time.time()
    
    # Run in thread pool to not block async
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: generate_speech_from_api(
            prompt=request.input,
            voice=request.voice,
            output_file=output_path
        )
    )
    
    generation_time = time.time() - start
    return output_path, generation_time

@app.post("/v1/audio/speech", tags=["Speech Synthesis"])
async def create_speech_openai(request: SpeechRequest):
    """
    Generate speech from text (OpenAI API v1 compatible).
    
    This endpoint is compatible with OpenAI's `/v1/audio/speech` endpoint format.
    Returns audio data directly as a binary stream.
    """
    if not request.input.strip():
        return create_error_response("invalid_input", "Input text cannot be empty")
    
    try:
        output_path, generation_time = await _generate_speech(request)
        
        # Verify file was created
        if not os.path.exists(output_path):
            return create_error_response(
                "file_not_found",
                "Generated audio file not found",
                500
            )
        
        # Return audio file directly (OpenAI compatibility)
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"speech_{request.voice}.wav",
            headers={
                "X-Generation-Time-Ms": str(int(generation_time * 1000)),
                "X-Voice": request.voice,
                "X-Model": request.model
            }
        )
    except asyncio.CancelledError:
        # Re-raise to let FastAPI handle properly
        raise
    except Exception as e:
        return create_error_response(
            "generation_failed",
            f"Speech generation failed: {str(e)}",
            500
        )

@app.post("/v1/speech/audio", tags=["Speech Synthesis"])
async def create_speech_alternate(request: SpeechRequest):
    """
    Generate speech from text (alternate endpoint).
    
    This is an alternative endpoint path that provides the same functionality
    as `/v1/audio/speech` for flexibility in API routing.
    """
    return await create_speech_openai(request)

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY API ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/speak", tags=["Speech Synthesis"])
async def speak_legacy(request: Request):
    """
    Legacy endpoint for backward compatibility with existing clients.
    
    Returns JSON with file path instead of streaming audio.
    """
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return create_error_response("invalid_json", "Invalid JSON in request body")
    except Exception:
        return create_error_response("invalid_request", "Failed to parse request body")
    
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)

    # Check for None or empty string before calling strip()
    if not text:
        return create_error_response("missing_text", "Missing 'text' field in request")
    
    if not text.strip():
        return create_error_response("missing_text", "Empty 'text' field in request")

    if voice not in AVAILABLE_VOICES:
        return create_error_response(
            "invalid_voice",
            f"Invalid voice '{voice}'. Must be one of: {', '.join(AVAILABLE_VOICES)}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    request_id = generate_request_id()
    
    start = time.time()
    
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: generate_speech_from_api(prompt=text, voice=voice, output_file=output_path)
        )
        
        # Verify file was created
        if not os.path.exists(output_path):
            return create_error_response(
                "file_not_found",
                "Generated audio file not found",
                500
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return create_error_response("generation_failed", f"Speech generation failed: {str(e)}", 500)
    
    generation_time_ms = int((time.time() - start) * 1000)

    return JSONResponse(content={
        "id": request_id,
        "object": "audio.speech",
        "created": int(time.time()),
        "status": "completed",
        "voice": voice,
        "output_file": output_path,
        "generation_time": round(generation_time_ms / 1000, 2),
        "generation_time_ms": generation_time_ms
    })

# ═══════════════════════════════════════════════════════════════════════════════
# WEB INTERFACE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def root(request: Request):
    """Main web interface for TTS generation."""
    return templates.TemplateResponse(
        "tts.html",
        {"request": request, "voices": AVAILABLE_VOICES, "DEFAULT_VOICE": DEFAULT_VOICE}
    )

@app.get("/web/", response_class=HTMLResponse, tags=["Web Interface"])
async def web_ui(request: Request):
    """Alternative web UI route."""
    return templates.TemplateResponse(
        "tts.html",
        {"request": request, "voices": AVAILABLE_VOICES, "DEFAULT_VOICE": DEFAULT_VOICE}
    )

@app.post("/web/", response_class=HTMLResponse, tags=["Web Interface"])
async def generate_from_web(
    request: Request,
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE)
):
    """Handle form submission from web UI."""
    if not text.strip():
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": "Please enter some text to synthesize.",
                "voices": AVAILABLE_VOICES,
                "DEFAULT_VOICE": DEFAULT_VOICE
            }
        )
    
    if voice not in AVAILABLE_VOICES:
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": f"Invalid voice '{voice}'. Please select a valid voice.",
                "voices": AVAILABLE_VOICES,
                "DEFAULT_VOICE": DEFAULT_VOICE
            }
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    start = time.time()
    
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: generate_speech_from_api(prompt=text, voice=voice, output_file=output_path)
        )
        
        # Verify file was created
        if not os.path.exists(output_path):
            return templates.TemplateResponse(
                "tts.html",
                {
                    "request": request,
                    "error": "Generation completed but audio file not found",
                    "voices": AVAILABLE_VOICES,
                    "DEFAULT_VOICE": DEFAULT_VOICE
                }
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": f"Generation failed: {str(e)}",
                "voices": AVAILABLE_VOICES,
                "DEFAULT_VOICE": DEFAULT_VOICE
            }
        )
    
    generation_time = round(time.time() - start, 2)
    
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "success": True,
            "text": text,
            "voice": voice,
            "output_file": output_path,
            "generation_time": generation_time,
            "voices": AVAILABLE_VOICES,
            "DEFAULT_VOICE": DEFAULT_VOICE
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SERVER ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  🚀 ORPHEUS-FASTAPI 2026 EDITION                             ║")
    print("║  Next-Generation Neural Text-to-Speech Engine                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  📡 Server: http://0.0.0.0:5005")
    print("  📚 API Docs: http://0.0.0.0:5005/docs")
    print("  🎙️ Voices: tara, leah, jess, leo, dan, mia, zac, zoe")
    print()
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5005,
        reload=True,
        log_level="info"
    )
