"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TTS Engine - Orpheus 2026 Neural Voice Synthesis                            ║
║                                                                              ║
║  Core components for audio generation:                                       ║
║  • inference.py  - Token generation and LLM API handling                     ║
║  • speechpipe.py - SNAC neural audio codec pipeline                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Make key components available at package level
from .inference import (
    generate_speech_from_api,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    list_available_voices
)
