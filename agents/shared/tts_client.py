"""
ElevenLabs TTS Client with Word-Level Timestamps

Usage:
    from shared.tts_client import generate_speech_with_timestamps
    
    result = await generate_speech_with_timestamps(
        text="Why you're anxious...",
        voice_id="marcus"
    )
    
    # Returns audio bytes + word-level timestamps for subtitles

Features:
    - Native ElevenLabs "with-timestamps" endpoint
    - Word-level timing for Remotion subtitles
    - Automatic subtitle formatting
"""

import os
import base64
import httpx
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
# Langfuse OSS observability (optional)
try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func): return func
        return decorator

# Environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Voice presets for different content types
VOICE_PRESETS = {
    "marcus": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Default, replace with custom
        "stability": 0.7,
        "similarity_boost": 0.8,
        "style": 0.5,
    },
    "serene": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Female calm voice
        "stability": 0.8,
        "similarity_boost": 0.7,
        "style": 0.3,
    },
    "chronicle": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",  # Documentary style
        "stability": 0.6,
        "similarity_boost": 0.85,
        "style": 0.4,
    },
}


# ============================================================
# Models
# ============================================================

class SubtitleWord(BaseModel):
    """A single word with timing for subtitles"""
    word: str
    start: float  # Start time in seconds
    end: float    # End time in seconds


class TTSResult(BaseModel):
    """Result from TTS generation"""
    audio_bytes: bytes
    audio_base64: str
    subtitles: List[SubtitleWord]
    duration_seconds: float
    character_count: int
    estimated_cost: float


# ============================================================
# TTS Functions
# ============================================================

@observe(name="generate_speech_with_timestamps", run_type="tool")
async def generate_speech_with_timestamps(
    text: str,
    voice: str = "marcus",  # Preset name or voice_id
    model_id: str = "eleven_flash_v2_5",
    output_format: str = "mp3_44100_128",
) -> TTSResult:
    """
    Generate speech with word-level timestamps for subtitles.
    
    Args:
        text: Text to synthesize
        voice: Voice preset name or ElevenLabs voice_id
        model_id: ElevenLabs model to use
        output_format: Audio format
    
    Returns:
        TTSResult with audio and subtitle timing
    """
    # Get voice settings
    if voice in VOICE_PRESETS:
        preset = VOICE_PRESETS[voice]
        voice_id = preset["voice_id"]
        voice_settings = {
            "stability": preset["stability"],
            "similarity_boost": preset["similarity_boost"],
            "style": preset.get("style", 0.5),
        }
    else:
        voice_id = voice
        voice_settings = {"stability": 0.7, "similarity_boost": 0.75}
    
    # Call ElevenLabs with-timestamps endpoint
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{ELEVENLABS_BASE_URL}/text-to-speech/{voice_id}/with-timestamps",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": model_id,
                "voice_settings": voice_settings,
                "output_format": output_format,
            },
        )
        response.raise_for_status()
        data = response.json()
    
    # Extract audio
    audio_base64 = data.get("audio_base64", "")
    audio_bytes = base64.b64decode(audio_base64) if audio_base64 else b""
    
    # Parse alignment data
    alignment = data.get("normalized_alignment") or data.get("alignment", {})
    characters = alignment.get("characters", [])
    start_times = alignment.get("character_start_times_seconds", [])
    end_times = alignment.get("character_end_times_seconds", [])
    
    # Convert to SubtitleWord format
    subtitles = []
    for word, start, end in zip(characters, start_times, end_times):
        if word.strip():  # Skip empty strings
            subtitles.append(SubtitleWord(
                word=word.strip(),
                start=start,
                end=end,
            ))
    
    # Calculate duration
    duration = max(end_times) if end_times else 0.0
    
    # Estimate cost (~$0.30 per 1000 characters)
    char_count = len(text)
    estimated_cost = (char_count / 1000) * 0.30
    
    return TTSResult(
        audio_bytes=audio_bytes,
        audio_base64=audio_base64,
        subtitles=subtitles,
        duration_seconds=duration,
        character_count=char_count,
        estimated_cost=estimated_cost,
    )


def format_subtitles_for_remotion(subtitles: List[SubtitleWord]) -> List[Dict[str, Any]]:
    """
    Format subtitles for Remotion's SubtitleWord[] component.
    
    Args:
        subtitles: List of SubtitleWord from TTS
    
    Returns:
        List of dicts compatible with Remotion
    """
    return [
        {
            "word": s.word,
            "start": s.start,
            "end": s.end,
        }
        for s in subtitles
    ]


def group_subtitles_by_sentence(
    subtitles: List[SubtitleWord],
    max_words_per_group: int = 8,
) -> List[Dict[str, Any]]:
    """
    Group words into sentence-level subtitles for better readability.
    
    Args:
        subtitles: Word-level subtitles
        max_words_per_group: Maximum words per subtitle group
    
    Returns:
        List of subtitle groups with text and timing
    """
    groups = []
    current_group = []
    current_start = None
    
    for word in subtitles:
        if current_start is None:
            current_start = word.start
        
        current_group.append(word.word)
        
        # Check if we should end this group
        is_sentence_end = word.word.endswith(('.', '!', '?', ':'))
        is_max_length = len(current_group) >= max_words_per_group
        
        if is_sentence_end or is_max_length:
            groups.append({
                "text": " ".join(current_group),
                "start": current_start,
                "end": word.end,
            })
            current_group = []
            current_start = None
    
    # Don't forget remaining words
    if current_group:
        groups.append({
            "text": " ".join(current_group),
            "start": current_start,
            "end": subtitles[-1].end if subtitles else 0,
        })
    
    return groups


# ============================================================
# SRT/VTT Export
# ============================================================

def export_to_srt(subtitles: List[SubtitleWord]) -> str:
    """Export word-level subtitles to SRT format."""
    groups = group_subtitles_by_sentence(subtitles)
    srt_lines = []
    
    for i, group in enumerate(groups, 1):
        start = _format_srt_time(group["start"])
        end = _format_srt_time(group["end"])
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(group["text"])
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing TTS client...")
        
        result = await generate_speech_with_timestamps(
            text="Why are you anxious? Marcus Aurelius answered this question two thousand years ago.",
            voice="marcus",
        )
        
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Characters: {result.character_count}")
        print(f"Estimated cost: ${result.estimated_cost:.3f}")
        print(f"Words: {len(result.subtitles)}")
        
        # Show first few subtitles
        for word in result.subtitles[:5]:
            print(f"  [{word.start:.2f}-{word.end:.2f}] {word.word}")
    
    asyncio.run(test())
