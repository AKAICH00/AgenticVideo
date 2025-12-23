"""
Shared Agent Modules

This package contains shared utilities for all video generation agents:

- db: Database operations (PostgreSQL via NocoDB)
- llm: Gemini 2.0 Flash with Langfuse tracing (OSS)
- kie_client: Kie.ai multi-model API (Veo 3, Midjourney, Flux, etc.)
- tts_client: ElevenLabs TTS with word-level timestamps
- lipsync_client: ComfyUI Wav2Lip (primary, local GPU) + Sync Labs (fallback)
- visual_router: Intelligent task routing

Usage:
    from agents.shared import (
        generate_script,
        generate_visual_prompts,
        generate_speech_with_timestamps,
        apply_lip_sync,
        route_visual_task,
    )
"""

from .db import get_db, Database
from .llm import (
    generate_script,
    generate_visual_prompts,
    critique_content,
    ScriptDraft,
    VisualScene,
    VisualSceneList,
    CritiqueResult,
)
from .kie_client import (
    KieClient,
    generate_video,
    generate_image,
    GenerationResult,
)
from .tts_client import (
    generate_speech_with_timestamps,
    format_subtitles_for_remotion,
    group_subtitles_by_sentence,
    export_to_srt,
    TTSResult,
    SubtitleWord,
)
from .lipsync_client import (
    apply_lip_sync,
    sync_labs_lipsync,
    comfyui_wav2lip,
    estimate_lipsync_cost,
    LipSyncResult,
)
from .visual_router import (
    route_visual_task,
    estimate_cost,
    generate_scene,
    generate_all_scenes,
    VisualScene as RouterVisualScene,
    SceneResult,
)

__all__ = [
    # Database
    "get_db",
    "Database",
    # LLM
    "generate_script",
    "generate_visual_prompts",
    "critique_content",
    "ScriptDraft",
    "VisualScene",
    "VisualSceneList",
    "CritiqueResult",
    # Kie.ai
    "KieClient",
    "generate_video",
    "generate_image",
    "GenerationResult",
    # TTS
    "generate_speech_with_timestamps",
    "format_subtitles_for_remotion",
    "group_subtitles_by_sentence",
    "export_to_srt",
    "TTSResult",
    "SubtitleWord",
    # Lip Sync
    "apply_lip_sync",
    "sync_labs_lipsync",
    "comfyui_wav2lip",
    "estimate_lipsync_cost",
    "LipSyncResult",
    # Router
    "route_visual_task",
    "estimate_cost",
    "generate_scene",
    "generate_all_scenes",
    "RouterVisualScene",
    "SceneResult",
]
