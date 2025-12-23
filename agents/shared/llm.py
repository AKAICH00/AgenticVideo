"""
LLM Client - Gemini 2.0 Flash with LangSmith tracing

Usage:
    from shared.llm import generate_script, generate_visual_prompts

Features:
    - Gemini 2.0 Flash for fast generation
    - LangSmith tracing for observability
    - Structured output with Pydantic models
"""

import os
import json
from typing import Optional, List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Langfuse OSS observability (optional)
try:
    from langfuse.decorators import observe
except ImportError:
    # Fallback: no-op decorator if langfuse not installed
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")  # Self-hosted

# Lazy-load Gemini client
_client = None

def get_client():
    """Get or create the Gemini client (lazy-loaded)."""
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client

# ============================================================
# Output Models
# ============================================================

class ScriptDraft(BaseModel):
    """A complete script for a viral video"""
    hook_line: str = Field(description="Opening hook that grabs attention in first 3 seconds")
    body_text: str = Field(description="Main script content with scene markers like [SCENE 1]")
    call_to_action: str = Field(description="Closing CTA")
    estimated_duration_seconds: int = Field(description="Estimated video duration")
    key_points: List[str] = Field(description="3-5 key takeaways from the script")


class VisualScene(BaseModel):
    """A single visual scene prompt"""
    scene_order: int = Field(description="Order in video (1-indexed)")
    timestamp_start: float = Field(description="Start time in seconds")
    timestamp_end: float = Field(description="End time in seconds")
    visual_prompt: str = Field(description="Detailed prompt for image/video generation")
    scene_type: str = Field(description="Type: opening, nature, architecture, transition, closing")
    recommended_model: str = Field(description="veo3_quality, veo3_fast, kling, flux_pro")


class VisualSceneList(BaseModel):
    """List of visual scenes for a video"""
    scenes: List[VisualScene]


class CritiqueResult(BaseModel):
    """Critic agent evaluation result"""
    approved: bool = Field(description="Whether the content is approved")
    score: int = Field(description="Quality score 1-10")
    issues: List[str] = Field(description="List of issues found")
    suggestions: List[str] = Field(description="Improvement suggestions")
    regenerate: bool = Field(description="Whether to trigger regeneration")


# ============================================================
# Generation Functions
# ============================================================

@observe(name="generate_script")
def generate_script(
    topic: str,
    niche: str,
    format: str = "short",  # short (60s) or long (8min)
    style_notes: Optional[str] = None
) -> ScriptDraft:
    """
    Generate a script for a viral video.
    
    Args:
        topic: The main topic (e.g., "Why you're anxious according to Marcus Aurelius")
        niche: The niche (e.g., "Stoic Philosophy")
        format: "short" for 60s TikTok, "long" for 8min YouTube
        style_notes: Additional style guidance
    
    Returns:
        ScriptDraft with hook, body, CTA
    """
    duration_guide = "60 seconds (120-150 words)" if format == "short" else "8 minutes (1200-1500 words)"
    
    system_prompt = f"""You are an expert viral content scriptwriter for {niche} videos.

Your scripts are:
- Hook-driven: First 3 seconds must grab attention
- Emotionally resonant: Connect with viewer's struggles
- Value-dense: Every sentence teaches something
- Retention-optimized: Use pattern interrupts and curiosity gaps

Format: {format} video ({duration_guide})

Include scene markers like [SCENE 1: Ancient library] for visual cues.
"""

    user_prompt = f"""Write a viral script about: {topic}

{f"Style notes: {style_notes}" if style_notes else ""}

Output a complete script with:
1. Hook line (grabs attention in first 3 seconds)
2. Body with scene markers
3. Call to action
4. Key points (3-5 takeaways)
"""

    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=ScriptDraft,
            temperature=0.7,
            max_output_tokens=4096,
        ),
    )
    
    return ScriptDraft.model_validate_json(response.text)


@observe(name="generate_visual_prompts")
def generate_visual_prompts(
    script_text: str,
    niche: str,
    duration_seconds: int = 60
) -> VisualSceneList:
    """
    Generate visual scene prompts from a script.
    
    Args:
        script_text: The full script with scene markers
        niche: The niche for style consistency
        duration_seconds: Total video duration
    
    Returns:
        List of VisualScene prompts
    """
    system_prompt = f"""You are a visual director for {niche} content.

Your visual prompts:
- Are cinematic and atmospheric
- Use dark academia / philosophical aesthetic
- Specify camera movements (dolly, crane, orbit)
- Include lighting direction (golden hour, soft diffused)
- Are optimized for AI generation (Veo 3, Midjourney, Flux)

Model recommendations:
- veo3_quality: Hero shots, opening/closing (expensive, high quality)
- veo3_fast: Main scenes (balanced)
- kling: Transitions, quick cuts (fast, cheap)
- flux_pro: Static images with Ken Burns effect (cheapest)
"""

    user_prompt = f"""Analyze this script and generate visual scenes:

SCRIPT:
{script_text}

VIDEO DURATION: {duration_seconds} seconds

For each [SCENE] marker, generate a detailed visual prompt.
Distribute scenes across the timeline.
Include variety: nature, architecture, abstract, transitions.
"""

    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=VisualSceneList,
            temperature=0.6,
            max_output_tokens=4096,
        ),
    )
    
    return VisualSceneList.model_validate_json(response.text)


@observe(name="critique_content")
def critique_content(
    content_type: str,  # "script" or "visual"
    content: str,
    niche: str,
    quality_threshold: int = 7
) -> CritiqueResult:
    """
    Critique content for quality and viral potential.
    
    Args:
        content_type: "script" or "visual"
        content: The content to critique
        niche: The niche for context
        quality_threshold: Minimum score to approve (1-10)
    
    Returns:
        CritiqueResult with approval status and feedback
    """
    system_prompt = f"""You are a quality control expert for {niche} viral content.

Evaluation criteria for {content_type}s:
- Hook strength (does it grab attention?)
- Value density (is every moment valuable?)
- Emotional resonance (does it connect?)
- Retention optimization (will viewers stay?)
- Technical quality (execution quality)

Be strict but fair. Score 7+ means "would perform well on social media".
Score below 7 means "needs improvement before publishing".
"""

    user_prompt = f"""Critique this {content_type}:

{content}

Provide:
1. Approval decision (approve if score >= {quality_threshold})
2. Quality score (1-10)
3. Issues found
4. Improvement suggestions
5. Whether to regenerate (yes if score < 5)
"""

    response = get_client().models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=CritiqueResult,
            temperature=0.3,  # More deterministic for evaluation
            max_output_tokens=2048,
        ),
    )
    
    return CritiqueResult.model_validate_json(response.text)


# ============================================================
# Utility Functions
# ============================================================

def get_langsmith_trace_url(run_id: str) -> str:
    """Generate a LangSmith trace URL for debugging."""
    return f"https://smith.langchain.com/o/{LANGSMITH_PROJECT}/runs/{run_id}"


if __name__ == "__main__":
    # Test script generation
    print("Testing LLM module...")
    
    script = generate_script(
        topic="Why you're anxious according to Marcus Aurelius",
        niche="Stoic Philosophy",
        format="short"
    )
    
    print(f"\n=== Generated Script ===")
    print(f"Hook: {script.hook_line}")
    print(f"Duration: {script.estimated_duration_seconds}s")
    print(f"Key points: {script.key_points}")
    print(f"\n{script.body_text[:500]}...")
