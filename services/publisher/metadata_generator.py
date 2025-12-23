"""
Metadata Generator for YouTube SEO

Generates optimized titles, descriptions, and tags using AI
to maximize discoverability and engagement.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Literal
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SEOMetadata:
    """Generated SEO-optimized metadata."""
    title: str
    description: str
    tags: list[str]
    hashtags: list[str] = field(default_factory=list)
    category_id: str = "22"
    thumbnail_text: Optional[str] = None
    hook_line: Optional[str] = None

    def to_youtube_tags(self) -> list[str]:
        """Convert to YouTube-compatible tag format."""
        # YouTube allows max 500 chars total for tags
        result = []
        char_count = 0
        for tag in self.tags:
            tag = tag.strip()[:30]  # Max 30 chars per tag
            if char_count + len(tag) + 1 <= 500:
                result.append(tag)
                char_count += len(tag) + 1  # +1 for comma
        return result


@dataclass
class VideoContext:
    """Context about the video for metadata generation."""
    topic: str
    niche: str
    script_hook: Optional[str] = None
    script_summary: Optional[str] = None
    target_audience: Optional[str] = None
    video_duration_seconds: int = 60
    format_type: Literal["short", "long"] = "short"
    trending_keywords: list[str] = field(default_factory=list)


# Category mappings for common niches
NICHE_CATEGORIES = {
    "tech": "28",       # Science & Technology
    "finance": "22",    # People & Blogs (no Finance category)
    "gaming": "20",     # Gaming
    "education": "27",  # Education
    "entertainment": "24",  # Entertainment
    "music": "10",      # Music
    "sports": "17",     # Sports
    "news": "25",       # News & Politics
    "howto": "26",      # Howto & Style
    "comedy": "23",     # Comedy
    "travel": "19",     # Travel & Events
    "pets": "15",       # Pets & Animals
    "autos": "2",       # Autos & Vehicles
    "film": "1",        # Film & Animation
}


# Power words for titles by niche
POWER_WORDS = {
    "tech": ["REVEALED", "Game-Changer", "Mind-Blowing", "Insane", "Revolutionary"],
    "finance": ["Secret", "Wealth", "Rich", "Millionaire", "Passive Income"],
    "gaming": ["EPIC", "Insane", "Impossible", "Pro", "God-Tier"],
    "education": ["Master", "Ultimate Guide", "Everything", "Secret", "Genius"],
    "default": ["Shocking", "Amazing", "Incredible", "Must-See", "Viral"],
}


class MetadataGenerator:
    """
    Generates SEO-optimized metadata for YouTube videos.

    Uses AI to create engaging titles, descriptions, and tags
    that maximize discoverability and click-through rates.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self.model = model
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(self, context: VideoContext) -> SEOMetadata:
        """
        Generate SEO-optimized metadata for a video.

        Args:
            context: Video context with topic, niche, and script info

        Returns:
            SEOMetadata with title, description, tags
        """
        try:
            # Generate using AI
            metadata = await self._generate_with_ai(context)

            # Validate and fix
            metadata = self._validate_metadata(metadata, context)

            return metadata

        except Exception as e:
            logger.error(f"AI generation failed: {e}, using fallback")
            return self._generate_fallback(context)

    async def _generate_with_ai(self, context: VideoContext) -> SEOMetadata:
        """Generate metadata using Gemini API."""
        session = await self._get_session()

        prompt = self._build_prompt(context)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 1024,
            },
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

        async with session.post(
            url,
            params={"key": self.api_key},
            json=payload,
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise ValueError(f"Gemini API error: {error}")

            data = await resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]

            return self._parse_ai_response(text, context)

    def _build_prompt(self, context: VideoContext) -> str:
        """Build the prompt for metadata generation."""
        format_note = "YouTube Short (under 60 seconds)" if context.format_type == "short" else "Long-form YouTube video"

        return f"""Generate SEO-optimized YouTube metadata for this video:

TOPIC: {context.topic}
NICHE: {context.niche}
FORMAT: {format_note}
DURATION: {context.video_duration_seconds} seconds
{"HOOK: " + context.script_hook if context.script_hook else ""}
{"SUMMARY: " + context.script_summary if context.script_summary else ""}
{"TARGET AUDIENCE: " + context.target_audience if context.target_audience else ""}
{"TRENDING KEYWORDS: " + ", ".join(context.trending_keywords) if context.trending_keywords else ""}

Generate the following (be specific and engaging):

TITLE: (max 60 chars, include power word, create curiosity gap)

DESCRIPTION: (first 2 lines are critical for SEO, include CTA, 150-300 words total)

TAGS: (15-20 relevant tags, mix of broad and specific)

HASHTAGS: (3-5 hashtags for shorts)

THUMBNAIL_TEXT: (2-4 words for thumbnail overlay)

HOOK_LINE: (one punchy line for video hook)

Format your response exactly like:
TITLE: [title here]
DESCRIPTION: [description here]
TAGS: tag1, tag2, tag3, ...
HASHTAGS: #hash1 #hash2 #hash3
THUMBNAIL_TEXT: [text]
HOOK_LINE: [hook]"""

    def _parse_ai_response(self, text: str, context: VideoContext) -> SEOMetadata:
        """Parse the AI response into SEOMetadata."""
        lines = text.strip().split("\n")

        title = ""
        description = ""
        tags = []
        hashtags = []
        thumbnail_text = None
        hook_line = None

        current_field = None
        description_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
                current_field = "title"
            elif line.startswith("DESCRIPTION:"):
                desc_content = line.replace("DESCRIPTION:", "").strip()
                if desc_content:
                    description_lines.append(desc_content)
                current_field = "description"
            elif line.startswith("TAGS:"):
                tags_str = line.replace("TAGS:", "").strip()
                tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                current_field = "tags"
            elif line.startswith("HASHTAGS:"):
                hashtags_str = line.replace("HASHTAGS:", "").strip()
                hashtags = [h.strip() for h in hashtags_str.split() if h.startswith("#")]
                current_field = "hashtags"
            elif line.startswith("THUMBNAIL_TEXT:"):
                thumbnail_text = line.replace("THUMBNAIL_TEXT:", "").strip()
                current_field = "thumbnail"
            elif line.startswith("HOOK_LINE:"):
                hook_line = line.replace("HOOK_LINE:", "").strip()
                current_field = "hook"
            elif current_field == "description" and line:
                description_lines.append(line)

        description = "\n".join(description_lines)

        # Add standard footer to description
        description = self._add_description_footer(description, context)

        return SEOMetadata(
            title=title,
            description=description,
            tags=tags,
            hashtags=hashtags,
            category_id=NICHE_CATEGORIES.get(context.niche, "22"),
            thumbnail_text=thumbnail_text,
            hook_line=hook_line,
        )

    def _add_description_footer(self, description: str, context: VideoContext) -> str:
        """Add standard SEO footer to description."""
        footer = f"""

---
🔔 Subscribe for more {context.niche} content!
👍 Like if you found this valuable
💬 Comment your thoughts below

#shorts #{context.niche} #{context.topic.replace(" ", "")}
"""
        return description + footer

    def _validate_metadata(self, metadata: SEOMetadata, context: VideoContext) -> SEOMetadata:
        """Validate and fix metadata to meet YouTube requirements."""
        # Title: max 100 chars (but 60 is optimal)
        if len(metadata.title) > 100:
            metadata.title = metadata.title[:97] + "..."

        # Description: max 5000 chars
        if len(metadata.description) > 5000:
            metadata.description = metadata.description[:4997] + "..."

        # Tags: max 500 chars total
        metadata.tags = metadata.to_youtube_tags()

        # Ensure we have basic tags
        if len(metadata.tags) < 5:
            metadata.tags.extend([
                context.niche,
                context.topic.split()[0] if context.topic else "",
                "viral",
                "trending",
            ])

        # Ensure category is valid
        if metadata.category_id not in NICHE_CATEGORIES.values():
            metadata.category_id = "22"

        return metadata

    def _generate_fallback(self, context: VideoContext) -> SEOMetadata:
        """Generate basic metadata without AI."""
        power_words = POWER_WORDS.get(context.niche, POWER_WORDS["default"])
        power_word = power_words[0] if power_words else ""

        title = f"{power_word}: {context.topic}"[:60]

        description = f"""🔥 {context.topic}

{context.script_summary or f"Discover the truth about {context.topic} in this video."}

---
🔔 Subscribe for more {context.niche} content!
👍 Like if you found this valuable
💬 Comment your thoughts below

#shorts #{context.niche}
"""

        tags = [
            context.niche,
            context.topic,
            "viral",
            "trending",
            "2024",
            f"{context.niche} tips",
            f"{context.niche} advice",
        ]

        if context.trending_keywords:
            tags.extend(context.trending_keywords[:5])

        hashtags = [f"#{context.niche}", "#shorts", "#viral"]

        return SEOMetadata(
            title=title,
            description=description,
            tags=tags,
            hashtags=hashtags,
            category_id=NICHE_CATEGORIES.get(context.niche, "22"),
            thumbnail_text=context.topic.split()[0].upper() if context.topic else "WATCH",
            hook_line=f"You won't believe what we discovered about {context.topic}",
        )

    async def generate_ab_variants(
        self,
        context: VideoContext,
        num_variants: int = 3,
    ) -> list[SEOMetadata]:
        """
        Generate multiple title/thumbnail variants for A/B testing.

        Args:
            context: Video context
            num_variants: Number of variants to generate

        Returns:
            List of SEOMetadata variants
        """
        variants = []

        # Generate base metadata
        base = await self.generate(context)
        variants.append(base)

        # Generate additional title variants
        for i in range(num_variants - 1):
            variant = SEOMetadata(
                title=await self._generate_title_variant(context, i),
                description=base.description,
                tags=base.tags,
                hashtags=base.hashtags,
                category_id=base.category_id,
                thumbnail_text=await self._generate_thumbnail_variant(context, i),
                hook_line=base.hook_line,
            )
            variants.append(variant)

        return variants

    async def _generate_title_variant(self, context: VideoContext, variant_num: int) -> str:
        """Generate a title variant for A/B testing."""
        styles = [
            "question",      # "Did You Know...?"
            "number",        # "5 Reasons Why..."
            "controversy",   # "Why Everyone Is Wrong About..."
        ]

        style = styles[variant_num % len(styles)]

        session = await self._get_session()

        prompt = f"""Generate a YouTube title for this topic in {style} style:
TOPIC: {context.topic}
NICHE: {context.niche}

Style guide:
- question: Start with a question
- number: Start with a number/list
- controversy: Challenge conventional thinking

Return ONLY the title, max 60 characters."""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.9, "maxOutputTokens": 100},
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

        try:
            async with session.post(url, params={"key": self.api_key}, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()[:60]
        except Exception as e:
            logger.warning(f"Title variant generation failed: {e}")

        # Fallback
        power_words = POWER_WORDS.get(context.niche, POWER_WORDS["default"])
        return f"{power_words[variant_num % len(power_words)]}: {context.topic}"[:60]

    async def _generate_thumbnail_variant(self, context: VideoContext, variant_num: int) -> str:
        """Generate thumbnail text variant."""
        variants = [
            context.topic.split()[0].upper() if context.topic else "WATCH",
            "MUST SEE",
            "😱 WOW",
        ]
        return variants[variant_num % len(variants)]


# Convenience function for orchestrator integration
async def generate_metadata(
    topic: str,
    niche: str,
    script_hook: Optional[str] = None,
    script_summary: Optional[str] = None,
    format_type: str = "short",
    **kwargs,
) -> dict:
    """
    Generate SEO metadata for a video.

    Returns:
        Dict with title, description, tags, etc.
    """
    generator = MetadataGenerator()

    try:
        context = VideoContext(
            topic=topic,
            niche=niche,
            script_hook=script_hook,
            script_summary=script_summary,
            format_type=format_type,
            **kwargs,
        )

        metadata = await generator.generate(context)

        return {
            "title": metadata.title,
            "description": metadata.description,
            "tags": metadata.tags,
            "hashtags": metadata.hashtags,
            "category_id": metadata.category_id,
            "thumbnail_text": metadata.thumbnail_text,
            "hook_line": metadata.hook_line,
        }

    finally:
        await generator.close()
