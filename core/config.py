"""
Configuration management for AgenticVideo.

Centralizes all configuration including:
- API keys and endpoints
- Model selections
- Quality tier settings
- Infrastructure endpoints
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class APIConfig:
    """API configuration for video generation services."""

    # Aggregator APIs (Kie AI / Fal AI)
    kie_api_key: str = field(default_factory=lambda: os.getenv("KIE_API_KEY", ""))
    kie_api_base: str = "https://api.kie.ai/api/v1"

    fal_api_key: str = field(default_factory=lambda: os.getenv("FAL_API_KEY", ""))
    fal_api_base: str = "https://api.fal.ai"

    # Direct APIs (fallback)
    runway_api_key: str = field(default_factory=lambda: os.getenv("RUNWAY_API_KEY", ""))
    sora_api_key: str = field(default_factory=lambda: os.getenv("SORA_API_KEY", ""))

    # Audio
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))

    # LLM
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    pool_min_size: int = 2
    pool_max_size: int = 10


@dataclass
class StorageConfig:
    """Storage configuration for generated assets."""
    r2_account_id: str = field(default_factory=lambda: os.getenv("R2_ACCOUNT_ID", ""))
    r2_access_key: str = field(default_factory=lambda: os.getenv("R2_ACCESS_KEY", ""))
    r2_secret_key: str = field(default_factory=lambda: os.getenv("R2_SECRET_KEY", ""))
    r2_bucket: str = field(default_factory=lambda: os.getenv("R2_BUCKET", "agentic-video"))
    r2_public_url: str = field(default_factory=lambda: os.getenv("R2_PUBLIC_URL", ""))


@dataclass
class GPUConfig:
    """Local GPU configuration for Wan 2.1 and motion extraction."""
    enabled: bool = field(default_factory=lambda: os.getenv("GPU_ENABLED", "false").lower() == "true")
    device: str = "cuda:0"
    wan_model_path: str = field(default_factory=lambda: os.getenv("WAN_MODEL_PATH", "/models/wan-2.1"))
    dwpose_model_path: str = field(default_factory=lambda: os.getenv("DWPOSE_MODEL_PATH", "/models/dwpose"))
    cotracker_model_path: str = field(default_factory=lambda: os.getenv("COTRACKER_MODEL_PATH", "/models/cotracker"))


@dataclass
class ModelConfig:
    """Model selection configuration."""

    # Video generation models available through Kie.ai aggregator
    # See: https://kie.ai/market
    premium_models: list[str] = field(default_factory=lambda: [
        "veo-3.1",          # Google Veo 3.1 - best quality with audio
        "veo-3.1-fast",     # Google Veo 3.1 Fast - faster, cheaper
        "runway-aleph",     # Runway Aleph - advanced scene reasoning
        "sora-2",           # OpenAI Sora 2 - realistic motion
        "kling-2.5",        # Kling 2.5 - WARNING: renders Chinese text
    ])

    bulk_models: list[str] = field(default_factory=lambda: [
        "wan-2.1",          # Wan 2.1 - good balance
        "wan-t2v-1.3b",     # Smaller model for 8GB GPUs
    ])

    # Default model per tier (avoid Kling to prevent Chinese text)
    default_premium: str = "veo-3.1-fast"
    default_bulk: str = "wan-2.1"

    # LLM models
    script_model: str = "claude-sonnet-4-20250514"
    planning_model: str = "claude-sonnet-4-20250514"


@dataclass
class QualityTierConfig:
    """Configuration for quality tiers."""

    # Premium tier (API-based, higher quality)
    premium_max_duration: int = 60  # seconds
    premium_default_fps: int = 24
    premium_default_resolution: str = "1080p"
    premium_cost_per_second: float = 0.10  # Approximate USD

    # Bulk tier (self-hosted, lower cost)
    bulk_max_duration: int = 10  # seconds per generation
    bulk_default_fps: int = 24
    bulk_default_resolution: str = "720p"
    bulk_cost_per_second: float = 0.003  # Approximate USD (GPU time)


@dataclass
class Config:
    """Main configuration class."""

    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    quality: QualityTierConfig = field(default_factory=QualityTierConfig)

    # Orchestration settings
    max_retries: int = 3
    default_quality_tier: Literal["premium", "bulk"] = "premium"

    # Progress streaming
    sse_enabled: bool = True
    websocket_enabled: bool = False  # SSE is simpler for CLI

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls()

    def get_aggregator(self) -> Literal["kie", "fal"]:
        """Determine which aggregator to use based on available keys."""
        if self.api.fal_api_key:
            return "fal"
        if self.api.kie_api_key:
            return "kie"
        raise ValueError("No video API aggregator configured (KIE_API_KEY or FAL_API_KEY required)")

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.api.kie_api_key and not self.api.fal_api_key:
            issues.append("No video API aggregator key configured")

        if not self.database.url:
            issues.append("DATABASE_URL not configured")

        if not self.api.anthropic_api_key:
            issues.append("ANTHROPIC_API_KEY not configured (needed for script generation)")

        return issues


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reload_config():
    """Reload configuration from environment."""
    global _config
    _config = Config.from_env()
