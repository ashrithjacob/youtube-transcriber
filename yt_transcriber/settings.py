"""Environment configuration, loaded from the process env and `.env`.

Only the visual track needs anything here; plain audio transcription runs
without a `.env` at all.
"""

from __future__ import annotations

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TranscriberSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter (vision-language model for on-screen content)
    openrouter_api_key: SecretStr | None = None
    openrouter_vision_model: str = "z-ai/glm-5.3-flash"
    """Must accept image input; text-only models such as z-ai/glm-5.2 will not work."""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_http_referer: str | None = None
    openrouter_x_title: str | None = "yt-transcriber"
    openrouter_timeout_s: float = 120.0

    # Keyframe selection
    vision_scene_threshold: float = 0.3
    """ffmpeg scene-change score (0-1) above which a keyframe is kept."""
    vision_max_gap_s: float = 60.0
    """Take a frame at least this often even if the scene looks unchanged."""
    vision_dedup_distance: int = 6
    """Max dHash Hamming distance (of 64 bits) for a frame to count as a duplicate."""
    vision_max_frames: int = 300
    """Hard cap on frames sent to the VLM per video (cost guard)."""
    vision_concurrency: int = 8

    @field_validator("openrouter_api_key", "openrouter_http_referer", "openrouter_x_title", mode="before")
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        return None if isinstance(v, str) and not v.strip() else v
