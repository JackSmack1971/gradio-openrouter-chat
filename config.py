from __future__ import annotations  # FIXED: Double underscores, not asterisks
import os
from dataclasses import dataclass
from typing import Optional  # ADDED: For better Python < 3.10 compatibility
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "OpenRouter AI Assistant")
    
    # IMPROVED: More explicit None handling
    referer: Optional[str] = os.getenv("APP_REFERER")
    x_title: Optional[str] = os.getenv("APP_X_TITLE") 
    
    # IMPROVED: Better validation for required API key
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"   # per OpenRouter quickstart
    default_model: str = os.getenv("DEFAULT_MODEL", "openai/gpt-4o")
    
    # IMPROVED: Added validation for numeric values
    temperature: float = max(0.0, min(2.0, float(os.getenv("TEMPERATURE", "0.3"))))
    
    # Safety/perf - IMPROVED: Added bounds checking
    max_input_chars: int = max(100, min(50000, int(os.getenv("MAX_INPUT_CHARS", "8000"))))
    max_history_messages: int = max(1, min(200, int(os.getenv("MAX_HISTORY_MESSAGES", "40"))))
    
    # Rate limit (approx per-IP) - IMPROVED: Added bounds checking
    rate_limit_per_min: int = max(1, min(1000, int(os.getenv("RATE_LIMIT_REQUESTS_PER_MIN", "60"))))
    
    # Analytics
    enable_analytics: bool = os.getenv("ENABLE_ANALYTICS", "true").lower() in ("true", "1", "yes", "on")
    
    # Deploy
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = max(1024, min(65535, int(os.getenv("PORT", "7860"))))  # IMPROVED: Valid port range
    
    # IMPROVED: More robust list parsing with filtering
    trusted_proxies: list[str] = [
        p.strip() 
        for p in os.getenv("TRUSTED_PROXIES", "").split(",") 
        if p.strip() and p.strip() != ""
    ]

    def __post_init__(self):
        """Validate critical settings after initialization."""
        # ADDED: Runtime validation for required fields
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )
        
        # ADDED: Validate base_url format
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url format: {self.base_url}")

# ADDED: Helper function for safe settings access
def get_settings() -> Settings:
    """Get validated settings instance with proper error handling."""
    try:
        return Settings()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected configuration error: {e}")
        raise ValueError("Failed to load configuration") from e

# Create settings instance
settings = get_settings()