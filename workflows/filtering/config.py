"""
Configuration for the Literature Filtering Agent.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model to use (can be changed to any OpenRouter-supported model)
# Examples: "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-001"
MODEL = "google/gemini-3-flash-preview"

# File paths
INPUT_CSV = "data/metadata/CT_filtered.csv"
OUTPUT_CSV = "workflows/filtering/CT_selected.csv"

# Batch settings
RATE_LIMIT_DELAY = 0.5  # seconds between API calls to avoid rate limiting
MAX_PAPERS = None  # Maximum number of papers to process (set to None for all papers)
