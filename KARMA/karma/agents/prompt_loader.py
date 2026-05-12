import toml
import os
from typing import Dict, Any

def load_prompts_config() -> Dict[str, Any]:
    """Load the prompts configuration from the TOML file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "prompts.toml")
    
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompts file not found at {prompt_file}")
        
    with open(prompt_file, "r", encoding="utf-8") as f:
        return toml.load(f)

def get_agent_config(agent_name: str) -> Dict[str, str]:
    """Get configuration for a specific agent."""
    config = load_prompts_config()
    return config.get(agent_name, {})
