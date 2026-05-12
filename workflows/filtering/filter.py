"""
Literature Filtering Agent - Main Script

This script filters academic literature based on relevance to helping special educators
understand and respond to challenging behaviors in students with intellectual disabilities.
"""
import json
import time
import sys
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from workflows.filtering.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL,
    INPUT_CSV,
    OUTPUT_CSV,
    RATE_LIMIT_DELAY,
    MAX_PAPERS,
)
from workflows.filtering.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def create_client() -> OpenAI:
    """Create OpenRouter client."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Please set it with: export OPENROUTER_API_KEY='your-key-here'"
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


def evaluate_paper(client: OpenAI, title: str, manual_tags: str, abstract: str) -> dict:
    """
    Use LLM to evaluate if a paper is relevant.
    
    Returns:
        dict with keys: relevant (bool), response (str)
    """
    # Handle missing fields
    if pd.isna(abstract) or not str(abstract).strip():
        abstract = "(No abstract available)"
    if pd.isna(manual_tags) or not str(manual_tags).strip():
        manual_tags = "(No tags available)"
    if pd.isna(title) or not str(title).strip():
        title = "(No title available)"
    
    user_prompt = USER_PROMPT_TEMPLATE.format(title=title, manual_tags=manual_tags, abstract=abstract)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent evaluation
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse In/Out response
        content_lower = content.lower().strip()
        is_relevant = content_lower.startswith("in") or content_lower == "in"
        
        return {
            "relevant": is_relevant,
            "response": content,
        }
        
    except Exception as e:
        print(f"  Error evaluating paper: {e}")
        return {
            "relevant": False,
            "response": f"Error: {str(e)}",
        }


def filter_literature(input_path: str = None, output_path: str = None, max_papers: int = None):
    """
    Main function to filter literature from CSV.
    
    Args:
        input_path: Path to input CSV (defaults to config value)
        output_path: Path to output CSV (defaults to config value)
        max_papers: Maximum number of papers to process (defaults to config value)
    """
    input_path = input_path or str(PROJECT_ROOT / INPUT_CSV)
    output_path = output_path or str(PROJECT_ROOT / OUTPUT_CSV)
    max_papers = max_papers if max_papers is not None else MAX_PAPERS
    
    print(f"Loading literature from: {input_path}")
    df = pd.read_csv(input_path)
    total_papers = len(df)
    
    # Limit papers if max_papers is set
    if max_papers is not None and max_papers < len(df):
        df = df.head(max_papers)
        print(f"Total papers in file: {total_papers} (processing first {max_papers})")
    else:
        print(f"Total papers to evaluate: {len(df)}")
    
    # Initialize client
    client = create_client()
    print(f"Using model: {MODEL}")
    print("-" * 60)
    
    # Lists to store results
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering papers"):
        title = row.get("Title", "")
        manual_tags = row.get("Manual Tags", "")
        abstract = row.get("Abstract Note", "")
        
        result = evaluate_paper(client, title, manual_tags, abstract)
        results.append(result)
        
        status = "✓ IN" if result["relevant"] else "✗ OUT"
        tqdm.write(f"  {str(title)[:60]}... {status}")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Add results to dataframe
    df["LLM_Relevant"] = [r["relevant"] for r in results]
    df["LLM_Response"] = [r["response"] for r in results]
    
    # Filter and save relevant papers
    relevant_df = df[df["LLM_Relevant"] == True]
    rejected_df = df[df["LLM_Relevant"] == False]
    
    print("-" * 60)
    print(f"Filtering complete!")
    print(f"  Total papers: {len(df)}")
    print(f"  Relevant papers: {len(relevant_df)}")
    print(f"  Rejected papers: {len(rejected_df)}")
    
    # Save results
    relevant_df.to_csv(output_path, index=False)
    print(f"\nRelevant papers saved to: {output_path}")
    
    # Save rejected papers
    rejected_output = output_path.replace(".csv", "_rejected.csv")
    rejected_df.to_csv(rejected_output, index=False)
    print(f"Rejected papers saved to: {rejected_output}")
    
    # Also save full results with all evaluations
    full_output = output_path.replace(".csv", "_full.csv")
    df.to_csv(full_output, index=False)
    print(f"Full evaluation results saved to: {full_output}")
    
    return relevant_df, rejected_df


if __name__ == "__main__":
    filter_literature()
