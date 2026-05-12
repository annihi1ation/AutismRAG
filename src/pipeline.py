import os
import sys
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import parse_pdf

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    sys.exit(1)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def extract_information(pdf_path):
    # 1. Parse PDF
    print(f"Parsing PDF: {pdf_path}...")
    pdf_text = parse_pdf.parse_pdf(pdf_path)
    if not pdf_text:
        print("Failed to extract text from PDF.")
        return

    # 2. Read template
    template_path = os.path.join(os.path.dirname(__file__), 'table.md')
    try:
        with open(template_path, 'r') as f:
            template_content = f.read()
    except Exception as e:
        print(f"Error reading template file: {e}")
        return

    # 3. Construct Prompt
    prompt = f"""
You are an expert researcher.
Refer to this template and extract information from the text below.
If no information is relevant for a field, keep it blank (null or empty string).

Template (Markdown Table):
{template_content}

Text from PDF:
{pdf_text}

Return the extracted information as a JSON object where keys correspond to the "Extraction Field" in the template.
"""

    # 4. Call API
    print("Calling OpenRouter API (google/gemini-3-flash-preview)...")
    try:
        response = client.chat.completions.create(
            model="google/gemini-3-flash-preview", # Using the model specified by user
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # 5. Save to JSON
        output_path = os.path.splitext(pdf_path)[0] + ".json"
        try:
            # Try to parse the content as JSON to ensure it's valid
            json_content = json.loads(content)
            with open(output_path, 'w') as f:
                json.dump(json_content, f, indent=4)
            print(f"Successfully saved extracted information to: {output_path}")
        except json.JSONDecodeError:
            print("Error: API response was not valid JSON.")
            print("Raw response:")
            print(content)
            # Save raw response just in case
            with open(output_path + ".raw", 'w') as f:
                f.write(content)

    except Exception as e:
        print(f"Error during API call: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to_pdf>")
    else:
        extract_information(sys.argv[1])
