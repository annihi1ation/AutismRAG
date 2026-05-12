#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import fitz
from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_QUESTION = "Review these pre-extracted patient facts and produce a cautious preliminary consultation summary."
REQUIRED_KEYS = {
    "patient_summary": "",
    "chief_complaint": "",
    "key_findings": [],
    "risk_level": "unknown",
    "red_flags": [],
    "preliminary_assessment": [],
    "recommended_next_steps": [],
    "questions_to_confirm": [],
    "disclaimer": "",
}

SYSTEM_PROMPT = """You are a cautious English-language clinical pre-consultation assistant.

You will receive patient facts that have already been extracted from a PDF. The PDF parsing step has already happened upstream.
Do not try to reinterpret document formatting, references, or bibliography. Use only the supplied facts.

Rules:
1. Use only the supplied facts. Do not invent labs, diagnoses, medications, timelines, or outcomes.
2. If the facts are insufficient, say so explicitly.
3. Highlight urgent safety signals when present, such as self-harm, suicidality, seizures, altered consciousness, acute breathing difficulty, high fever, severe pain, or sudden regression.
4. Do not provide a definitive diagnosis. This is decision support, not a substitute for in-person care.
5. Return JSON only with exactly these keys:
{
  "patient_summary": "one-sentence summary",
  "chief_complaint": "main presenting issue",
  "key_findings": ["finding 1", "finding 2"],
  "risk_level": "low|medium|high|unknown",
  "red_flags": ["red flag 1"],
  "preliminary_assessment": ["possible issue 1", "possible issue 2"],
  "recommended_next_steps": ["next step 1", "next step 2"],
  "questions_to_confirm": ["question 1"],
  "disclaimer": "brief disclaimer"
}
"""

REFERENCE_MARKERS = (
    "\nreferences\n",
    "\nreference\n",
    "\nbibliography\n",
    "\nliterature cited\n",
)

CATEGORY_KEYWORDS = {
    "presenting_problem_sentences": (
        "chief complaint",
        "presented",
        "reported",
        "problem behavior",
        "symptom",
        "noise",
        "anxiety",
        "pain",
        "aggression",
        "self-injury",
        "self inj",
        "self-harm",
        "tantrum",
        "avoidant",
        "freezing",
        "screaming",
        "crying",
    ),
    "assessment_sentences": (
        "assessment",
        "evaluation",
        "exam",
        "screening",
        "history",
        "admitted",
        "referred",
        "ruled out",
        "functional analysis",
        "baseline",
        "hearing",
    ),
    "intervention_sentences": (
        "treatment",
        "therapy",
        "intervention",
        "medication",
        "cbt",
        "desensitization",
        "reinforcement",
        "coping",
        "extinction",
        "differential reinforcement",
        "systematically increased",
    ),
    "outcome_sentences": (
        "improved",
        "decreased",
        "reduced",
        "resolved",
        "worse",
        "worsened",
        "follow-up",
        "tolerate",
        "independently",
        "to zero",
        "learned",
        "successful desensitization",
        "ability to endure",
    ),
    "red_flag_sentences": (
        "self-injury",
        "self inj",
        "self-harm",
        "suic",
        "seiz",
        "unconscious",
        "altered consciousness",
        "aggression",
        "choking",
        "kicking",
        "forcefully hitting",
        "severe pain",
        "high fever",
        "breath",
        "regression",
    ),
}

SCORING_KEYWORDS = {
    "year-old": 5,
    "male": 2,
    "female": 2,
    "diagnosed": 4,
    "patient": 2,
    "admitted": 3,
    "referred": 3,
    "behavior": 2,
    "anxiety": 3,
    "problem": 2,
    "aggression": 4,
    "self-injury": 4,
    "self inj": 4,
    "pain": 3,
    "seiz": 4,
    "treatment": 2,
    "assessment": 2,
    "therapy": 2,
    "improved": 3,
    "decreased": 3,
    "noise": 2,
    "coping": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PDF-based intelligent consultation demo")
    parser.add_argument("--pdf", required=True, help="Path to the patient PDF")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Optional user question")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter API base URL")
    parser.add_argument("--max-chars", type=int, default=18000, help="Max PDF chars sent to the model")
    parser.add_argument("--output", default="", help="Optional path to save JSON output")
    return parser.parse_args()


def extract_pdf_text(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as document:
        return "\n".join(page.get_text("text") for page in document)


def normalize_text(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r"^Method\s+Participant\s+", "", sentence)
    sentence = re.sub(r"^Study\s+\d+:\s*", "", sentence)
    sentence = re.sub(r"^Keywords:\s*", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()


def is_noise_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    noise_tokens = (
        "keywords:",
        "doi",
        "et al",
        "author information",
        "funding:",
        "conflict of interest",
        "published in final edited form",
        "journal of",
        "assessment and treatment of noise hypersensitivity in a teenager with autism spectrum disorder: a case study. by",
    )
    if any(token in lower for token in noise_tokens):
        return True
    if sentence.count(";") >= 4:
        return True
    if lower.endswith("a case study.") and len(sentence) < 160:
        return True
    if sentence.count("(") >= 3 and re.search(r"\b\d{4}\b", sentence):
        return True
    return False


def is_patient_specific(sentence: str) -> bool:
    lower = sentence.lower()
    patient_tokens = (
        "aaron",
        "the patient",
        "this patient",
        "mother reported",
        "father reported",
        "caregiver reported",
        "at intake",
        "he was",
        "she was",
        "his ",
        "her ",
        "admitted",
        "referred",
    )
    return bool(re.search(r"\b\d{1,2}-year-old\b", lower)) or any(token in lower for token in patient_tokens)


def strip_reference_section(text: str) -> str:
    lowered = text.lower()
    cut_positions = [lowered.find(marker) for marker in REFERENCE_MARKERS if marker in lowered]
    if not cut_positions:
        return text
    cut_at = min(position for position in cut_positions if position >= 0)
    return text[:cut_at].strip()


def split_sentences(text: str) -> list[str]:
    flattened = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", flattened)
    sentences: list[str] = []
    for part in parts:
        cleaned = clean_sentence(part)
        if not 20 <= len(cleaned) <= 700:
            continue
        if is_noise_sentence(cleaned):
            continue
        sentences.append(cleaned)
    return sentences


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def sentence_score(sentence: str) -> int:
    lower = sentence.lower()
    score = 0
    for keyword, weight in SCORING_KEYWORDS.items():
        if keyword in lower:
            score += weight

    if re.search(r"\b\d{1,2}-year-old\b", lower):
        score += 4
    if "diagnosed with" in lower:
        score += 3
    if any(token in lower for token in ("doi", "et al", "journal", "author information", "keywords:")):
        score -= 5
    if re.search(r"\b\d{4}\b", sentence) and "year-old" not in lower:
        score -= 1
    return score


def top_supporting_sentences(sentences: list[str], limit: int = 12) -> list[str]:
    ranked = sorted(((sentence_score(sentence), sentence) for sentence in sentences), reverse=True)
    filtered = [sentence for score, sentence in ranked if score >= 3]
    return dedupe_keep_order(filtered)[:limit]


def collect_category_sentences(sentences: list[str], keywords: tuple[str, ...], limit: int = 6) -> list[str]:
    matches = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    ranked = sorted(matches, key=sentence_score, reverse=True)
    return dedupe_keep_order(ranked)[:limit]


def extract_demographics(sentences: list[str]) -> list[str]:
    matches: list[str] = []
    age_pattern = re.compile(r"\b\d{1,2}-year-old\b", re.IGNORECASE)
    sex_pattern = re.compile(r"\b(male|female|boy|girl|man|woman|teenager|adolescent)\b", re.IGNORECASE)

    for sentence in sentences:
        if age_pattern.search(sentence) or (" was a " in sentence.lower() and sex_pattern.search(sentence)):
            matches.append(sentence)
    return dedupe_keep_order(matches)[:4]


def extract_diagnoses(sentences: list[str]) -> list[str]:
    patterns = (
        re.compile(r"diagnosed with ([^.]+)", re.IGNORECASE),
        re.compile(r"diagnosis of ([^.]+)", re.IGNORECASE),
        re.compile(r"co-occurring ([^.]+)", re.IGNORECASE),
    )
    diagnoses: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if not any(token in lower for token in ("diagnosed", "diagnosis", "co-occurring", "patient", "he ", "she ")):
            continue
        for pattern in patterns:
            for match in pattern.findall(sentence):
                cleaned = re.sub(r"\s+", " ", match).strip(" ,;:")
                cleaned = re.split(r"\b(who|which|and was|and he|and she|\. )\b", cleaned, maxsplit=1)[0].strip(" ,;:")
                cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip(" ,;:")
                if cleaned:
                    diagnoses.append(cleaned)
    return dedupe_keep_order(diagnoses)[:6]


def build_patient_facts(pdf_text: str, max_chars: int) -> dict[str, Any]:
    normalized_text = normalize_text(pdf_text)
    body_text = strip_reference_section(normalized_text)
    body_text = body_text[:max_chars] if len(body_text) > max_chars else body_text
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    title = ""
    for line in lines[:8]:
        lower = line.lower()
        if len(line) > 180:
            continue
        if any(token in lower for token in ("author", "funding", "conflict of interest", "journal", "doi")):
            continue
        title = line
        break

    sentences = split_sentences(body_text)
    patient_specific_sentences = [sentence for sentence in sentences if is_patient_specific(sentence)]
    preferred_sentences = patient_specific_sentences or sentences
    supporting = top_supporting_sentences(preferred_sentences)
    facts = {
        "document_title": title,
        "demographics": extract_demographics(preferred_sentences),
        "known_diagnoses": extract_diagnoses(supporting + preferred_sentences[:20]),
        "presenting_problem_sentences": collect_category_sentences(preferred_sentences, CATEGORY_KEYWORDS["presenting_problem_sentences"]),
        "assessment_sentences": collect_category_sentences(preferred_sentences, CATEGORY_KEYWORDS["assessment_sentences"]),
        "intervention_sentences": collect_category_sentences(preferred_sentences, CATEGORY_KEYWORDS["intervention_sentences"]),
        "outcome_sentences": collect_category_sentences(preferred_sentences, CATEGORY_KEYWORDS["outcome_sentences"]),
        "red_flag_sentences": collect_category_sentences(preferred_sentences, CATEGORY_KEYWORDS["red_flag_sentences"], limit=4),
        "supporting_sentences": supporting,
    }
    facts["non_empty_fact_groups"] = sum(1 for value in facts.values() if value)
    return facts



def cleanup_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(cleaned)

    normalized: dict[str, Any] = {}
    for key, default_value in REQUIRED_KEYS.items():
        value = data.get(key, default_value)
        if isinstance(default_value, list):
            normalized[key] = value if isinstance(value, list) else [str(value)]
        else:
            normalized[key] = value if isinstance(value, str) else str(value)
    if normalized["risk_level"] not in {"low", "medium", "high", "unknown"}:
        normalized["risk_level"] = "unknown"
    return normalized


def build_user_prompt(patient_facts: dict[str, Any], question: str) -> str:
    return (
        "The PDF has already been processed locally. Do not parse the source document again.\n\n"
        f"User request: {question}\n\n"
        "Pre-extracted patient facts (JSON):\n"
        f"{json.dumps(patient_facts, ensure_ascii=False, indent=2)}"
    )


def main() -> int:
    args = parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY is missing.", file=sys.stderr)
        return 1

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    pdf_text = extract_pdf_text(pdf_path).strip()
    if not pdf_text:
        print(f"Error: no text extracted from {pdf_path}", file=sys.stderr)
        return 1

    patient_facts = build_patient_facts(pdf_text, args.max_chars)
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=args.model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(patient_facts, args.question)},
        ],
    )
    raw_output = response.choices[0].message.content or "{}"
    result = cleanup_json(raw_output)

    result["source_pdf"] = str(pdf_path)
    result["model"] = args.model
    result["chars_sent_to_model"] = len(json.dumps(patient_facts, ensure_ascii=False))
    result["extracted_fact_groups"] = patient_facts["non_empty_fact_groups"]

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())