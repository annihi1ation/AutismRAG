"""Module 1: CaseParser — patient_profile + user_question → case_state.

LLM mode uses ChatLLM via the workflow's prompts.toml; rule-based stub fires
when the LLM is unavailable or its JSON cannot be parsed. Target-behavior
selection runs on the parsed candidates with no numerical score and no
implicit precedence between frequency and intensity.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .loaders import HyperRAGContext

logger = logging.getLogger(__name__)


_FREQUENCY_RANK = {"daily": 4, "weekly": 3, "monthly": 2, "occasional": 1}
_INTENSITY_RANK = {"severe": 3, "moderate": 2, "mild": 1}

_FREQ_PATTERNS = [
    (re.compile(r"\b(daily|every day|each day|every morning|every night)\b", re.I), "daily"),
    (re.compile(r"\b(weekly|every week|several times a week|few times a week)\b", re.I), "weekly"),
    (re.compile(r"\b(monthly|every month)\b", re.I), "monthly"),
    (re.compile(r"\b(occasional|occasionally|rarely|sometimes|once in a while)\b", re.I), "occasional"),
]
_INTENSITY_PATTERNS = [
    (re.compile(r"\b(severe|severely|extreme|extremely|dangerous|life[- ]threatening|major)\b", re.I), "severe"),
    (re.compile(r"\b(moderate|moderately|significant)\b", re.I), "moderate"),
    (re.compile(r"\b(mild|mildly|minor|low[- ]level)\b", re.I), "mild"),
]
_BEHAVIOR_PATTERNS = [
    (re.compile(r"\b(self[- ]injurious behaviou?r|self[- ]injury|head[- ]banging|sib)\b", re.I),
     "self-injurious behavior"),
    (re.compile(r"\baggressi(on|ve)\b", re.I), "aggression"),
    (re.compile(r"\b(elopement|wandering|running away)\b", re.I), "elopement"),
    (re.compile(r"\b(tantrum|meltdown)s?\b", re.I), "tantrum"),
    (re.compile(r"\b(stereotypy|stereotyped|repetitive behaviou?r)\b", re.I), "stereotypy"),
    (re.compile(r"\b(property destruction|destructive behaviou?r)\b", re.I), "property destruction"),
    (re.compile(r"\b(non[- ]compliance|noncompliance)\b", re.I), "non-compliance"),
    (re.compile(r"\b(challenging behaviou?r|problem behaviou?r)\b", re.I), "challenging behavior"),
]
_USER_STRATEGY_RE = re.compile(
    r"(?:how about|can we use|what about|should we (?:try|use)|use of)\s+(.+?)(?:[\?\.\!,]|\bfor\b|\bto\b|$)",
    re.I,
)


def choose_target_behavior(candidates: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """Return (target_behavior, ambiguity).

    No score, no weight. Pick a single target ONLY if one candidate dominates
    on BOTH frequency AND intensity (with strict dominance on at least one).
    Otherwise mark as ambiguous and let the answer enumerate candidates.
    """
    if not candidates:
        return "", False
    if len(candidates) == 1:
        behavior = (candidates[0].get("behavior") or "").strip()
        return behavior, not bool(behavior)

    def freq_rank(c: Dict[str, Any]) -> Optional[int]:
        return _FREQUENCY_RANK.get(str(c.get("frequency", "")).lower())

    def intensity_rank(c: Dict[str, Any]) -> Optional[int]:
        return _INTENSITY_RANK.get(str(c.get("intensity", "")).lower())

    for i, candidate in enumerate(candidates):
        f_i = freq_rank(candidate)
        s_i = intensity_rank(candidate)
        if f_i is None or s_i is None:
            continue
        dominates_others = True
        strict_freq = False
        strict_int = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            f_j = freq_rank(other)
            s_j = intensity_rank(other)
            if f_j is not None and f_i < f_j:
                dominates_others = False
                break
            if s_j is not None and s_i < s_j:
                dominates_others = False
                break
            if f_j is not None and f_i > f_j:
                strict_freq = True
            if s_j is not None and s_i > s_j:
                strict_int = True
        if dominates_others and (strict_freq or strict_int):
            return (candidate.get("behavior") or "").strip(), False
    return "", True


# Aligned with the canonical 18-type entity taxonomy. Two of these are scalar
# (developmental_stage, clinical_severity); the rest are lists of mentions.
PATIENT_FEATURE_LIST_FIELDS = (
    "gene",
    "neurobiology",
    "disorder",
    "trauma",
    "symptom_behavior",
    "symptom_emotion",
    "functional_ability",
    "assessment_tool",
    "diagnostic_framework",
    "medication",
    "therapeutic_approach",
    "educational_approach",
    "person_role",
    "social_environment",
    "intersectional_identity",
    "healthcare_access",
)
PATIENT_FEATURE_SCALAR_FIELDS = ("developmental_stage", "clinical_severity")


def _empty_patient_features() -> Dict[str, Any]:
    pf: Dict[str, Any] = {field: [] for field in PATIENT_FEATURE_LIST_FIELDS}
    for field in PATIENT_FEATURE_SCALAR_FIELDS:
        pf[field] = ""
    return pf


def _empty_case_state() -> Dict[str, Any]:
    return {
        "patient_features": _empty_patient_features(),
        "candidate_challenging_behaviors": [],
        "target_behavior": "",
        "possible_function": "",
        "user_strategy_x": None,
        "ambiguity": False,
    }


def _normalize_parsed(parsed: Dict[str, Any]) -> Dict[str, Any]:
    state = _empty_case_state()
    pf = parsed.get("patient_features") or {}
    for field in PATIENT_FEATURE_LIST_FIELDS:
        state["patient_features"][field] = _ensure_list(pf.get(field))
    for field in PATIENT_FEATURE_SCALAR_FIELDS:
        state["patient_features"][field] = str(pf.get(field) or "")

    candidates = parsed.get("candidate_challenging_behaviors") or []
    cleaned: List[Dict[str, Any]] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        cleaned.append(
            {
                "behavior": str(raw.get("behavior") or "").strip(),
                "frequency": _norm_token(raw.get("frequency"), {"daily", "weekly", "monthly", "occasional"}),
                "intensity": _norm_token(raw.get("intensity"), {"severe", "moderate", "mild"}),
                "antecedent": str(raw.get("antecedent") or ""),
                "consequence": str(raw.get("consequence") or ""),
            }
        )
    state["candidate_challenging_behaviors"] = [c for c in cleaned if c["behavior"]]

    state["possible_function"] = str(parsed.get("possible_function") or "")
    user_strategy = parsed.get("user_strategy_x")
    if isinstance(user_strategy, str) and user_strategy.strip():
        state["user_strategy_x"] = user_strategy.strip()
    return state


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()] if value.strip() else []
    return [str(value)]


def _norm_token(value: Any, allowed: set) -> str:
    if not value:
        return ""
    token = str(value).strip().lower()
    return token if token in allowed else ""


class CaseParser:
    def __init__(self, ctx: HyperRAGContext):
        self.ctx = ctx

    def run(self, patient_profile: str, user_question: str) -> Dict[str, Any]:
        patient_profile = patient_profile or ""
        user_question = user_question or ""

        parsed: Optional[Dict[str, Any]] = None
        if self.ctx.llm is not None:
            parsed = self._llm_parse(patient_profile, user_question)
        if parsed is None:
            parsed = self._stub_parse(patient_profile, user_question)

        case_state = _normalize_parsed(parsed)

        # user_strategy_x extraction is robust enough that we always re-run the
        # regex on the raw user_question and prefer the regex match if the LLM
        # forgot it. (LLMs sometimes drop this field.)
        if not case_state["user_strategy_x"]:
            match = _USER_STRATEGY_RE.search(user_question)
            if match:
                case_state["user_strategy_x"] = match.group(1).strip().rstrip(".?!,")

        target, ambiguity = choose_target_behavior(case_state["candidate_challenging_behaviors"])
        case_state["target_behavior"] = target
        case_state["ambiguity"] = ambiguity
        return case_state

    def _llm_parse(self, patient_profile: str, user_question: str) -> Optional[Dict[str, Any]]:
        section = self.ctx.prompts.get("case_parser") or {}
        system = section.get("system", "")
        user_template = section.get("user", "")
        if not system or not user_template:
            logger.warning("case_parser prompts missing — falling back to stub.")
            return None
        user_prompt = user_template.format(
            patient_profile=patient_profile, user_question=user_question
        )
        try:
            result = self.ctx.llm.chat(system_prompt=system, user_prompt=user_prompt, want_json=True)
        except Exception as err:  # noqa: BLE001
            logger.warning("CaseParser LLM call failed (%s) — using stub.", err)
            return None
        parsed = result.get("json")
        if not isinstance(parsed, dict):
            logger.warning("CaseParser LLM returned non-dict JSON — using stub.")
            return None
        return parsed

    def _stub_parse(self, patient_profile: str, user_question: str) -> Dict[str, Any]:
        text = f"{patient_profile}\n{user_question}"

        pf = _empty_patient_features()

        # DISORDER
        if re.search(r"\b(autism|autistic|asd)\b", text, re.I):
            pf["disorder"].append("autism")
        if re.search(r"\bintellectual disabilit", text, re.I):
            pf["disorder"].append("intellectual disability")
        if re.search(r"\b(adhd|attention[- ]deficit)\b", text, re.I):
            pf["disorder"].append("ADHD")
        if re.search(r"\bdown[' ]s? syndrome\b", text, re.I):
            pf["disorder"].append("Down syndrome")

        # DEVELOPMENTAL_STAGE
        m = re.search(r"\b(\d{1,2})[- ]?year[- ]?old\b", text, re.I)
        if m:
            age = int(m.group(1))
            if age < 6:
                pf["developmental_stage"] = "early childhood"
            elif age < 13:
                pf["developmental_stage"] = "school age"
            elif age < 18:
                pf["developmental_stage"] = "adolescent"
            else:
                pf["developmental_stage"] = "adult"

        # CLINICAL_SEVERITY (overall, not behavior intensity)
        if re.search(r"\b(severe|profound)\s+(autism|id|intellectual)", text, re.I):
            pf["clinical_severity"] = "severe"
        elif re.search(r"\bmoderate\s+(autism|id|intellectual)", text, re.I):
            pf["clinical_severity"] = "moderate"
        elif re.search(r"\bmild\s+(autism|id|intellectual)", text, re.I):
            pf["clinical_severity"] = "mild"

        # FUNCTIONAL_ABILITY
        if re.search(r"\bnonverbal|non[- ]verbal\b", text, re.I):
            pf["functional_ability"].append("nonverbal")
        elif re.search(r"\bminimally verbal\b", text, re.I):
            pf["functional_ability"].append("minimally verbal")
        elif re.search(r"\b(moderate|some) verbal", text, re.I):
            pf["functional_ability"].append("moderate verbal ability")
        elif re.search(r"\bverbal\b", text, re.I):
            pf["functional_ability"].append("verbal")

        # SOCIAL_ENVIRONMENT
        for pat, label in [
            (r"\bself[- ]contained classroom|special education classroom\b", "self-contained classroom"),
            (r"\bmainstream\b", "mainstream classroom"),
            (r"\bgroup home\b", "group home"),
            (r"\bday[- ]?care\b", "daycare"),
            (r"\bhome\b", "home"),
        ]:
            if re.search(pat, text, re.I):
                pf["social_environment"].append(label)

        # PERSON_ROLE (subsumes the old caregiving_context)
        for pat, label in [
            (r"\bmother\b", "mother"),
            (r"\bfather\b", "father"),
            (r"\bparents?\b", "parent"),
            (r"\b(grandparent|grandmother|grandfather)\b", "grandparent"),
            (r"\bsibling\b", "sibling"),
            (r"\bteacher\b", "teacher"),
            (r"\b(caregiver|carer)\b", "caregiver"),
        ]:
            if re.search(pat, text, re.I):
                pf["person_role"].append(label)

        # SYMPTOM_EMOTION
        for pat, label in [
            (r"\banxi(ous|ety)\b", "anxiety"),
            (r"\bdepress\w+\b", "depression"),
            (r"\birritabl\w+\b", "irritability"),
            (r"\bfrustrat\w+\b", "frustration"),
            (r"\bagitat\w+\b", "agitation"),
        ]:
            if re.search(pat, text, re.I):
                pf["symptom_emotion"].append(label)

        # MEDICATION (common psychotropics in ID/ASD)
        for pat, label in [
            (r"\brisperidone\b", "risperidone"),
            (r"\baripiprazole\b", "aripiprazole"),
            (r"\bfluoxetine\b", "fluoxetine"),
            (r"\bsertraline\b", "sertraline"),
            (r"\bmethylphenidate\b", "methylphenidate"),
            (r"\bnaltrexone\b", "naltrexone"),
            (r"\bantipsychotic\w*\b", "antipsychotic"),
        ]:
            if re.search(pat, text, re.I):
                pf["medication"].append(label)

        # THERAPEUTIC_APPROACH
        for pat, label in [
            (r"\b(applied behaviou?r analysis|aba)\b", "ABA"),
            (r"\bsensory integration\b", "sensory integration"),
            (r"\boccupational therapy\b", "occupational therapy"),
            (r"\bspeech therapy\b", "speech therapy"),
            (r"\bcbt|cognitive behaviou?ral therapy\b", "CBT"),
            (r"\bfunctional communication training|fct\b", "FCT"),
        ]:
            if re.search(pat, text, re.I):
                pf["therapeutic_approach"].append(label)

        # EDUCATIONAL_APPROACH
        for pat, label in [
            (r"\bteacch\b", "TEACCH"),
            (r"\bpecs\b", "PECS"),
            (r"\biep\b|individualized education", "IEP"),
            (r"\bvisual support|picture schedule\b", "visual supports"),
        ]:
            if re.search(pat, text, re.I):
                pf["educational_approach"].append(label)

        # ASSESSMENT_TOOL / DIAGNOSTIC_FRAMEWORK
        for pat, label in [
            (r"\bados\b", "ADOS"),
            (r"\bvineland\b", "Vineland"),
            (r"\b(aberrant behaviou?r checklist|abc)\b", "ABC"),
            (r"\biq\s+(test|score)|wisc\b", "IQ assessment"),
        ]:
            if re.search(pat, text, re.I):
                pf["assessment_tool"].append(label)
        for pat, label in [
            (r"\bdsm[- ]?5\b", "DSM-5"),
            (r"\bicd[- ]?(10|11)\b", "ICD"),
        ]:
            if re.search(pat, text, re.I):
                pf["diagnostic_framework"].append(label)

        # TRAUMA
        for pat, label in [
            (r"\b(abuse|neglect|maltreatment)\b", "abuse/neglect history"),
            (r"\bptsd|post[- ]traumatic\b", "PTSD"),
            (r"\btrauma\w*\b", "trauma exposure"),
        ]:
            if re.search(pat, text, re.I):
                pf["trauma"].append(label)

        # HEALTHCARE_ACCESS
        for pat, label in [
            (r"\bwait[- ]?list\b", "long waitlist"),
            (r"\bunmet need\b", "unmet service needs"),
            (r"\b(uninsured|no insurance)\b", "uninsured"),
            (r"\b(medicaid|medicare)\b", "public insurance"),
        ]:
            if re.search(pat, text, re.I):
                pf["healthcare_access"].append(label)

        # SYMPTOM_BEHAVIOR + candidate_challenging_behaviors
        candidates: List[Dict[str, Any]] = []
        for pat, label in _BEHAVIOR_PATTERNS:
            if pat.search(text):
                window = _extract_window(text, pat)
                freq = _first_match_value(window, _FREQ_PATTERNS)
                inten = _first_match_value(window, _INTENSITY_PATTERNS)
                candidates.append(
                    {
                        "behavior": label,
                        "frequency": freq,
                        "intensity": inten,
                        "antecedent": _extract_antecedent(window),
                        "consequence": "",
                    }
                )
                if label not in pf["symptom_behavior"]:
                    pf["symptom_behavior"].append(label)

        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for c in candidates:
            key = c["behavior"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        # Dedupe lists while preserving order.
        for field in PATIENT_FEATURE_LIST_FIELDS:
            seen_field: set = set()
            kept: List[str] = []
            for value in pf[field]:
                if value not in seen_field:
                    seen_field.add(value)
                    kept.append(value)
            pf[field] = kept

        possible_function = ""
        if re.search(r"\btransition", text, re.I):
            possible_function = "possibly triggered by transitions / change in routine"
        elif re.search(r"\bdemand|task|work\b", text, re.I):
            possible_function = "possibly demand-related (escape function hypothesis)"
        elif re.search(r"\battention\b", text, re.I):
            possible_function = "possibly attention-seeking (hypothesis)"
        elif re.search(r"\bsensor", text, re.I):
            possible_function = "possibly sensory-driven (hypothesis)"

        user_strategy = None
        match = _USER_STRATEGY_RE.search(user_question)
        if match:
            user_strategy = match.group(1).strip().rstrip(".?!,")

        return {
            "patient_features": pf,
            "candidate_challenging_behaviors": deduped,
            "possible_function": possible_function,
            "user_strategy_x": user_strategy,
        }


def _extract_window(text: str, pattern: re.Pattern, span: int = 80) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    start = max(0, match.start() - span)
    end = min(len(text), match.end() + span)
    return text[start:end]


def _first_match_value(text: str, patterns: List[Tuple[re.Pattern, str]]) -> str:
    for pat, label in patterns:
        if pat.search(text):
            return label
    return ""


def _extract_antecedent(window: str) -> str:
    m = re.search(r"\bwhen\s+([^.;,!?]+)", window, re.I)
    return m.group(1).strip() if m else ""
