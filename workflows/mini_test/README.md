# Mini Test

Minimal PDF consultation demo.

Behavior:
- Read a patient PDF.
- Extract text with PyMuPDF.
- Build patient facts locally from the PDF.
- Send only those facts to OpenRouter `qwen/qwen3.5-9b`.
- Return structured English JSON.

Run:

```bash
/data2/leyizhao/CommTool/.venv/bin/python workflows/mini_test/run.py \
  --pdf "testbase/Fodstad-2021-Assessment and Treatment of Noise.pdf" \
  --output workflows/mini_test/output/fodstad_consult.json
```

The script reads `OPENROUTER_API_KEY` from the repository root `.env`.

Output keys:
- `patient_summary`
- `chief_complaint`
- `key_findings`
- `risk_level`
- `red_flags`
- `preliminary_assessment`
- `recommended_next_steps`
- `questions_to_confirm`
- `disclaimer`