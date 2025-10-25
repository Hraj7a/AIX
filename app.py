# app.py
import os
import io
import time
import requests
import streamlit as st

# ---------- Optional but recommended libs ----------
from pypdf import PdfReader                     # pip install pypdf
from docx import Document                       # pip install python-docx
from langdetect import detect, DetectorFactory  # pip install langdetect

# Azure Translator (optional translation)
from azure.ai.translation.text import TextTranslationClient  # pip install azure-ai-translation-text
from azure.core.credentials import AzureKeyCredential        # pip install azure-core

# OpenAI (optional fallback / fusion)
from openai import OpenAI                                  # pip install openai


# =========================
# CONFIG / SECRETS
# =========================
# Prefer Streamlit Secrets; fall back to environment for local dev
HF_MODEL_ID = "mrm8488/T5-base-finetuned-cuad"

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN") or ""
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

AZURE_TRANSLATOR_KEY = st.secrets.get("AZURE_TRANSLATOR_KEY") or os.getenv("AZURE_TRANSLATOR_KEY") or ""
AZURE_TRANSLATOR_ENDPOINT = st.secrets.get("AZURE_TRANSLATOR_ENDPOINT") or os.getenv("AZURE_TRANSLATOR_ENDPOINT") or ""
AZURE_TRANSLATOR_REGION = st.secrets.get("AZURE_TRANSLATOR_REGION") or os.getenv("AZURE_TRANSLATOR_REGION") or ""

# OpenAI client (optional)
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Azure Translator client (optional)
translator_client = None
if AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT:
    translator_client = TextTranslationClient(
        endpoint=AZURE_TRANSLATOR_ENDPOINT,
        credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
    )

# Make langdetect deterministic
DetectorFactory.seed = 0


# =========================
# HELPERS
# =========================
def build_hf_prompt(text: str, country: str | None = None) -> str:
    """Compose the instruction for the T5 CUAD model."""
    if country:
        return f"""
Based on the following text that is taken from a contract document, and based on the laws of the country specified,
analyze and produce:
1) Extracted contract information (parties, effective dates, governing law, financial terms, etc.)
2) Missing information
3) Potential risks / non-standard clauses
4) Practical changes / opinions / law comparisons
5) A concise summary of #1–#4

Country/Region: {country}

Contract text:
{text}
""".strip()
    else:
        return f"""
Based on the following contract text, analyze and produce:
1) Extracted contract information (parties, effective dates, governing law, financial terms, etc.)
2) Missing information
3) Potential risks / non-standard clauses
4) Practical changes / opinions / law comparisons
5) A concise summary of #1–#4

Contract text:
{text}
""".strip()


def chunk_text(s: str, max_chars: int = 2400) -> list[str]:
    """Split a long string into smaller chunks for safer inference."""
    s = s or ""
    return [s[i:i + max_chars] for i in range(0, len(s), max_chars)] or [""]


def hf_generate(model_id: str, token: str, prompt: str, max_retries: int = 3):
    """
    Call Hugging Face Inference API with retries for 503/429 and normalize common response shapes.
    Returns: (text, error) where only one is non-empty.
    """
    if not token:
        return None, "missing_token"

    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512}}

    for _ in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            return None, f"network_error: {e}"

        # Cold start or rate limit
        if r.status_code in (503, 429):
            wait = 3.0
            try:
                d = r.json()
                wait = float(d.get("estimated_time", wait))
            except Exception:
                pass
            time.sleep(min(wait, 10.0))
            continue

        if r.status_code != 200:
            return None, f"http_{r.status_code}: {r.text[:800]}"

        # Parse JSON
        try:
            data = r.json()
        except ValueError:
            return None, "json_decode_error"

        # Defensive: sometimes API returns {"error": "..."} even with 200
        if isinstance(data, dict) and "error" in data:
            return None, f"api_error: {data['error'][:800]}"

        # Common shapes
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"], None
            if data and isinstance(data[0], dict) and "summary_text" in data[0]:
                return data[0]["summary_text"], None
            if data and isinstance(data[0], str):
                return data[0], None
        if isinstance(data, str):
            return data, None

        return None, "unrecognized_schema"

    return None, "retries_exhausted"


def extract_text_from_file(file) -> str:
    """Extract text from PDF, DOCX, or TXT (no OCR for scanned PDFs)."""
    name = file.name.lower()
    raw = file.read()

    if name.endswith(".pdf"):
        text_pages = []
        pdf = PdfReader(io.BytesIO(raw))
        for page in pdf.pages:
            text_pages.append(page.extract_text() or "")
        return "\n".join(text_pages)

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(raw))
        return "\n".join([p.text for p in doc.paragraphs])

    # Default: treat as text
    return raw.decode("utf-8", errors="ignore")


def translate_text(text: str, to_language: str = "en") -> str:
    """Translate via Azure AI Translator (if configured)."""
    if not translator_client:
        return text
    resp = translator_client.translate(body=[{"text": text}], to_language=[to_language])
    return resp[0].translations[0].text


def gpt_summary(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """Optional GPT fallback/ensemble."""
    if not openai_client:
        return ""
    try:
        r = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Answer clearly in 5–7 bullet points."},
                {"role": "user", "content": prompt}
            ]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"OpenAI error: {e}")
        return ""


# =========================
# UI
# =========================
st.set_page_config(page_title="⚖ Legal Document Analyzer", page_icon="⚖", layout="wide")
st.title("⚖ Multilingual Legal Document Analyzer")

col1, col2 = st.columns(2)
with col1:
    country = st.text_input("Country/Region for legal context (optional)", value="")
with col2:
    show_arabic = st.checkbox("Return Arabic summary when input is Arabic", value=True)

uploaded = st.file_uploader("📂 Upload a contract (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded and st.button("🔍 Analyze"):
    with st.spinner("Extracting text..."):
        text = extract_text_from_file(uploaded)
        if not text.strip():
            st.error("The file appears empty or unreadable. If it's scanned, add OCR.")
            st.stop()

        # Detect language (deterministic)
        try:
            lang = detect(text[:2000])
        except Exception:
            lang = "unknown"
        st.info(f"Detected language: **{lang.upper()}**")

        # Translate Arabic → English for analysis (optional)
        if lang == "ar":
            with st.spinner("Translating Arabic → English..."):
                text_en = translate_text(text, to_language="en")
        else:
            text_en = text

    # ===== Hugging Face analysis =====
    with st.spinner("Analyzing with specialized legal model..."):
        final_prompt = build_hf_prompt(text_en, country=country or None)
        chunks = chunk_text(final_prompt)

        hf_parts, last_err = [], None
        for ch in chunks[:3]:  # first few chunks are usually enough; adjust as needed
            out, err = hf_generate(HF_MODEL_ID, HF_TOKEN, ch, max_retries=3)
            if out:
                hf_parts.append(out.strip())
            else:
                last_err = err
                break

        if hf_parts:
            hf_analysis = "\n\n".join(hf_parts).strip()
        else:
            hf_analysis = ""
            detail = f" (detail: {last_err})" if last_err else ""
            st.warning("Hugging Face analysis unavailable or failed, continuing with GPT only." + detail)

    # ===== GPT fallback / combo =====
    if not hf_analysis:
        gpt_prompt = (
            "Summarize the following contract text focusing on parties, dates, governing law, "
            "financial terms, missing information, and non-standard risks:\n\n" + text_en[:6000]
        )
        gpt_out = gpt_summary(gpt_prompt, model_name="gpt-4o-mini")
        combined = gpt_out
    else:
        combined = hf_analysis

    # ===== Display =====
    st.subheader("📜 Legal Summary / Analysis")
    st.write(combined)

    if lang == "ar" and show_arabic:
        with st.spinner("Translating summary back to Arabic..."):
            ar = translate_text(combined, to_language="ar")
        st.subheader("📄 الملخص بالعربية")
        st.markdown(f"<div dir='rtl' style='line-height:1.7;'>{ar}</div>", unsafe_allow_html=True)

    st.subheader("✅ Suggested Next Steps")
    st.write(
        "- Verify party names, signatures, and effective dates.\n"
        "- Check governing law/jurisdiction clauses align with your needs.\n"
        "- Review termination, liability limits, indemnities, and IP ownership.\n"
        "- Ensure all referenced schedules/annexes are attached and consistent.\n"
        "- Have a licensed lawyer in the applicable jurisdiction review the final draft."
    )

    # Diagnostics (optional): uncomment when debugging
    # st.caption(f"HF_TOKEN present: {bool(HF_TOKEN)}  |  Azure Translator: {bool(translator_client)}  |  OpenAI: {bool(openai_client)}")
