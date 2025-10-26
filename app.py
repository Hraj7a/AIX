# legal_doc_analyzer_fixed.py
import streamlit as st
import io
import json
import time
from docx import Document
from pypdf import PdfReader
from langdetect import detect
import requests
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

# ------------------------------------------------------------
# 🔐 Secrets (Streamlit / env)
# ------------------------------------------------------------
AZURE_TRANSLATOR_KEY = st.secrets.get("AZURE_TRANSLATOR_KEY", "")
AZURE_TRANSLATOR_ENDPOINT = st.secrets.get("AZURE_TRANSLATOR_ENDPOINT", "")
AZURE_TRANSLATOR_REGION = st.secrets.get("AZURE_TRANSLATOR_REGION", "qatarcentral")

# The primary model you want to use (contract-specialized). If unavailable, fallback is used.
HF_MODEL_ID = st.secrets.get("HF_MODEL_ID", "google/flan-t5-large")  
HF_FALLBACK_MODEL_ID = st.secrets.get("HF_FALLBACK_MODEL_ID", "google/flan-t5-base")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# Stop early if the key you absolutely need is missing
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

# -------------------------------
# 🧰 Initialize clients
# -------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

translator_client = None
if AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT:
    try:
        translator_client = TextTranslationClient(
            endpoint=AZURE_TRANSLATOR_ENDPOINT,
            credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
        )
    except Exception as e:
        st.warning(f"Azure Translator client init failed: {e}")

# -------------------------------
# 🛠️ Helper functions
# -------------------------------
def translate_text(text: str, to_language: str = "en"):
    """Translate text using Azure if configured (returns original text if translator not available)."""
    if not translator_client:
        return text
    try:
        resp = translator_client.translate(body=[{"text": text}], to_language=[to_language])
        return resp[0].translations[0].text
    except Exception as e:
        st.warning(f"Translation error (returning original): {e}")
        return text


def extract_text_from_file(file):
    """Extract text from DOCX, PDF, or TXT file. Returns (text, approx_pages)."""
    name = file.name.lower()
    raw = file.read()
    if name.endswith(".pdf"):
        try:
            pdf = PdfReader(io.BytesIO(raw))
            text = "\n".join([(page.extract_text() or "") for page in pdf.pages])
            return text, len(pdf.pages)
        except Exception as e:
            st.error(f"PDF read error: {e}")
            return "", 0
    elif name.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(raw))
            text = "\n".join([para.text for para in doc.paragraphs])
            approx_pages = max(1, len(doc.paragraphs) // 20 + (1 if len(doc.paragraphs) % 20 else 0))
            return text, approx_pages
        except Exception as e:
            st.error(f"DOCX read error: {e}")
            return "", 0
    else:
        try:
            text = raw.decode("utf-8", errors="ignore")
            return text, max(1, len(text.splitlines()) // 40 + 1)
        except Exception as e:
            st.error(f"TXT read error: {e}")
            return "", 0


def chunk_text(text: str, max_chars: int = 3500):
    """Yield consecutive text chunks not exceeding max_chars (prefer sentence boundary naive split)."""
    text = text.strip()
    if len(text) <= max_chars:
        yield text
        return
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # try to break at nearest newline or sentence end within the last 200 chars
        if end < n:
            tail = text[start:end]
            sep = None
            for sep_char in ("\n\n", "\n", ". ", "; ", "? ", "! "):
                idx = tail.rfind(sep_char)
                if idx > max(0, len(tail) - 200):
                    sep = start + idx + len(sep_char)
                    break
            if sep:
                end = sep
        yield text[start:end].strip()
        start = end


def hf_post_with_retry(api_url: str, headers: dict, payload: dict, max_attempts: int = 4):
    """Post to HF endpoint with retry/backoff for 503 (model loading) and transient errors."""
    backoff = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
        except requests.RequestException as e:
            # network-level error: retry
            st.warning(f"Hugging Face network error (attempt {attempt}): {e}")
            if attempt < max_attempts:
                time.sleep(backoff)
                backoff *= 2
                continue
            class Dummy:
                status_code = 500
                def json(self): return {"error": str(e)}
            return Dummy()

        # If model is loading, HF returns 503 with estimated_time; wait and retry
        if resp.status_code == 503:
            try:
                body = resp.json()
                wait = int(body.get("estimated_time", 8))
            except Exception:
                wait = 8
            st.info(f"Hugging Face model loading (attempt {attempt}). Waiting ~{wait}s before retry...")
            time.sleep(min(max(wait, 4), 30))
            continue

        # For 429 (rate limit) retry after a pause
        if resp.status_code == 429:
            st.warning(f"Hugging Face rate limit (429). Attempt {attempt} of {max_attempts}.")
            time.sleep(backoff)
            backoff *= 2
            continue

        # Other statuses: return immediately (caller will handle)
        return resp

    # exhausted attempts
    class DummyExhausted:
        status_code = 500
        def json(self): return {"error": "Max retries exhausted"}
    return DummyExhausted()


def query_huggingface(model_id: str, token: str, text: str, country: str = "", fallback_model_id: str = None):
    """
    Send text to Hugging Face inference API. Handles chunking for long text and aggregates results.
    Returns a tuple (success_flag, aggregated_text, last_response_for_debug).
    """
    if not token:
        return False, "", {"error": "Missing HF_TOKEN"}

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}"}

    # build instruction prompt (for FLAN / instruction-tuned models)
    short_text = text[:3500]  # we will chunk anyway
    if country:
        prompt_template = (
            f"You are a legal contract analyst specializing in {country} law.\n"
            f"Analyze the following contract and provide:\n"
            f"1. Key details (parties, effective dates, governing law, financial terms)\n"
            f"2. Missing or irregular clauses\n"
            f"3. Potential legal/financial risks\n"
            f"4. Compliance recommendations\n\n"
            f"Contract text:\n{{chunk}}"
        )
    else:
        prompt_template = (
            "You are a legal contract analyst.\n"
            "Analyze the following contract and provide:\n"
            "1. Key details (parties, effective dates, governing law, financial terms)\n"
            "2. Missing or irregular clauses\n"
            "3. Potential legal/financial risks\n"
            "4. Compliance recommendations\n\n"
            "Contract text:\n{chunk}"
        )

    aggregated_outputs = []
    last_resp_debug = None

    # chunk text to avoid single huge request
    for i, chunk in enumerate(chunk_text(text, max_chars=3500)):
        prompt = prompt_template.format(chunk=chunk)
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512}}

        resp = hf_post_with_retry(api_url, headers, payload, max_attempts=4)
        last_resp_debug = {"status_code": getattr(resp, "status_code", None), "text": getattr(resp, "text", None)}

        # check for auth / model-not-found immediately
        if getattr(resp, "status_code", 0) in (401, 403):
            # Auth error; no point retrying further
            return False, "", {"error": "Authentication failed (401/403). Check HF_TOKEN.", "status": resp.status_code, "body": getattr(resp, "text", None)}

        if getattr(resp, "status_code", 0) == 404:
            return False, "", {"error": f"Model {model_id} not found (404).", "status": 404, "body": getattr(resp, "text", None)}

        if getattr(resp, "status_code", 0) != 200:
            # For other errors we will try fallback (outside) or return failure
            return False, "", {"error": f"Hugging Face returned status {getattr(resp,'status_code', None)}", "body": getattr(resp, "text", None)}

        # parse JSON and extract generated_text(s)
        try:
            j = resp.json()
            # HF can return list with dicts or a dict. Try to extract meaningful text.
            chunk_out = ""
            if isinstance(j, list):
                # join any generated_text fields
                parts = []
                for item in j:
                    if isinstance(item, dict) and "generated_text" in item:
                        parts.append(item["generated_text"])
                    elif isinstance(item, str):
                        parts.append(item)
                chunk_out = "\n".join(parts).strip()
            elif isinstance(j, dict):
                if "generated_text" in j:
                    chunk_out = j["generated_text"]
                elif "error" in j:
                    chunk_out = f"[error] {j.get('error')}"
                else:
                    chunk_out = json.dumps(j)[:1000]
            else:
                chunk_out = str(j)[:2000]
        except Exception as e:
            return False, "", {"error": f"Failed to parse HF response: {e}", "raw_text": getattr(resp, "text", None)}

        aggregated_outputs.append(f"--- CHUNK {i+1} ---\n{chunk_out}")

    aggregated_text = "\n\n".join(aggregated_outputs).strip()
    return True, aggregated_text, last_resp_debug


# -------------------------------
# 🎨 UI / App (original preserved)
# -------------------------------
def main():
    st.set_page_config(
        page_title="Contract Analysis",
        page_icon="logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS (kept unchanged)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        #MainMenu, footer, header { visibility: hidden; }
        .main { background-color: #FAFAFA; padding-top: 0 !important; }
        .stButton>button {
            background: linear-gradient(90deg, #1E8C7E 0%, #2AB598 100%);
            color: white !important;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 16px rgba(30,140,126,0.3);
            transition: all 0.3s ease;
            font-size: 1rem;
        }
        .output-box {
            background: rgba(30,140,126,0.05);
            border: 2px solid #1E8C7E;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            min-height: 150px;
        }
        .chat-container { max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }
        .chat-box-user {
            background: #F5F5F5; border-radius: 24px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
            display: flex; align-items: flex-start; gap: 1rem; max-width: 85%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .chat-box-ai {
            background: #FFFFFF; border-radius: 24px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; margin-left: auto;
            display: flex; align-items: flex-start; gap: 1rem; max-width: 85%;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 2rem 2rem 2rem; max-width: 1200px; margin: 0 auto;">
        <h1 style="font-size: 4rem; font-weight: 800; line-height: 1.2; margin-bottom: 1rem; margin-top: 0;">
            <span style="color: #2D3748;">Analyze your Contracts</span><br>
            <span style="color: #A0AEC0;">with</span>
            <span style="color: #1E8C7E;"> Naja7</span>
        </h1>
        <p style="font-size: 1.3rem; color: #718096; max-width: 800px; margin: 0 auto 2rem auto; line-height: 1.6;">
            Upload your PDF and Word documents to extract, analyze, and understand contract content with AI-powered insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "original_language" not in st.session_state:
        st.session_state.original_language = None

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Contract Document",
            type=["txt", "pdf", "docx"],
            help="Supported formats: PDF (.pdf), Word (.docx), and text (.txt) files"
        )
        country = st.text_input("Specify country/region (Optional)", "")

        if uploaded_file:
            with st.spinner("Processing document."):
                text, num_pages = extract_text_from_file(uploaded_file)

                if not text or not text.strip():
                    st.error(f"The file {uploaded_file.name} appears to be empty or unreadable.")
                    st.stop()

                try:
                    lang = detect(text[:500])
                    st.session_state.original_language = lang
                    st.info(f"Detected language: {lang.upper()}")
                except Exception:
                    lang = "unknown"
                    st.session_state.original_language = "en"
                    st.warning("Could not detect language, proceeding with analysis.")

                # Translate Arabic -> English for analysis
                if lang == "ar":
                    with st.spinner("Translating Arabic text to English."):
                        text = translate_text(text, to_language="en")

            with st.expander("📄 Document Content", expanded=False):
                st.text_area("Content", text, height=200)

            # Tabs
            t1, t2, t3 = st.tabs(["📝 Analysis", "📊 Stats", "🧾 Content"])

            # Hugging Face (analysis) -- with fallback and robust handling
            with st.spinner("Analyzing document using Hugging Face (primary model)..."):
                ok, hf_analysis, debug = query_huggingface(HF_MODEL_ID, HF_TOKEN, text, country)

            # If primary failed and fallback is configured, try fallback
            if not ok and HF_FALLBACK_MODEL_ID:
                st.warning(f"Primary HF model failed: {debug.get('error', debug)} — trying fallback model {HF_FALLBACK_MODEL_ID}...")
                with st.spinner("Analyzing document using Hugging Face (fallback model)..."):
                    ok2, hf_analysis2, debug2 = query_huggingface(HF_FALLBACK_MODEL_ID, HF_TOKEN, text, country)
                if ok2:
                    ok = True
                    hf_analysis = hf_analysis2
                    debug = debug2
                else:
                    # both failed
                    st.warning("Both primary and fallback Hugging Face models failed. See details below.")
                    st.warning(f"Primary debug: {debug}")
                    st.warning(f"Fallback debug: {debug2}")

            # If we have no analysis text, show warning but continue with UI
            if not ok or not hf_analysis:
                st.warning("Hugging Face analysis unavailable or failed, continuing with UI unchanged.")
                st.session_state.analysis_result = "No analysis was produced by Hugging Face."
            else:
                # Store English HF output (we keep original language in session to translate later)
                st.session_state.analysis_result = hf_analysis

            # --- Tabs content ---
            with t1:
                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.write(st.session_state.analysis_result)
                st.markdown('</div>', unsafe_allow_html=True)

            with t2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="margin: 0; color: white; font-size: 2.5rem;">{num_pages}</h1>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">📄 Pages</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    word_count = len(text.split())
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="margin: 0; color: white; font-size: 2.5rem;">{word_count:,}</h1>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">📝 Words</p>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    char_count = len(text)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="margin: 0; color: white; font-size: 2.5rem;">{char_count:,}</h1>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">🔤 Characters</p>
                    </div>
                    """, unsafe_allow_html=True)

            with t3:
                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.text_area("Document Content", text, height=200)
                st.markdown('</div>', unsafe_allow_html=True)

            # If original doc was Arabic and we have an analysis, translate the analysis back
            if st.session_state.original_language == "ar" and st.session_state.analysis_result and st.session_state.analysis_result != "No analysis was produced by Hugging Face.":
                try:
                    with st.spinner("Translating analysis to Arabic."):
                        arabic_response = translate_text(st.session_state.analysis_result, to_language="ar")
                        st.markdown("### 🔄 Arabic Translation")
                        st.markdown('<div class="output-box">', unsafe_allow_html=True)
                        st.write(arabic_response)
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Failed to translate analysis back to Arabic: {e}")

    # Right column: Chat with AI (unchanged UI)
    with col2:
        st.markdown("### 💬 Chat with AI")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask questions about the document."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                with st.spinner("ChatGPT thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a helpful legal assistant."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        reply = response.choices[0].message.content
                        st.write(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"OpenAI chat error: {e}")

# -------------------------------
# 🚀 Run App
# -------------------------------
if __name__ == "__main__":
    main()
