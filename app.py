# ============================================================
# 📘 Contract Analyzer App — Original UI Preserved (FLAN-T5)
# ============================================================

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

# -------------------------------
# 🔐 Secrets (Streamlit Cloud)
# -------------------------------
AZURE_TRANSLATOR_KEY = st.secrets.get("AZURE_TRANSLATOR_KEY", "")
AZURE_TRANSLATOR_ENDPOINT = st.secrets.get("AZURE_TRANSLATOR_ENDPOINT", "")
AZURE_TRANSLATOR_REGION = st.secrets.get("AZURE_TRANSLATOR_REGION", "qatarcentral")

# ✅ Use google/flan-t5-large by default
HF_MODEL_ID = st.secrets.get("HF_MODEL_ID", "google/flan-t5-large")
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
    """Translate text using Azure if configured."""
    if not translator_client:
        return text
    try:
        resp = translator_client.translate(
            body=[{"text": text}],
            to_language=[to_language]
        )
        return resp[0].translations[0].text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text


def extract_text_from_file(file):
    """Extract text from DOCX, PDF, or TXT file. Returns (text, approx_pages)."""
    if file.name.lower().endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file.read()))
        text = "\n".join([(page.extract_text() or "") for page in pdf.pages])
        return text, len(pdf.pages)
    elif file.name.lower().endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
        approx_pages = max(1, len(doc.paragraphs) // 20 + (1 if len(doc.paragraphs) % 20 else 0))
        return text, approx_pages
    else:
        raw = file.read().decode("utf-8", errors="ignore")
        return raw, max(1, len(raw.splitlines()) // 40 + 1)


def query_huggingface(model_id: str, token: str, text: str, country: str = ""):
    """Send text to the Hugging Face inference API (instruction prompt for FLAN-T5)."""
    if not token:
        # If no token provided, return a dummy-like response
        class DummyResp:
            status_code = 401
            def json(self): return {"error": "Missing HF_TOKEN"}
        return DummyResp()

    headers = {"Authorization": f"Bearer {token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    # Instruction-style prompt for FLAN-T5 (trim input for reliability)
    short_text = text[:3500]
    if country:
        prompt = (
            f"You are a legal contract analyst specializing in {country} law.\n"
            f"Analyze the following contract and provide:\n"
            f"1. Key details (parties, effective dates, governing law, financial terms)\n"
            f"2. Missing or irregular clauses\n"
            f"3. Potential legal/financial risks\n"
            f"4. Compliance recommendations\n\n"
            f"Contract text:\n{short_text}"
        )
    else:
        prompt = (
            "You are a legal contract analyst.\n"
            "Analyze the following contract and provide:\n"
            "1. Key details (parties, effective dates, governing law, financial terms)\n"
            "2. Missing or irregular clauses\n"
            "3. Potential legal/financial risks\n"
            "4. Compliance recommendations\n\n"
            f"Contract text:\n{short_text}"
        )

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512}}

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
        print("HF STATUS", resp.status_code, "BODY", resp.text[:500])

        # Retry once if model is loading (503 with estimated_time)
        if resp.status_code == 503 and "estimated_time" in resp.text:
            try:
                wait = json.loads(resp.text).get("estimated_time", 10)
            except Exception:
                wait = 10
            time.sleep(min(max(int(wait), 5), 20))
            resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
            print("HF RETRY STATUS", resp.status_code, "BODY", resp.text[:500])

        return resp
    except Exception as e:
        class DummyErr:
            status_code = 500
            def json(self): return {"error": str(e)}
        return DummyErr()

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

    # Custom CSS (same style intent as your version)
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

            # Tabs (as in your original structure: Analysis / Stats / Content)
            t1, t2, t3 = st.tabs(["📝 Analysis", "📊 Stats", "🧾 Content"])

            # Hugging Face (pre-analysis → now the main analysis)
            with st.spinner("Analyzing document using Hugging Face (FLAN-T5)..."):
                hf_response = query_huggingface(HF_MODEL_ID, HF_TOKEN, text, country)

            hf_analysis = ""
            if getattr(hf_response, "status_code", 0) == 200:
                try:
                    j = hf_response.json()
                    # Accept both list and dict shapes
                    if isinstance(j, list) and j and isinstance(j[0], dict) and "generated_text" in j[0]:
                        hf_analysis = j[0]["generated_text"]
                    elif isinstance(j, dict) and "generated_text" in j:
                        hf_analysis = j["generated_text"]
                    elif isinstance(j, dict) and "error" in j:
                        print("HF ERROR BODY:", j.get("error"))
                except Exception as e:
                    print("HF PARSE ERROR:", e)

            if not hf_analysis:
                st.warning("Hugging Face analysis unavailable or failed, continuing with UI unchanged.")

            # Set analysis_result (for the Analysis tab)
            st.session_state.analysis_result = hf_analysis if hf_analysis else "No analysis was produced."

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

            # If original doc was Arabic, optionally translate analysis back to Arabic
            if st.session_state.original_language == "ar" and hf_analysis:
                with st.spinner("Translating analysis to Arabic."):
                    arabic_response = translate_text(st.session_state.analysis_result, to_language="ar")
                    st.markdown("### 🔄 Arabic Translation")
                    st.markdown('<div class="output-box">', unsafe_allow_html=True)
                    st.write(arabic_response)
                    st.markdown('</div>', unsafe_allow_html=True)

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

# -------------------------------
# 🚀 Run App
# -------------------------------
if __name__ == "__main__":
    main()
