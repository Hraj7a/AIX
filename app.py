# ============================================================
# 📘 Contract Analyzer App — Original UI Preserved
# ============================================================

import streamlit as st
import io
from docx import Document
from pypdf import PdfReader
from langdetect import detect
import requests
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

# ============================================================
# 🔐 Secrets / Configurations
# ============================================================
AZURE_TRANSLATOR_KEY = st.secrets.get("AZURE_TRANSLATOR_KEY", "")
AZURE_TRANSLATOR_ENDPOINT = st.secrets.get("AZURE_TRANSLATOR_ENDPOINT", "")
AZURE_TRANSLATOR_REGION = st.secrets.get("AZURE_TRANSLATOR_REGION", "qatarcentral")

HF_MODEL_ID = st.secrets.get("HF_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

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

# ============================================================
# 🌐 Helper Functions
# ============================================================

def translate_text(text: str, to_language: str = "en"):
    """Translate text using Azure Translator."""
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
    """Extract text from DOCX, PDF, or TXT."""
    if file.name.lower().endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file.read()))
        text = "\n".join([(page.extract_text() or "") for page in pdf.pages])
        return text, len(pdf.pages)
    elif file.name.lower().endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        text = "\n".join([p.text for p in doc.paragraphs])
        approx_pages = max(1, len(doc.paragraphs)//20 + (1 if len(doc.paragraphs)%20 else 0))
        return text, approx_pages
    else:
        raw = file.read().decode("utf-8", errors="ignore")
        return raw, max(1, len(raw.splitlines())//40 + 1)


def query_huggingface(model_id: str, token: str, text: str, country: str = ""):
    """Send text to Hugging Face for legal contract analysis."""
    if not token:
        class DummyResp:
            status_code = 401
            def json(self): return {"error": "Missing HF_TOKEN"}
        return DummyResp()

    headers = {"Authorization": f"Bearer {token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    short_text = text[:4000]
    if country:
        prompt = (
            f"You are a legal contract analyst specializing in {country} law.\n"
            f"Analyze the following contract and provide:\n"
            f"1. Key details (parties, effective dates, governing law, finances)\n"
            f"2. Missing or irregular clauses\n"
            f"3. Potential legal/financial risks\n"
            f"4. Compliance recommendations\n\n"
            f"Contract text:\n{short_text}"
        )
    else:
        prompt = (
            "You are a legal contract analyst.\n"
            "Analyze the following contract and provide:\n"
            "1. Key details (parties, effective dates, governing law, finances)\n"
            "2. Missing or irregular clauses\n"
            "3. Potential legal/financial risks\n"
            "4. Compliance recommendations\n\n"
            f"Contract text:\n{short_text}"
        )

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1024}}

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
        print("HF STATUS", resp.status_code, "BODY", resp.text[:400])
        return resp
    except Exception as e:
        class DummyErr:
            status_code = 500
            def json(self): return {"error": str(e)}
        return DummyErr()

# ============================================================
# 🎨 Main App (Original UI)
# ============================================================

def main():
    st.set_page_config(page_title="Contract Analysis", page_icon="logo.png", layout="wide")

    # --- Custom CSS ---
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
        </style>
    """, unsafe_allow_html=True)

    # --- Hero Header ---
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

    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "original_language" not in st.session_state:
        st.session_state.original_language = None

    # --- Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload Contract Document", type=["txt", "pdf", "docx"])
        country = st.text_input("Specify country/region (Optional)", "")

        if uploaded_file:
            with st.spinner("Processing document..."):
                text, num_pages = extract_text_from_file(uploaded_file)

            if not text.strip():
                st.error("The file appears empty or unreadable.")
                st.stop()

            try:
                lang = detect(text[:500])
                st.session_state.original_language = lang
                st.info(f"Detected language: {lang.upper()}")
            except Exception:
                lang = "unknown"
                st.session_state.original_language = "en"
                st.warning("Could not detect language, proceeding with analysis.")

            # Arabic → English
            if lang == "ar":
                with st.spinner("Translating Arabic text to English..."):
                    text = translate_text(text, to_language="en")

            with st.expander("📄 Document Content", expanded=False):
                st.text_area("Content", text, height=200)

            # --- Hugging Face Analysis (Llama) ---
            with st.spinner("Analyzing document using Llama model..."):
                hf_response = query_huggingface(HF_MODEL_ID, HF_TOKEN, text, country)

            if getattr(hf_response, "status_code", 0) == 200:
                j = hf_response.json()
                if isinstance(j, list) and len(j) > 0 and "generated_text" in j[0]:
                    hf_analysis = j[0]["generated_text"]
                    st.subheader("📝 Analysis (Hugging Face Llama)")
                    st.markdown('<div class="output-box">', unsafe_allow_html=True)
                    st.write(hf_analysis)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.session_state.analysis_result = hf_analysis
                else:
                    st.warning("No analysis text returned from model.")
                    hf_analysis = ""
            else:
                st.warning("Hugging Face analysis unavailable or failed.")
                hf_analysis = ""

            # Arabic translation back
            if lang == "ar" and hf_analysis:
                with st.spinner("Translating analysis to Arabic..."):
                    hf_analysis_ar = translate_text(hf_analysis, to_language="ar")
                    st.markdown("### 🔄 Arabic Translation (Llama Output)")
                    st.markdown('<div class="output-box">', unsafe_allow_html=True)
                    st.write(hf_analysis_ar)
                    st.markdown('</div>', unsafe_allow_html=True)

    # --- ChatGPT Interactive Section ---
    with col2:
        st.markdown("### 💬 Chat with AI")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask about the document..."):
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

# ============================================================
# 🚀 Run
# ============================================================
if __name__ == "__main__":
    main()
