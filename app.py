# ============================================================
# 📘 Contract Analyzer App (Streamlined Flow)
# ============================================================

import streamlit as st
import io
import os
from docx import Document
from pypdf import PdfReader
from langdetect import detect
import requests
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

# ============================================================
# 🔐 Secrets / Keys
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

# ============================================================
# 🧰 Initialize Clients
# ============================================================
client = OpenAI(api_key=OPENAI_API_KEY)
translator_client = None
if AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT:
    try:
        translator_client = TextTranslationClient(
            endpoint=AZURE_TRANSLATOR_ENDPOINT,
            credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
        )
    except Exception as e:
        st.warning(f"Azure Translator init failed: {e}")

# ============================================================
# 🌐 Helper Functions
# ============================================================

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
    """Extract text from DOCX, PDF, or TXT file."""
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
    """Send text to Hugging Face model for analysis."""
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
# 🎨 Streamlit App
# ============================================================
def main():
    st.set_page_config(page_title="Contract Analysis", page_icon="📜", layout="wide")
    st.title("📘 Contract Analysis System")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload Contract", type=["txt", "pdf", "docx"])
        country = st.text_input("Specify country/region (optional)", "")

        if uploaded_file:
            with st.spinner("Reading document..."):
                text, num_pages = extract_text_from_file(uploaded_file)

            if not text.strip():
                st.error("The file appears empty or unreadable.")
                st.stop()

            try:
                lang = detect(text[:500])
                st.info(f"Detected language: {lang.upper()}")
            except Exception:
                lang = "unknown"
                st.warning("Language not detected, assuming English.")

            # Translate Arabic → English before analysis
            if lang == "ar":
                with st.spinner("Translating Arabic text to English..."):
                    text = translate_text(text, to_language="en")

            # ==============================
            # 🔹 Hugging Face Analysis
            # ==============================
            with st.spinner("Analyzing with Hugging Face (Llama)..."):
                hf_response = query_huggingface(HF_MODEL_ID, HF_TOKEN, text, country)

            if getattr(hf_response, "status_code", 0) == 200:
                j = hf_response.json()
                if isinstance(j, list) and len(j) > 0 and "generated_text" in j[0]:
                    hf_analysis = j[0]["generated_text"]
                    st.subheader("🔹 Hugging Face (Llama) Analysis")
                    st.markdown('<div class="output-box">', unsafe_allow_html=True)
                    st.write(hf_analysis)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Hugging Face returned no analysis text.")
                    hf_analysis = ""
            else:
                st.warning("Hugging Face analysis unavailable or failed.")
                hf_analysis = ""

            # Translate analysis back to Arabic if needed
            if lang == "ar" and hf_analysis:
                with st.spinner("Translating analysis to Arabic..."):
                    hf_ar = translate_text(hf_analysis, to_language="ar")
                    st.subheader("🔄 Arabic Translation (Llama Output)")
                    st.markdown('<div class="output-box">', unsafe_allow_html=True)
                    st.write(hf_ar)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Document stats
            st.info(f"📄 Pages: {num_pages} | 🔤 Characters: {len(text):,}")

    # ===================================
    # 💬 ChatGPT Chat Section
    # ===================================
    with col2:
        st.markdown("### 💬 Chat with AI")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input("Ask about the document or analysis..."):
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
# 🚀 Run App
# ============================================================
if __name__ == "__main__":
    main()
