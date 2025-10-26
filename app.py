# legal_doc_analyzer.py
import streamlit as st
import os
import requests
import io
from docx import Document
from PyPDF2 import PdfReader
from langdetect import detect
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
import openai
import functools
import time

# ------------------------------------------------------------
# ⚙️ CONFIGURATION
# ------------------------------------------------------------
HF_MODEL_ID = "mrm8488/T5-base-finetuned-cuad"
HF_TOKEN = os.getenv("HF_TOKEN")

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = "https://salahalali.cognitiveservices.azure.com/"
AZURE_TRANSLATOR_REGION = "qatarcentral"

api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=api_key)

# ------------------------------------------------------------
# ⚡ PERFORMANCE UTILITIES
# ------------------------------------------------------------
@functools.lru_cache(maxsize=10)
def translate_text_cached(text, to_language="en"):
    """Cached translation for repeated calls"""
    translator_client = TextTranslationClient(
        endpoint=AZURE_TRANSLATOR_ENDPOINT,
        credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
    )
    response = translator_client.translate(body=[{"text": text}], to_language=[to_language])
    return response[0].translations[0].text

@functools.lru_cache(maxsize=10)
def query_huggingface_cached(model_id, token, text):
    """Cached Hugging Face response for repeated runs"""
    headers = {"Authorization": f"Bearer {token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    payload = {"inputs": f"For the following legal text: {text}", "parameters": {"max_new_tokens": 256}}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    return response


def extract_text_from_file(file):
    """Extract text efficiently from DOCX, PDF, or TXT."""
    if file.name.endswith(".pdf"):
        try:
            pdf = PdfReader(io.BytesIO(file.read()))
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            return text.strip()
        except Exception:
            return ""
    elif file.name.endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    else:
        return file.read().decode("utf-8", errors="ignore").strip()


# ------------------------------------------------------------
# 🧠 STREAMLIT INTERFACE
# ------------------------------------------------------------
st.title("⚖️ Multilingual Legal Document Analyzer")
st.write("Upload a contract (Arabic or English). The app detects language, translates if needed, and summarizes key insights.")

user_token = st.text_input("Enter your Hugging Face API Token (optional)", type="password")
if user_token:
    HF_TOKEN = user_token

uploaded_file = st.file_uploader("📂 Upload your contract (DOCX, PDF, or TXT)", type=["docx", "pdf", "txt"])

if st.button("🔍 Analyze Document") and uploaded_file:
    start_time = time.time()
    with st.spinner("Extracting and analyzing document..."):
        text = extract_text_from_file(uploaded_file)

        if not text:
            st.error("The file appears to be empty or unreadable.")
            st.stop()

        # detect language faster by truncating text
        try:
            lang = detect(text[:400])
        except Exception:
            lang = "unknown"

        st.info(f"Detected language: **{lang.upper()}**")

        # Step 1: Translate only if Arabic
        if lang == "ar":
            st.write("🌐 Translating Arabic contract to English...")
            try:
                text_en = translate_text_cached(text[:5000], to_language="en")  # limit text
            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.stop()
        else:
            text_en = text[:5000]

        # Step 2: Query Hugging Face
        st.write("🤖 Analyzing legal content using CUAD model...")
        resp = query_huggingface_cached(HF_MODEL_ID, HF_TOKEN, text_en)

        if resp.status_code == 200:
            try:
                data = resp.json()
                summary = data[0]["generated_text"] if isinstance(data, list) else str(data)

                st.subheader("📜 Legal Summary / Analysis")
                st.write(summary)

                # Step 3: Translate back to Arabic if needed
                if lang == "ar":
                    st.write("🔁 Translating summary back to Arabic...")
                    summary_ar = translate_text_cached(summary, to_language="ar")
                    st.subheader("📄 Arabic Summary")
                    st.write(summary_ar)

                # Step 4: Suggestions
                st.subheader("💬 Suggested Next Steps")
                st.write(
                    "- Review key obligations and liabilities.\n"
                    "- Verify party names, dates, and jurisdictions.\n"
                    "- Check for missing clauses or ambiguous terms.\n"
                    "- Ensure compliance with local laws."
                )

            except Exception as e:
                st.error(f"Error parsing model output: {e}")
                st.text(resp.text)
        else:
            st.error(f"Request failed ({resp.status_code}): {resp.text}")

    st.success(f"✅ Done in {round(time.time() - start_time, 2)} seconds")
