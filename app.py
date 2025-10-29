import streamlit as st
import os
from docx import Document
from pypdf import PdfReader
from langdetect import detect
import requests
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
import io


# ------------------------------------------------------------
# 🔑 CONFIGURATION
# ------------------------------------------------------------

# Azure Translation
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://salahalali.cognitiveservices.azure.com/")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "qatarcentral")

translator_client = TextTranslationClient(
    endpoint=AZURE_TRANSLATOR_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
)

# HuggingFace
HF_MODEL_ID = "aun09/flan-t5-legal-summary"
HF_TOKEN = os.getenv("HF_TOKEN")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# PAGE MANAGEMENT STATE
# ----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = "analyze"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hf_result" not in st.session_state:
    st.session_state.hf_result = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "original_language" not in st.session_state:
    st.session_state.original_language = "en"

# ------------------------------------------------------------
# 🛠 HELPER FUNCTIONS
# ------------------------------------------------------------

def get_chatgpt_response(text, country="", model="gpt-4o-mini", hf_analysis=None):
    """Get analysis from ChatGPT with optional HuggingFace analysis integration."""
    try:
        if country:
                prompt = f"""Based on the previous legal analysis and the contract text, and considering the laws of {country},
                provide a comprehensive analysis incorporating the previous insights and add your analysis on:
                1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
                2 - Give me details on the missing information from the document if there is any,
                3 - Analysis of potential risks, such as non-standard clauses,
                4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
                5 - Summarized overview of extracted information, missing items, and potential risks.
                
                Previous Legal Analysis:
                {hf_analysis}
                
                Contract Text:
                {text}"""
        else:
                prompt = f"""Based on the previous legal analysis and the contract text, 
                provide a comprehensive analysis incorporating the previous insights and add your analysis on:
                1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
                2 - Give me details on the missing information from the document if there is any,
                3 - Analysis of potential risks, such as non-standard clauses,
                4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
                5 - Summarized overview of extracted information, missing items, and potential risks.
                
                Previous Legal Analysis:
                {hf_analysis}
                
                Contract Text:
                {text}"""

        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Error: Could not get response from ChatGPT"

def translate_text(text, to_language="en"):
    """Translate text to target language using Azure Translator."""
    try:
        response = translator_client.translate(
            body=[{"text": text}],
            to_language=[to_language]
        )
        return response[0].translations[0].text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def extract_text_from_file(file):
    """Extract text from DOCX, PDF, or TXT file."""
    if file.name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file.read()))
        return "\n".join([page.extract_text() or "" for page in pdf.pages]), len(pdf.pages)
    elif file.name.endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs]), len(doc.paragraphs) // 20 + 1
    else:
        text = file.read().decode("utf-8", errors="ignore")
        return text, len(text.split('\n'))

def query_huggingface(model_id, token, text, country=""):
    """Send text to the Hugging Face inference API."""
    headers = {"Authorization": f"Bearer {token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    
    if country:
        prompt = f"""Based on the following text that is taken from a contract document, and based on the laws of {country},
        analyze it for:
        1 - Key contract information and parties
        2 - Missing critical information
        3 - Potential legal risks and non-standard clauses
        4 - Recommendations for improvements
        5 - Overall legal assessment
        Text from legal document: {text}"""
    else:
        prompt = f"""Based on the following text that is taken from a contract document,
        analyze it for:
        0- which parties does this conract fevor more "answer only by it name at first line"?
        1 - Key contract information and parties
        2 - Missing critical information
        3 - Potential legal risks and non-standard clauses
        4 - Recommendations for improvements
        5 - Overall legal assessment
        Text from legal document: {text}"""
    
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1024}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        return response
    except Exception as e:
        return None

# ------------------------------------------------------------
# 🎨 UI CONFIGURATION
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Naja7 Contract Analyzer", page_icon="📄")

    # === PAGE 1: UPLOAD + ANALYSIS ===
    if st.session_state.mode == "analyze":
        st.title("📄 Naja7 Contract Analyzer")
        st.caption("AI-powered contract insights by Naja7")

        uploaded_file = st.file_uploader("Upload your contract", type=["pdf", "docx", "txt"])
        country = st.text_input("Country (optional)")
        analyze_btn = st.button("🔍 Analyze")

        if uploaded_file and analyze_btn:
            with st.spinner("Analyzing your contract with Naja7 AI..."):
                # Replace this mock text with your HuggingFace + GPT integration
                st.session_state.hf_result = "🧾 [HuggingFace] This contract defines obligations and penalties..."
                st.session_state.analysis_result = (
                    "🧠 [Naja7 AI] The contract seems valid but missing a confidentiality clause."
                )

            st.success("✅ Naja7 has completed the analysis!")

            st.subheader("📘 HuggingFace Summary")
            st.write(st.session_state.hf_result)

            st.subheader("🧠 Naja7 AI Legal Insights")
            st.write(st.session_state.analysis_result)

            st.markdown("---")
            if st.button("💬 Ask Naja7 About This Contract"):
                st.session_state.mode = "chat"
                st.rerun()

    # === PAGE 2: Q&A CHAT ===
    elif st.session_state.mode == "chat":
        st.title("💬 Naja7 Legal Q&A")
        st.caption("Ask questions and get AI-powered legal insights from Naja7")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_input = st.chat_input("Ask Naja7 about your contract...")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Replace this with your GPT API logic
            response = (
                f"🤖 [Naja7 AI] This clause refers to payment terms. "
                f"It aligns with standard contract practices."
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        st.markdown("---")
        if st.button("⬅ Back to Naja7 Contract Analysis"):
            st.session_state.mode = "analyze"
            st.rerun()


if __name__ == "__main__":
    main()
