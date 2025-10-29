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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'mode' not in st.session_state:
    st.session_state.mode = 'analyze'  # can be 'analyze' or 'chat'
if 'original_language' not in st.session_state:
    st.session_state.original_language = None
if 'hf_result' not in st.session_state:
    st.session_state.hf_result = None

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
    st.set_page_config(page_title="Contract Analysis", page_icon="📄", layout="wide")

    # ----------------------------
    # PAGE 1: UPLOAD + ANALYSIS
    # ----------------------------
    if st.session_state.mode == "analyze":
        st.markdown("""
        <div style="text-align:center; padding:1rem 2rem; max-width:1200px; margin:auto;">
            <h1 style="font-size:3rem; font-weight:800;">
                Analyze your Contracts with <span style="color:#1E8C7E;">Naja7</span>
            </h1>
            <p style="font-size:1.1rem; color:#6B7280;">
                Upload your contract to extract, analyze, and understand its content using AI.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("📎 Upload Contract Document", type=["txt", "pdf", "docx"])
        country = st.text_input("🌍 Specify country/region (Optional)", "")
        analyze_button = st.button("🔍 Analyze")

        if uploaded_file and analyze_button:
            with st.spinner("Performing legal analysis..."):
                # 1️⃣ Extract text
                text, _ = extract_text_from_file(uploaded_file)
                if not text.strip():
                    st.error("The file appears to be empty or unreadable.")
                    return

                # 2️⃣ Detect & translate Arabic if needed
                try:
                    lang = detect(text[:500])
                    st.session_state.original_language = lang
                    st.info(f"Detected language: {lang.upper()}")
                    if lang == "ar":
                        text = translate_text(text, "en")
                except Exception as e:
                    st.warning("Language detection failed; proceeding in English.")
                    lang = "en"

                # 3️⃣ Hugging Face analysis
                hf_response = query_huggingface(HF_MODEL_ID, HF_TOKEN, text, country)
                hf_result = None
                if hf_response and hf_response.status_code == 200:
                    data = hf_response.json()
                    if isinstance(data, list) and "generated_text" in data[0]:
                        hf_result = data[0]["generated_text"]
                    elif isinstance(data, dict) and "generated_text" in data:
                        hf_result = data["generated_text"]
                else:
                    st.warning("⚠️ Hugging Face model did not return a valid response.")
                    hf_result = None

                st.session_state.hf_result = hf_result or "No Hugging Face result available."

                # 4️⃣ GPT enhancement
                with st.spinner("Enhancing analysis with GPT..."):
                    gpt_response = get_chatgpt_response(
                        text,
                        country,
                        hf_analysis=st.session_state.hf_result
                    )
                    st.session_state.analysis_result = gpt_response

                # 5️⃣ Show analysis results on same page
                st.success("✅ Analysis complete!")

                with st.expander("📘 Hugging Face Analysis", expanded=True):
                    st.write(st.session_state.hf_result)

                with st.expander("🧠 GPT Comprehensive Analysis", expanded=True):
                    st.write(st.session_state.analysis_result)

                # 6️⃣ Ask Question Button
                st.markdown("---")
                if st.button("💬 Ask a Question About This Contract"):
                    st.session_state.mode = "chat"
                    st.rerun()

    # ----------------------------
    # PAGE 2: CHAT / Q&A PAGE
    # ----------------------------
    elif st.session_state.mode == "chat":
        st.markdown("## 💬 Ask Questions About the Contract")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_question = st.text_input("Type your question...")
        col1, col2 = st.columns([1, 3])
        with col1:
            back_btn = st.button("⬅ Back to Analysis")
        with col2:
            send_btn = st.button("Send")

        if back_btn:
            st.session_state.mode = "analyze"
            st.rerun()

        if send_btn and user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            context = f"""Based on this analysis:
            {st.session_state.analysis_result}

            Question: {user_question}"""
            response = get_chatgpt_response(context, model="gpt-4o-mini")

            if st.session_state.original_language == "ar":
                response = translate_text(response, "ar")

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()



if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = ""
    if "hf_result" not in st.session_state:
        st.session_state.hf_result = None
    main()
