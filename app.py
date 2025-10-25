import streamlit as st
import os
from docx import Document
from pypdf import PdfReader
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

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
translator_client = TextTranslationClient(
    endpoint=AZURE_TRANSLATOR_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ------------------------------------------------------------

def get_chatgpt_response(text, country="", model="gpt-3.5-turbo"):
    """Get analysis from ChatGPT."""
    try:
        if country:
            prompt = f"""Based on the following text that is taken from a contract document, and based on the laws of {country},
            I want you to analyze it and produce the following:
            1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
            2 - Give me details on the missing information from the document if there is any,
            3 - Analysis of potential risks, such as non-standard clauses,
            4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
            5 - Summarized overview of extracted information, missing items, and potential risks.
            Text from legal document: {text}"""
        else:
            prompt = f"""Based on the following text that is taken from a contract document,
            I want you to analyze it and produce the following:
            1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
            2 - Give me details on the missing information from the document if there is any,
            3 - Analysis of potential risks, such as non-standard clauses,
            4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
            5 - Summarized overview of extracted information, missing items, and potential risks.
            Text from legal document: {text}"""

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
        print(f"Error getting ChatGPT response: {e}")
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
        return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode("utf-8", errors="ignore")

# ------------------------------------------------------------
# 🎨 UI CONFIGURATION
# ------------------------------------------------------------

def main():
    st.set_page_config(page_title="Contract Analysis", page_icon="📄", layout="wide")
    st.title("Contract Analysis with AI")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Contract Document", type=["txt", "pdf", "docx"])
        country = st.text_input("Country (Optional)", "")
        model = st.selectbox("Select AI Model", ["gpt-3.5-turbo", "gpt-4"])
        translate = st.checkbox("Enable Translation")
        target_lang = "en"
        if translate:
            target_lang = st.selectbox("Target Language", ["en", "ar"])

    # Main content area
    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
        
        # Translation if needed
        if translate and target_lang != "en":
            text = translate_text(text, target_lang)

        # Display document content
        with st.expander("Document Content", expanded=False):
            st.text_area("Content", text, height=200)

        # Initialize chat
        if len(st.session_state.messages) == 0:
            response = get_chatgpt_response(text, country, model)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Input for follow-up questions
        if prompt := st.chat_input("Ask follow-up questions about the contract"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Combine original text with follow-up question
            context = f"Original contract text: {text}\n\nQuestion: {prompt}"
            response = get_chatgpt_response(context, country, model)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please upload a contract document to begin analysis.")

if __name__ == "__main__":
    main()
import os
from docx import Document
from pypdf import PdfReader
from langdetect import detect
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
import io
import requests

# ------------------------------------------------------------
# 🔑 CONFIGURATION
# ------------------------------------------------------------

# Azure Translation
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://salahalali.cognitiveservices.azure.com/")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "qatarcentral")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
translator_client = TextTranslationClient(
    endpoint=AZURE_TRANSLATOR_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# ------------------------------------------------------------

def get_chatgpt_response(text, country="", model="gpt-3.5-turbo"):
    """Get analysis from ChatGPT."""
    try:
        if country:
            prompt = f"""Based on the following text that is taken from a contract document, and based on the laws of {country},
            I want you to analyze it and produce the following:
            1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
            2 - Give me details on the missing information from the document if there is any,
            3 - Analysis of potential risks, such as non-standard clauses,
            4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
            5 - Summarized overview of extracted information, missing items, and potential risks.
            Text from legal document: {text}"""
        else:
            prompt = f"""Based on the following text that is taken from a contract document,
            I want you to analyze it and produce the following:
            1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
            2 - Give me details on the missing information from the document if there is any,
            3 - Analysis of potential risks, such as non-standard clauses,
            4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
            5 - Summarized overview of extracted information, missing items, and potential risks.
            Text from legal document: {text}"""

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
        print(f"Error getting ChatGPT response: {e}")
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
        return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode("utf-8", errors="ignore")

# ------------------------------------------------------------
# 🎨 UI CONFIGURATION
# ------------------------------------------------------------

def main():
    st.set_page_config(page_title="Contract Analysis", page_icon="📄", layout="wide")
    st.title("Contract Analysis with AI")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Contract Document", type=["txt", "pdf", "docx"])
        country = st.text_input("Country (Optional)", "")
        model = st.selectbox("Select AI Model", ["gpt-3.5-turbo", "gpt-4"])
        translate = st.checkbox("Enable Translation")
        target_lang = "en"
        if translate:
            target_lang = st.selectbox("Target Language", ["en", "ar"])

    # Main content area
    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
        
        # Translation if needed
        if translate and target_lang != "en":
            text = translate_text(text, target_lang)

        # Display document content
        with st.expander("Document Content", expanded=False):
            st.text_area("Content", text, height=200)

        # Initialize chat
        if len(st.session_state.messages) == 0:
            response = get_chatgpt_response(text, country, model)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Input for follow-up questions
        if prompt := st.chat_input("Ask follow-up questions about the contract"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Combine original text with follow-up question
            context = f"Original contract text: {text}\n\nQuestion: {prompt}"
            response = get_chatgpt_response(context, country, model)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please upload a contract document to begin analysis.")

if __name__ == "__main__":
    main()

def get_chatgpt_response(text, country="", model="gpt-3.5-turbo"):
    """Get analysis from ChatGPT."""
    try:
        if country:
            prompt = f"""Based on the following text that is taken from a contract document, and based on the laws of {country},
            I want you to analyze it and produce the following:
            1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
            2 - Give me details on the missing information from the document if there is any,
            3 - Analysis of potential risks, such as non-standard clauses,
            4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
            5 - Summarized overview of extracted information, missing items, and potential risks.
            Text from legal document: {text}"""
        else:
            prompt = f"""Based on the following text that is taken from a contract document,
            I want you to analyze it and produce the following:
            1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
            2 - Give me details on the missing information from the document if there is any,
            3 - Analysis of potential risks, such as non-standard clauses,
            4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
            5 - Summarized overview of extracted information, missing items, and potential risks.
            Text from legal document: {text}"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting ChatGPT response: {e}")
        return "Error: Could not get response from ChatGPT"



def query_huggingface(model_id, token, text, country=""):
    """Send text to the Hugging Face inference API."""
    headers = {"Authorization": f"Bearer {token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    
    if country:
        prompt = f"""Based on the following text that is taken from a contract document, and based on the laws of {country},
        I want you to analyze it and produce the following:
        1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
        2 - Give me details on the missing information from the document if there is any,
        3 - Analysis of potential risks, such as non-standard clauses,
        4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
        5 - Summarized overview of extracted information, missing items, and potential risks.
        Text from legal document: {text}"""
    else:
        prompt = f"""Based on the following text that is taken from a contract document,
        I want you to analyze it and produce the following:
        1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
        2 - Give me details on the missing information from the document if there is any,
        3 - Analysis of potential risks, such as non-standard clauses,
        4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
        5 - Summarized overview of extracted information, missing items, and potential risks.
        Text from legal document: {text}"""
    
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1024}}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    return response

# ------------------------------------------------------------
# 🎨 UI CONFIGURATION
# ------------------------------------------------------------

def main():
    st.set_page_config(page_title="Contract Analysis", page_icon="📄", layout="wide")
    st.title("Contract Analysis with AI")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Contract Document", type=["txt", "pdf", "docx"])
        country = st.text_input("Country (Optional)", "")
        model = st.selectbox("Select AI Model", ["gpt-3.5-turbo", "gpt-4"])
        translate = st.checkbox("Enable Translation")
        target_lang = "en"
        if translate:
            target_lang = st.selectbox("Target Language", ["en", "ar"])

    # Main content area
    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
        
        # Translation if needed
        if translate and target_lang != "en":
            text = translate_text(text, target_lang)

        # Display document content
        with st.expander("Document Content", expanded=False):
            st.text_area("Content", text, height=200)

        # Initialize chat
        if len(st.session_state.messages) == 0:
            response = get_chatgpt_response(text, country, model)
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Input for follow-up questions
        if prompt := st.chat_input("Ask follow-up questions about the contract"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Combine original text with follow-up question
            context = f"Original contract text: {text}\n\nQuestion: {prompt}"
            response = get_chatgpt_response(context, country, model)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please upload a contract document to begin analysis.")

if __name__ == "__main__":
    main()

def query_huggingface(model_id, token, text, country=""):
    """Send text to the Hugging Face inference API."""
    headers = {"Authorization": f"Bearer {token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    
    if country:
        prompt = f"""Based on the following text that is taken from a contract document, and based on the laws of {country},
        I want you to analyze it and produce the following:
        1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
        2 - Give me details on the missing information from the document if there is any,
        3 - Analysis of potenial risks, such as non-standard clauses,
        4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
        5 - Summarized overview of extracted infromation, missing items, and potential risks.
        Text from legal document: {text}"""
    else:
        prompt = f"""Based on the following text that is taken from a contract document,
        I want you to analyze it and produce the following:
        1 - Extracted contract information (eg. contracting parties, effective dates, governing laws, financial terms, etc),
        2 - Give me details on the missing information from the document if there is any,
        3 - Analysis of potenial risks, such as non-standard clauses,
        4 - Give me legal advice on what to change in the document, opinions, or law comparisons,
        5 - Summarized overview of extracted infromation, missing items, and potential risks.
        Text from legal document: {text}"""
    
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1024}}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    return response

translator_client = TextTranslationClient(
    endpoint=AZURE_TRANSLATOR_ENDPOINT,
    credential=AzureKeyCredential(AZURE_TRANSLATOR_KEY)
)

def translate_text(text, to_language="en"):
    """Translate text to target language using Azure Translator."""
    response = translator_client.translate(
        body=[{"text": text}],
        to_language=[to_language]
    )
    return response[0].translations[0].text

def extract_text_from_file(file):
    """Extract text from DOCX, PDF, or TXT file."""
    if file.name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file.read()))
        return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode("utf-8", errors="ignore")

# ------------------------------------------------------------
# 🎨 UI STYLING
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS (copied from streamlit_demo.py)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background-color: #FAFAFA;
        padding-top: 0 !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #1E8C7E 0%, #2AB598 100%);
        color: white !important;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 16px rgba(30, 140, 126, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .output-box {
        background: rgba(30, 140, 126, 0.05);
        border: 2px solid #1E8C7E;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        min-height: 150px;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# 📄 MAIN APP
# ------------------------------------------------------------
# Hero section
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

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    # File uploader 
    uploaded_files = st.file_uploader(
        "Drop your PDF or Word files here or click to browse",
        type=["pdf", "docx", "doc"],
        help="Supported formats: PDF (.pdf) and Word (.docx)",
        accept_multiple_files=True
    )

    # Optional country/region input
    st.markdown("### Specify the country/region (optional)")
    country = st.text_input("Enter country/region for legal context", key="country_input")

    # Process files when uploaded
    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Processing: {file.name}")
            
            with st.spinner("Extracting and analyzing document..."):
                # Extract text
                text = extract_text_from_file(file)

                if not text.strip():
                    st.error(f"The file {file.name} appears to be empty or unreadable.")
                    continue

                # Detect language
                try:
                    lang = detect(text[:500])
                    st.info(f"Detected language: **{lang.upper()}**")
                except Exception:
                    lang = "unknown"
                    st.warning("Could not detect language, proceeding with analysis...")

                # Translate if Arabic
                if lang == "ar":
                    st.write("🌐 Translating Arabic contract to English...")
                    try:
                        text_en = translate_text(text, to_language="en")
                    except Exception as e:
                        st.error(f"Translation failed: {e}")
                        continue
