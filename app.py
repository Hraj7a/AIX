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
from translationItems import classify_text
from apifirst import translate_text_with_model, translator_client

# ------------------------------------------------------------
# 🔑 CONFIGURATION
# ------------------------------------------------------------

# Azure Translation
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://salahalali.cognitiveservices.azure.com/")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "qatarcentral")

# HuggingFace
HF_MODEL_ID = "mrm8488/T5-base-finetuned-cuad"
HF_TOKEN = os.getenv("HF_TOKEN")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'original_language' not in st.session_state:
    st.session_state.original_language = None

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
    st.set_page_config(
        page_title="Contract Analysis",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styling
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

        .chat-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem 1rem;
        }
        
        .chat-box-user {
            background: #F5F5F5;
            border-radius: 24px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            max-width: 85%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            animation: slideInLeft 0.6s ease-out;
        }
        
        .chat-box-ai {
            background: #FFFFFF;
            border-radius: 24px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            margin-left: auto;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            max-width: 85%;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            animation: slideInRight 0.6s ease-out 0.3s both;
        }
        </style>
    """, unsafe_allow_html=True)

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

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader section
        uploaded_file = st.file_uploader(
            "Upload Contract Document",
            type=["txt", "pdf", "docx"],
            help="Supported formats: PDF (.pdf), Word (.docx), and text (.txt) files"
        )
        country = st.text_input("Specify country/region (Optional)", "")

        if uploaded_file:
            # Process uploaded file
            with st.spinner("Processing document..."):
                text, num_pages = extract_text_from_file(uploaded_file)
                
                if not text.strip():
                    st.error(f"The file {uploaded_file.name} appears to be empty or unreadable.")
                    return

                try:
                    lang = detect(text[:500])
                    st.session_state.original_language = lang
                    st.info(f"Detected language: {lang.upper()}")
                except:
                    lang = "unknown"
                    st.session_state.original_language = "en"
                    st.warning("Could not detect language, proceeding with analysis...")

                # Translate if Arabic
                if lang == "ar":
                    with st.spinner("Translating Arabic text to English..."):
                        try:
                            text = translate_text(text, to_language="en")
                        except Exception as e:
                            st.error(f"Translation error: {e}")
                            return

            # Display document content
            with st.expander("📄 Document Content", expanded=False):
                st.text_area("Content", text, height=200)

            # First, process with HuggingFace model
            with st.spinner("Analyzing with specialized legal model..."):
                try:
                    hf_response = query_huggingface(HF_MODEL_ID, HF_TOKEN, text, country)
                    if hf_response.status_code == 200:
                        hf_analysis = hf_response.json()[0]['generated_text']
                    else:
                        st.warning("HuggingFace model analysis failed, proceeding with GPT analysis only.")
                        hf_analysis = ""
                except Exception as e:
                    st.warning(f"HuggingFace model error: {str(e)}")
                    hf_analysis = ""

            # Then analyze with GPT
            with st.spinner("Performing detailed analysis with GPT..."):
                if not st.session_state.analysis_result:
                    # Combine HF analysis with GPT input if available
                    if hf_analysis:
                        enhanced_prompt = f"Previous legal analysis:\n{hf_analysis}\n\nPlease provide additional insights and analysis for this contract."
                        response = get_chatgpt_response(text + "\n\n" + enhanced_prompt, country)
                    else:
                        response = get_chatgpt_response(text, country)
                    
                    st.session_state.analysis_result = response
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # Display analysis results
            # Analysis Results with Tabs
            st.markdown("### 📊 Analysis Results")
            
            tab1, tab2, tab3 = st.tabs(["📝 Analysis", "📊 Statistics", "🔍 Details"])
            
            with tab1:
                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.write(st.session_state.analysis_result)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                # Statistics section
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="margin: 0; color: white; font-size: 2.5rem;">{num_pages}</h1>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">📄 Pages</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    word_count = len(text.split())
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="margin: 0; color: white; font-size: 2.5rem;">{word_count:,}</h1>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">📝 Words</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    char_count = len(text)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 2rem; border-radius: 15px; color: white; text-align: center;
                                box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                        <h1 style="margin: 0; color: white; font-size: 2.5rem;">{char_count:,}</h1>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">🔤 Characters</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                st.text_area("Document Content", text, height=200)
                st.markdown('</div>', unsafe_allow_html=True)

            # Translate back to Arabic if needed
            if lang == "ar":
                with st.spinner("Translating analysis to Arabic..."):
                    try:
                        arabic_response = translate_text(st.session_state.analysis_result, to_language="ar")
                        st.markdown("### 🔄 Arabic Translation")
                        st.markdown('<div class="output-box">', unsafe_allow_html=True)
                        st.write(arabic_response)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Translation error: {e}")

    # Chat interface in the second column
    with col2:
        st.markdown("### 💬 Chat with AI")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask questions about the document..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get AI response
            if st.session_state.analysis_result:
                context = f"""Based on this analysis of the legal document:
                {st.session_state.analysis_result}
                
                Answer this question: {prompt}"""
                
                response = get_chatgpt_response(context)
                
                # Translate response if needed
                if st.session_state.original_language == "ar":
                    try:
                        response = translate_text(response, to_language="ar")
                    except Exception as e:
                        st.error(f"Translation error: {e}")
                
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
