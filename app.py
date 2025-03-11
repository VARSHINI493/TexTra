import streamlit as st
import sqlite3
import hashlib
import re
import pandas as pd
import fitz  # PyMuPDF for PDF text extraction
from collections import Counter
from langdetect import detect
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer

# Page configuration
st.set_page_config(
    page_title="TexTra",
    page_icon="📝",
    layout="wide"
)

# Database connection
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

# Password Hashing Function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication Functions
def signup(email, password):
    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    if cursor.fetchone():
        return False  # Email already exists
    cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hash_password(password)))
    conn.commit()
    return True

def login(email, password):
    cursor.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hash_password(password)))
    return cursor.fetchone() is not None

# Text Analysis Functions
def analyze_text(text):
    char_count = len(text)
    words = [word.lower() for word in text.split()]
    word_count = len(words)
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    word_freq = Counter(words)
    repeated_words = {word: count for word, count in word_freq.items() if count > 1}

    single_word_repeated = sum(1 for count in word_freq.values() if count == 2)
    double_word_repeated = sum(1 for count in word_freq.values() if count == 3)
    triple_word_repeated = sum(1 for count in word_freq.values() if count == 4)

    return {
        "Total Words": word_count,
        "Total Characters": char_count,
        "Total Sentences": sentence_count,
        "Single Word Repeated": single_word_repeated,
        "Double Word Repeated": double_word_repeated,
        "Triple Word Repeated": triple_word_repeated
    }

# Language Detection and Translation Functions
def detect_language(text):
    try:
        lang_code = detect(text)
        lang_map = {
            "en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu", 
            "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
            "ja": "Japanese", "ko": "Korean", "zh-cn": "Chinese", "ru": "Russian",
            "ar": "Arabic", "pt": "Portuguese"
        }
        return lang_map.get(lang_code, f"Other ({lang_code})")
    except:
        return "Could not detect"

def translate_text(text, target_lang):
    translator = Translator()
    languages = {
        "Tamil": "ta", "English": "en", "Hindi": "hi", "Telugu": "te", 
        "French": "fr", "Spanish": "es", "German": "de", "Italian": "it",
        "Japanese": "ja", "Korean": "ko", "Chinese": "zh-cn", "Russian": "ru",
        "Arabic": "ar", "Portuguese": "pt"
    }
    target_code = languages.get(target_lang, "en")
    translated = translator.translate(text, dest=target_code)
    return translated.text

# PDF Text Extraction Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from the uploaded PDF file."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

# Text Summarization Functions
def split_into_sentences(text):
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def fallback_summarize(text, num_sentences=5):
    """Fallback summarization method when sumy fails."""
    sentences = split_into_sentences(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Process sentences
    processed_sentences = []
    for s in sentences:
        s = re.sub(r'[^\w\s]', '', s.lower())
        processed_sentences.append(s)

    # Calculate word frequency
    word_freq = Counter()
    for sentence in processed_sentences:
        for word in sentence.split():
            word_freq[word] += 1

    # Score sentences
    sentence_scores = []
    for i, sentence in enumerate(processed_sentences):
        score = sum(word_freq[word] for word in sentence.split())
        if len(sentence.split()) > 0:
            score /= len(sentence.split())  # Normalize by sentence length
        sentence_scores.append((i, score))

    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])

    # Create summary
    summary = " ".join([sentences[i] for i, _ in top_sentences])
    return summary

def summarize_text(text, num_sentences=5):
    """Try to summarize with sumy, fall back to simpler method if it fails."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
    except Exception:
        return fallback_summarize(text, num_sentences)

# Topic Modeling Function
def extract_topics(text, num_topics=5):
    """Extract topics (keywords) from the text using frequency analysis."""
    # Preprocess the text: Lowercase and remove punctuation
    processed_text = re.sub(r'[^\w\s]', '', text.lower())
    words = processed_text.split()

    # Calculate word frequencies
    word_freq = Counter(words)

    # Common stop words to exclude
    stop_words = set([
        "the", "and", "is", "in", "of", "for", "a", "an", "to", "with", 
        "on", "at", "by", "from", "as", "are", "be", "this", "that", "it", 
        "they", "we", "you", "he", "she", "them", "his", "her", "its", "their", 
        "our", "your", "my", "mine", "yours", "ours", "theirs", "him", "hers", 
        "us", "me", "i", "have", "has", "had", "do", "does", "did", "can", 
        "could", "may", "might", "must", "will", "would", "should", "am", 
        "are", "was", "were", "been", "being", "get", "gets", "got", "getting"
    ])

    filtered_word_freq_dict = {word: freq for word, freq in word_freq.items() 
                               if word not in stop_words and len(word) > 2}
    filtered_word_freq = Counter(filtered_word_freq_dict)

    # Get top N topics
    top_topics = filtered_word_freq.most_common(num_topics)
    return [topic for topic, freq in top_topics]

# Navigation Functions
def go_home():
    """Navigate to the home/options page"""
    st.session_state.page = "options"
    st.rerun()

def go_back(page):
    """Navigate to the specified previous page"""
    st.session_state.page = page
    st.rerun()

# Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "translate_input" not in st.session_state:
    st.session_state.translate_input = ""
if "detected_language" not in st.session_state:
    st.session_state.detected_language = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "summary_output" not in st.session_state:
    st.session_state.summary_output = ""
if "extracted_topics" not in st.session_state:
    st.session_state.extracted_topics = []

# Page Navigation Function
def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# Custom CSS for 70% width and enhanced styling
st.markdown("""
<style>
    .main-container {
        max-width: 70%;
        margin: 0 auto;
        background-color: #f9f9f9;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        animation: fadeIn 1.5s;
    }
    .logo-text {
        font-size: 42px;
        font-weight: bold;
        background: linear-gradient(45deg, #1E88E5, #26A69A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    .welcome-message {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
        animation: slideIn 1s;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #26A69A;
        margin-bottom: 15px;
        border-left: 4px solid #26A69A;
        padding-left: 10px;
    }
    .option-card {
        background-color: #26A69A;
        border-radius: 10px;
        padding: 3.5px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border-top: 3px solid #3f51b5;
    }
    .option-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    .result-container {
        background-color: #26A69A;
        padding: 2px;
        border-radius: 10px;
        margin-top: 10px;
        border-left: 4px solid #8bc34a;
        animation: fadeIn 0.5s;
    }
    .success-message {
        color: #2e7d32;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        margin: 10px 0;
        background-color: #c8e6c9;
        border-radius: 5px;
        animation: pulse 1.5s;
    }
    .navigation-area {
        display: flex;
        justify-content: space-between;
        margin-top: 30px;
    }
    .nav-button {
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s;
    }
    .back-button {
        background-color: #78909C;
        color: white;
    }
    .back-button:hover {
        background-color: #546e7a;
        transform: translateX(-3px);
    }
    .home-button {
        background-color: #2196F3;
        color: white;
    }
    .home-button:hover {
        background-color: #1976d2;
        transform: translateX(3px);
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        padding: 10px 15px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
    }
    .greeting-text {
        font-size: 22px;
        color: #3f51b5;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    /* Reduce padding in Streamlit components */
    .css-18e3th9, .css-1d391kg {
        padding: 1rem 1rem !important;
    }
    /* Input fields styling */
    .stTextInput input, .stTextArea textarea {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    /* Results metrics */
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .metric-label {
        font-size: 16px;
        color: #555;
        text-align: center;
    }
    .icon-text {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    /* Black background for output text boxes */
    .black-output-box {
        background-color: #262626;
        color: #ddd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #262626;
        margin-top: 10px;
        margin-bottom: 10px;
        font-family: "Source Sans Pro", sans-serif;
        overflow-wrap: break-word;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Login & Signup Page
if st.session_state.page == "login":
    # Create a container with reduced width
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="welcome-message">✨ Your All-in-One Text Processing Solution ✨</div>', unsafe_allow_html=True)
        
        auth_option = st.radio("Choose an option:", ["Login", "Signup"])

        if auth_option == "Login":
            st.markdown('<div class="greeting-text">👋 Welcome back! Please sign in to continue</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="greeting-text">🌟 Join TexTra today! Create your account</div>', unsafe_allow_html=True)
            
        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Password", type="password")

        if auth_option == "Signup":
            if st.button("🔒 Create Account"):
                if signup(email, password):
                    st.markdown('<div class="success-message">✅ Account created successfully! Please login.</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Email already exists!")

        elif auth_option == "Login":
            if st.button("🔓 Sign In"):
                if login(email, password):
                    st.markdown('<div class="success-message">🎉 Login successful! Redirecting...</div>', unsafe_allow_html=True)
                    st.session_state.authenticated = True
                    set_page("options")
                else:
                    st.error("❌ Invalid Email or Password")
                    
        st.markdown('</div>', unsafe_allow_html=True)

# Options Page (Home)
elif st.session_state.authenticated and st.session_state.page == "options":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">👋 Hello! What would you like to do today?</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="fas fa-chart-bar"></i> 📊 Text Analysis</h3>', unsafe_allow_html=True)
            st.write("Analyze text properties: word count, characters, repeated phrases and more.")
            if st.button("✨ Open Text Analysis", key="text_analysis_btn"):
                set_page("text_analysis_input")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="fas fa-language"></i> 🌍 Text Translation</h3>', unsafe_allow_html=True)
            st.write("Translate content between multiple languages with automatic detection.")
            if st.button("✨ Open Translation", key="translation_btn"):
                set_page("translation_input")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="option-card">', unsafe_allow_html=True)
            st.markdown('<h3><i class="fas fa-file-alt"></i> 📄 PDF Processor</h3>', unsafe_allow_html=True)
            st.write("Extract text, generate summaries, and identify key topics from PDF documents.")
            if st.button("✨ Open PDF Processor", key="pdf_btn"):
                set_page("pdf_upload")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Text Analysis - Input Page
elif st.session_state.authenticated and st.session_state.page == "text_analysis_input":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📊 Text Analysis Tool</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">📝 Enter your text below for detailed analysis</div>', unsafe_allow_html=True)
        
        text_input = st.text_area("✏️ Enter your text here:", height=250)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clear Text"):
                st.session_state.text_input = ""
                st.rerun()
        
        with col2:
            if st.button("🔍 Analyze Text"):
                if not text_input.strip():
                    st.warning("⚠️ Please enter some text before analyzing.")
                else:
                    st.session_state.text_input = text_input
                    st.session_state.analysis_results = analyze_text(text_input)
                    set_page("text_analysis_results")
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_analysis"):
                go_back("options")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_analysis"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Text Analysis - Results Page
elif st.session_state.authenticated and st.session_state.page == "text_analysis_results":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">🎉 Here\'s your text analysis!</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            
            # Display results in a more visual way with metrics
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            col5, col6 = st.columns(2)
            
            with col1:
                st.metric("📝 Total Words", st.session_state.analysis_results["Total Words"])
            
            with col2:
                st.metric("🔤 Total Characters", st.session_state.analysis_results["Total Characters"])
                
            with col3:
                st.metric("📋 Total Sentences", st.session_state.analysis_results["Total Sentences"])
                
            with col4:
                st.metric("🔁 Single Word Repeated", st.session_state.analysis_results["Single Word Repeated"])
                
            with col5:
                st.metric("🔄 Double Word Repeated", st.session_state.analysis_results["Double Word Repeated"])
                
            with col6:
                st.metric("🔃 Triple Word Repeated", st.session_state.analysis_results["Triple Word Repeated"])
                
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("❌ No analysis results available. Please go back and try again.")
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_analysis_results"):
                go_back("text_analysis_input")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_analysis_results"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Translation - Input Page
elif st.session_state.authenticated and st.session_state.page == "translation_input":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🌍 Text Translation</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">✨ Break language barriers with our translation tool</div>', unsafe_allow_html=True)
        
        translate_input = st.text_area("✏️ Enter text to translate:", height=250)
        
        if st.button("🔍 Detect Language & Continue"):
            if not translate_input.strip():
                st.warning("⚠️ Please enter some text before continuing.")
            else:
                with st.spinner("🔄 Detecting language..."):
                    st.session_state.translate_input = translate_input
                    st.session_state.detected_language = detect_language(translate_input)
                    set_page("translation_select")
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_translation"):
                go_back("options")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_translation"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Translation - Language Selection Page
elif st.session_state.authenticated and st.session_state.page == "translation_select":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🌍 Text Translation</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="greeting-text">🔍 Detected Language: <span style="color:#ff6f00;font-weight:bold">{st.session_state.detected_language}</span></div>', unsafe_allow_html=True)
        
        st.text_area("Your text:", st.session_state.translate_input, height=150, disabled=True)
        
        languages = [
            "English", "Tamil", "Hindi", "Telugu", "French", "Spanish", 
            "German", "Italian", "Japanese", "Korean", "Chinese", "Russian",
            "Arabic", "Portuguese"
        ]
        target_language = st.selectbox("🌐 Select target language for translation:", languages)
        
        if st.button("🔄 Translate Now"):
            with st.spinner("🌍 Translating your text..."):
                translated = translate_text(st.session_state.translate_input, target_language)
                st.session_state.translated_text = translated
                set_page("translation_results")
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_translation_select"):
                go_back("translation_input")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_translation_select"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Translation - Results Page
elif st.session_state.authenticated and st.session_state.page == "translation_results":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🌍 Translation Results</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">🎉 Translation Complete! 🎉</div>', unsafe_allow_html=True)
        st.write("Here's your translated text:")
        # Changed to black background box for translated text
        st.markdown(f'<div class="black-output-box">{st.session_state.translated_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_translation_results"):
                go_back("translation_select")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_translation_results"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# PDF Summarizer - Upload Page
elif st.session_state.authenticated and st.session_state.page == "pdf_upload":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📄 PDF Processor</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">📑 Turn your PDF documents into actionable insights</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("📎 Upload a PDF document", type=["pdf"])
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_sentences = st.slider("📝 Number of sentences in summary", min_value=1, max_value=10, value=5)
            
            with col2:
                num_topics = st.slider("🏷️ Number of topics to extract", min_value=1, max_value=10, value=5)
# Process button
            if st.button("🔍 Process PDF"):
                with st.spinner("📄 Extracting text from PDF..."):
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    
                    if extracted_text:
                        st.session_state.pdf_text = extracted_text
                        
                        with st.spinner("✨ Generating summary..."):
                            st.session_state.summary_output = summarize_text(extracted_text, num_sentences)
                            
                        with st.spinner("🔍 Extracting topics..."):
                            st.session_state.extracted_topics = extract_topics(extracted_text, num_topics)
                            
                        set_page("pdf_results")
                    else:
                        st.error("❌ Could not extract text from the PDF. Please try another file.")
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_pdf"):
                go_back("options")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_pdf"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# PDF Processor - Results Page
elif st.session_state.authenticated and st.session_state.page == "pdf_results":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📄 PDF Processing Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">🎉 PDF Processing Complete!</div>', unsafe_allow_html=True)
        
        tabs = st.tabs(["📃 Extracted Text", "📊 Summary", "🏷️ Key Topics"])
        
        with tabs[0]:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("### 📃 Extracted Text")
            st.markdown(f'<div class="black-output-box" style="max-height: 400px; overflow-y: auto;">{st.session_state.pdf_text}</div>', unsafe_allow_html=True)
            
            word_count = len(st.session_state.pdf_text.split())
            char_count = len(st.session_state.pdf_text)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📝 Total Words", word_count)
            with col2:
                st.metric("🔤 Total Characters", char_count)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[1]:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Text Summary")
            st.markdown(f'<div class="black-output-box">{st.session_state.summary_output}</div>', unsafe_allow_html=True)
            
            # Summary stats
            summary_words = len(st.session_state.summary_output.split())
            original_words = len(st.session_state.pdf_text.split())
            
            if original_words > 0:
                reduction_percent = 100 - (summary_words / original_words * 100)
            else:
                reduction_percent = 0
                
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📏 Summary Length", f"{summary_words} words")
            with col2:
                st.metric("📉 Reduction", f"{reduction_percent:.1f}%")
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[2]:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("### 🏷️ Key Topics Identified")
            
            if st.session_state.extracted_topics:
                for i, topic in enumerate(st.session_state.extracted_topics, 1):
                    st.markdown(f"**{i}.** {topic}")
            else:
                st.info("No key topics were identified in this document.")
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export options
        st.markdown("### 💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Text"):
                text_data = st.session_state.pdf_text
                df = pd.DataFrame({"Text": [text_data]})
                st.download_button(
                    label="📥 Confirm Download",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="extracted_text.csv",
                    mime="text/csv"
                )
                
        with col2:
            if st.button("📥 Download Summary"):
                summary_data = st.session_state.summary_output
                df = pd.DataFrame({"Summary": [summary_data]})
                st.download_button(
                    label="📥 Confirm Download",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="text_summary.csv",
                    mime="text/csv"
                )
                
        with col3:
            if st.button("📥 Download Topics"):
                topics_data = ", ".join(st.session_state.extracted_topics)
                df = pd.DataFrame({"Topics": [topics_data]})
                st.download_button(
                    label="📥 Confirm Download",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="key_topics.csv",
                    mime="text/csv"
                )
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_pdf_results"):
                go_back("pdf_upload")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_pdf_results"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# User profile page
elif st.session_state.authenticated and st.session_state.page == "user_profile":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">👤 User Profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="greeting-text">Welcome to your profile page!</div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Usage Statistics")
        
        # Placeholder for usage statistics - in a real app, this would be pulled from a database
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Text Analyses", "5")
        with col2:
            st.metric("🌍 Translations", "3")
        with col3:
            st.metric("📄 PDFs Processed", "2")
            
        st.markdown("### 🔑 Account Settings")
        
        # Change password section
        with st.expander("🔒 Change Password"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("✅ Update Password"):
                if not current_password or not new_password or not confirm_password:
                    st.warning("⚠️ Please fill in all password fields")
                elif new_password != confirm_password:
                    st.error("❌ New passwords do not match")
                else:
                    # In a real app, you would validate the current password and update it in the database
                    st.success("✅ Password updated successfully!")
        
        # Account deletion section
        with st.expander("⚠️ Delete Account"):
            st.warning("⚠️ Account deletion is permanent. All your data will be lost.")
            confirm_delete = st.text_input("Type 'DELETE' to confirm account deletion:")
            
            if st.button("❌ Delete My Account"):
                if confirm_delete != "DELETE":
                    st.error("❌ Please type 'DELETE' to confirm")
                else:
                    # In a real app, you would delete the user from the database
                    st.success("✅ Account deleted successfully. Logging out...")
                    st.session_state.authenticated = False
                    set_page("login")
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_profile"):
                go_back("options")
                
        with col2:
            if st.button("🚪 Logout", key="logout_btn"):
                st.session_state.authenticated = False
                set_page("login")
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# About Page
elif st.session_state.authenticated and st.session_state.page == "about":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="logo-text">TexTra</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">ℹ️ About TexTra</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🌟 Welcome to TexTra - Your All-in-One Text Processing Solution!
        
        TexTra provides a suite of powerful tools to analyze, translate, and process text content from various sources.
        
        #### ✨ Key Features:
        
        - **📊 Text Analysis**: Get detailed insights about your text content including word count, character statistics, and repetition analysis.
        
        - **🌍 Translation Service**: Break language barriers with our multi-language translation tool with automatic language detection.
        
        - **📄 PDF Processing**: Extract text from PDF documents, generate concise summaries, and identify key topics automatically.
        
        #### 💡 How to Use:
        
        1. **Navigate to the Home page** to select your desired tool
        2. **Follow the intuitive interface** to input your content
        3. **Review the results** and export as needed
        
        #### 🔒 Privacy & Security:
        
        TexTra values your privacy. All text processing happens on our secure servers, and we don't store your content after processing.
        
        #### 📱 Contact & Support:
        
        For assistance or feedback, please contact support@textra.com or visit our help center at www.textra.com/support
        """)
        
        # Navigation buttons
        st.markdown('<div class="navigation-area">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("⬅️ Back", key="back_btn_about"):
                go_back("options")
                
        with col2:
            if st.button("🏠 Home", key="home_btn_about"):
                go_home()
        st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# Add a footer to the options page
if st.session_state.authenticated and st.session_state.page == "options":
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #f5f5f5;">
        <div style="display: flex; justify-content: center; gap: 20px;">
            <a href="#" onclick="script:handleNavigation('user_profile')" style="text-decoration: none; color: #1E88E5;">👤 Profile</a>
            <a href="#" onclick="script:handleNavigation('about')" style="text-decoration: none; color: #1E88E5;">ℹ️ About</a>
            <a href="#" onclick="script:handleLogout()" style="text-decoration: none; color: #1E88E5;">🚪 Logout</a>
        </div>
    </div>
    
    <script>
    function handleNavigation(page) {
        const elements = window.parent.document.querySelectorAll('button');
        for (const element of elements) {
            if (element.innerText.includes(page === 'user_profile' ? 'Profile' : 'About')) {
                element.click();
                break;
            }
        }
    }
    
    function handleLogout() {
        const elements = window.parent.document.querySelectorAll('button');
        for (const element of elements) {
            if (element.innerText.includes('Logout')) {
                element.click();
                break;
            }
        }
    }
    </script>
    """, unsafe_allow_html=True)

# JavaScript to handle footer navigation
if st.session_state.authenticated and st.session_state.page == "options":
    if st.button("👤 Profile", key="profile_btn", help="View your profile"):
        set_page("user_profile")
    
    if st.button("ℹ️ About", key="about_btn", help="About TexTra"):
        set_page("about")
    
    if st.button("🚪 Logout", key="logout_btn", help="Logout from your account"):
        st.session_state.authenticated = False
        set_page("login")            