import os
import sys
import uuid
import tempfile
import streamlit as st
import shutil
from typing import List, Tuple, Optional
from datetime import datetime
import logging

# Ensure project root is on sys.path for `src` imports when running via Streamlit
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import agent functions from our pipeline
try:
    from src.query import (
        ask as default_ask,
        rewrite_query,
        reason_over_context,
        finalize_answer,
    )
except ImportError as e:
    st.error(f"Failed to import query functions: {e}")
    st.stop()

# Additional LangChain imports for on-the-fly ingestion
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    from langchain_core.documents import Document
except ImportError as e:
    st.error(f"Failed to import LangChain dependencies: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent RAG System",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Custom CSS: red / silver / black theme
st.markdown("""
<style>
    :root {
        --silver: #c0c0c0;
        --silver-2: #e6e6e6;
        --black: #0a0a0a;
        --black-2: #121212;
        --red: #c62828;
        --red-dark: #8e0000;
        --muted: #9e9e9e;
    }

    /* Global typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: Tahoma, sans-serif !important;
        font-size: 1.5em !important; /* ~1.5x larger */
    }

    /* App backgrounds */
    [data-testid="stAppViewContainer"] {
        background: var(--black-2) !important;
        color: var(--silver-2) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--black), var(--black-2)) !important;
        color: var(--silver-2) !important;
        border-right: 1px solid rgba(192,192,192,0.12);
    }

    /* Sidebar font sizes (increased by ~100%) */
    [data-testid="stSidebar"] * { font-size: 1rem !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 { font-size: 1.8rem !important; }
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code { font-size: 1.4em !important; }

    /* Heading anchors (hide link icons next to headings) */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }

    /* Typography colors */
    h1, h2, h3, h4, h5, h6 { color: var(--silver-2) !important; }
    p, label, span, div { color: inherit; }
    a { color: var(--silver); text-decoration: underline; }

    /* Header banner */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, var(--red) 0%, var(--red-dark) 100%);
        color: var(--silver-2);
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid rgba(192,192,192,0.15);
    }

    /* Cards and containers */
    .metric-container {
        background: rgba(192,192,192,0.06);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--red);
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border: 1px solid rgba(192,192,192,0.15);
        background: rgba(10,10,10,0.35);
        color: var(--silver-2);
    }
    .user-message { border-left: 4px solid var(--silver); }
    .assistant-message { border-left: 4px solid var(--red); box-shadow: 0 2px 4px rgba(0,0,0,0.25); }

    .source-item {
        background: rgba(192,192,192,0.06);
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid var(--red);
        color: var(--silver-2);
    }

    /* Inputs */
    textarea, input, .stTextInput > div > div > input {
        background: rgba(18,18,18,0.65) !important;
        color: var(--silver-2) !important;
        border: 1px solid rgba(192,192,192,0.18) !important;
        border-radius: 8px !important;
    }
    textarea:focus, input:focus {
        outline-color: var(--red) !important;
        border-color: var(--red) !important;
        box-shadow: 0 0 0 1px var(--red) !important;
    }

    /* Buttons */
    div.stButton > button, button[kind="primary"] {
        background: var(--red) !important;
        color: var(--silver-2) !important;
        border: 0 !important;
        border-radius: 6px !important;
    }
    div.stButton > button:hover, button[kind="primary"]:hover {
        background: var(--red-dark) !important;
        color: var(--silver-2) !important;
    }

    /* Code blocks and divider */
    pre, code, .stCodeBlock { background: rgba(10,10,10,0.45) !important; color: var(--silver-2) !important; }
    hr, .stDivider { border-color: rgba(192,192,192,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "custom_chroma_dir": None,
        "custom_vectordb_ready": False,
        "custom_vectordb": None,
        "custom_embedding": None,
        "history": [],
        "processing": False,
        "current_file_name": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Utility helpers
def cleanup_temp_directories():
    """Clean up old temporary directories"""
    try:
        sessions_dir = os.path.join(PROJECT_ROOT, "tmp_sessions")
        if os.path.exists(sessions_dir):
            import time
            current_time = time.time()
            for dirname in os.listdir(sessions_dir):
                dir_path = os.path.join(sessions_dir, dirname)
                if os.path.isdir(dir_path):
                    dir_age = current_time - os.path.getctime(dir_path)
                    if dir_age > 3600:
                        shutil.rmtree(dir_path, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp directories: {e}")

# Build session-scoped vectordb
def build_session_vectordb(file_bytes: bytes, filename: str) -> Tuple[str, Chroma, OllamaEmbeddings]:
    try:
        cleanup_temp_directories()
        tmp_dir = os.path.join(PROJECT_ROOT, "tmp_sessions", str(uuid.uuid4()))
        os.makedirs(tmp_dir, exist_ok=True)

        tmp_pdf_path = os.path.join(tmp_dir, filename)
        with open(tmp_pdf_path, "wb") as f:
            f.write(file_bytes)

        with st.spinner("Loading PDF content..."):
            loader = PyPDFLoader(tmp_pdf_path)
            docs = loader.load()
        if not docs:
            raise ValueError("No content could be extracted from the PDF")

        with st.spinner("Splitting document into chunks..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks: List[Document] = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No chunks could be created from the document")

        with st.spinner("Creating embeddings and building vector index..."):
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            embedding = OllamaEmbeddings(model=embed_model)
            vectordb = Chroma.from_documents(
                chunks,
                embedding,
                persist_directory=tmp_dir
            )

        logger.info(f"Successfully indexed {len(chunks)} chunks from {filename}")
        return tmp_dir, vectordb, embedding
    except Exception as e:
        logger.error(f"Error building vector database: {e}")
        if 'tmp_dir' in locals() and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

# Enhanced ask
def ask_with_optional_custom_db(query_text: str) -> Tuple[str, List[str]]:
    try:
        if not st.session_state.custom_vectordb_ready:
            return default_ask(query_text)
        vectordb = st.session_state.custom_vectordb
        if vectordb is None:
            raise ValueError("Vector database is not properly initialized")
        with st.spinner("Rewriting query for better retrieval..."):
            rewritten_query = rewrite_query(query_text)
        with st.spinner("Retrieving relevant documents..."):
            top_k = int(os.getenv("RAG_TOP_K", "4"))
            docs = vectordb.similarity_search(rewritten_query, k=top_k)
        if not docs:
            return "I couldn't find any relevant information in the uploaded document to answer your question.", []
        with st.spinner("Analyzing context and reasoning..."):
            draft = reason_over_context(query_text, docs)
        with st.spinner("Finalizing answer..."):
            final_answer = finalize_answer(query_text, draft)
        seen = set()
        sources = []
        for d in docs:
            src = d.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return final_answer, sources
    except Exception as e:
        logger.error(f"Error in ask_with_optional_custom_db: {e}")
        return f"An error occurred while processing your question: {str(e)}", []

# --- Main UI Layout ---

# Header
st.title("Multi-Agent RAG Demo")
st.caption("Retrieval (rewrite) • Reasoning • Finalization — powered by Ollama models")

st.markdown("---")
# --- Sidebar config ---
with st.sidebar:

    st.subheader("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a PDF document to create a custom knowledge base"
    )

    process_clicked = st.button(
        "🔄 Process PDF",
        use_container_width=True,
        disabled=uploaded_file is None or st.session_state.processing
    )

    st.markdown("---")

    st.header("⚙️ Pipeline Setup")
    st.code(
        """
EMBED_MODEL=nomic-embed-text
RETRIEVER_MODEL=llama3
REASONER_MODEL=llama3
FINALIZER_MODEL=llama3
TOP_K=4
        """.strip(),
        language="bash",
    )

    st.markdown("---")

# --- Query input ---
with st.form("qa_form", clear_on_submit=False):
    user_q = st.text_area(
        "Ask a question",
        placeholder="What does the document say about...?",
        height=100,
        help="Ask detailed questions about your uploaded PDF document or general knowledge"
    )

    col_submit, col_clear = st.columns([1, 1])
    with col_submit:
        submitted = st.form_submit_button("🔍 Ask", use_container_width=True)
    with col_clear:
        if st.form_submit_button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# Handle PDF processing
if process_clicked and uploaded_file is not None:
    st.session_state.processing = True
    try:
        with st.spinner("Processing your PDF document..."):
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            if len(file_bytes) > 50 * 1024 * 1024:
                st.error("File too large. Please upload a PDF smaller than 50MB.")
                st.session_state.processing = False
                st.stop()
            chroma_dir, vectordb, embedding = build_session_vectordb(file_bytes, filename)
        st.session_state.custom_chroma_dir = chroma_dir
        st.session_state.custom_vectordb = vectordb
        st.session_state.custom_embedding = embedding
        st.session_state.custom_vectordb_ready = True
        st.session_state.current_file_name = filename
        st.session_state.processing = False
        st.success(f"✅ Successfully processed '{filename}'! You can now ask questions about this document.")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        st.session_state.processing = False

# Handle question submission
if submitted and user_q.strip():
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("🤔 Processing your question..."):
                answer, sources = ask_with_optional_custom_db(user_q.strip())
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.history.insert(0, {
                'question': user_q.strip(),
                'answer': answer,
                'sources': sources,
                'timestamp': timestamp
            })
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Display current answer
if st.session_state.history:
    current = st.session_state.history[0]
    st.subheader("💡 Answer")
    st.markdown(current['answer'])
    if current['sources']:
        st.subheader("📚 Sources")
        for source in current['sources']:
            st.markdown(f"""
            <div class="source-item">
                📄 {os.path.basename(source)}
            </div>
            """, unsafe_allow_html=True)

# Display conversation history
if len(st.session_state.history) > 1:
    with st.expander(f"📜 Previous Questions ({len(st.session_state.history) - 1})"):
        for i, item in enumerate(st.session_state.history[1:], 1):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            if item['sources']:
                st.markdown("**Sources:**")
                for source in item['sources']:
                    st.markdown(f"- {os.path.basename(source)}")
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #c0c0c0; font-size: 0.8em;">
    <p>Multi-Agent RAG System • Powered by Ollama and LangChain</p>
</div>
""", unsafe_allow_html=True)