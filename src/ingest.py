
import os
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Resolve absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_PDF = os.path.join(PROJECT_ROOT, "data", "LLR.pdf")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Configurable embed model
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# 1. Load document
loader = PyPDFLoader(DATA_PDF)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Store in Chroma DB with Ollama embeddings
embedding = OllamaEmbeddings(model=EMBED_MODEL)
# Note: With langchain_chroma, persistence is automatic when persist_directory is provided
vectordb = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_DIR)

print(f"✅ Data ingested and stored in Chroma DB ({CHROMA_DIR})")
