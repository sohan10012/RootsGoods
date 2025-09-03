import os
import re
from typing import List, Tuple
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
RETRIEVER_MODEL = os.getenv("OLLAMA_RETRIEVER_MODEL", "llama3")
REASONER_MODEL = os.getenv("OLLAMA_REASONER_MODEL", "llama3")
FINALIZER_MODEL = os.getenv("OLLAMA_FINAL_MODEL", "llama3")

TOP_K = int(os.getenv("RAG_TOP_K", "4"))
FAST_MODE = os.getenv("RAG_FAST", "0") in {"1", "true", "True", "yes", "on"}
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "0")) or None
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "0")) or None

ollama_common_kwargs = {"temperature": 0}
if NUM_CTX is not None:
    ollama_common_kwargs["num_ctx"] = NUM_CTX
if NUM_PREDICT is not None:
    ollama_common_kwargs["num_predict"] = NUM_PREDICT

embedding = OllamaEmbeddings(model=EMBED_MODEL)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

retrieval_agent = OllamaLLM(model=RETRIEVER_MODEL, **ollama_common_kwargs)
reasoning_agent = OllamaLLM(model=REASONER_MODEL, **ollama_common_kwargs)
final_agent = OllamaLLM(model=FINALIZER_MODEL, **ollama_common_kwargs)

def _clean_answer(text: str) -> str:
    """Remove boilerplate like 'Here's an improved draft:' and 'Note:' prefaces."""
    patterns = [
        r"^\s*here'?s an improved draft:?\s*\n?",
        r"^\s*note:\s*",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned.strip()

def rewrite_query(user_query: str) -> str:
    """Use the retrieval agent to optimize the query for vector search."""
    if FAST_MODE:
        return user_query
    prompt = (
        "You rewrite user questions into a single, concise search query that maximizes recall "
        "for vector similarity search over technical PDFs. Do not answer the question. "
        "Return only the rewritten query text.\n\n"
        f"User Question: {user_query}\n"
        "Rewritten Search Query:"
    )
    rewritten = retrieval_agent.invoke(prompt)
    if isinstance(rewritten, dict) and "text" in rewritten:
        return rewritten["text"].strip()
    return str(rewritten).strip()

def retrieve_documents(search_query: str, k: int = TOP_K) -> List[Document]:
    """Retrieve top-k relevant documents from Chroma."""
    return vectordb.similarity_search(search_query, k=k)

def reason_over_context(user_query: str, docs: List[Document]) -> str:
    """Use the reasoning agent to synthesize a grounded draft based on retrieved docs."""
    context_blocks = []
    for i, d in enumerate(docs, start=1):
        meta_src = d.metadata.get("source", "Unknown")
        context_blocks.append(f"[Chunk {i} | Source: {meta_src}]\n{d.page_content}")
    context_text = "\n\n".join(context_blocks)

    prompt = (
        "You are a careful, grounded reasoner. Using ONLY the provided context, "
        "produce a precise, structured draft that answers the user question. "
        "Cite the chunks inline like [Chunk 2] where relevant. If information is missing, "
        "state it clearly. Do not include prefaces like 'Here's an improved draft:' or 'Note:'.\n\n"
        f"User Question: {user_query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Draft (with inline [Chunk N] citations):"
    )

    draft = reasoning_agent.invoke(prompt)
    if isinstance(draft, dict) and "text" in draft:
        return _clean_answer(draft["text"])
    return _clean_answer(str(draft))

def finalize_answer(user_query: str, draft: str) -> str:
    """Use the final agent to polish the draft into a concise, helpful answer."""
    if FAST_MODE:
        return _clean_answer(draft)
    prompt = (
        "You are an expert technical writer. Improve the draft to be clear and direct, "
        "preserving any factual claims and inline citations like [Chunk N]. "
        "Answer succinctly, avoid speculation, and do not add prefaces like 'Here's an improved draft:' or 'Note:'.\n\n"
        f"User Question: {user_query}\n\n"
        f"Draft:\n{draft}\n\n"
        "Final Answer:"
    )

    final = final_agent.invoke(prompt)
    if isinstance(final, dict) and "text" in final:
        return _clean_answer(final["text"])
    return _clean_answer(str(final))

def unique_sources(docs: List[Document]) -> List[str]:
    """Collect unique source identifiers from documents in a stable order."""
    seen = set()
    ordered = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            ordered.append(src)
    return ordered

def ask(query: str) -> Tuple[str, List[str]]:
    rewritten_query = rewrite_query(query)
    docs = retrieve_documents(rewritten_query, k=TOP_K)
    draft = reason_over_context(query, docs)
    final_answer = finalize_answer(query, draft)
    sources = unique_sources(docs)
    return final_answer, sources


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer, sources = ask(question)
        print("\nQuestion:", question)
        print("\nAnswer:\n", answer)
        print("\nSources:")
        for src in sources:
            print(" -", src)
    else:
        while True:
            user_q = input("\nAsk a question (or type 'exit'): ")
            if user_q.lower() in ["exit", "quit"]:
                break
            answer, sources = ask(user_q)
            print("\nAnswer:\n", answer)
            print("\nSources:")
            for src in sources:
                print(" -", src)
