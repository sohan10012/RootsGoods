# src/query.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# 1. Load Chroma DB
embedding = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# 2. Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 3. Load LLM (Ollama running locally)
llm = Ollama(model="llama2")  # Change to "mistral" if you pulled that model

# 4. Build RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def ask(query: str):
    result = qa.invoke({"query": query})
    print("\n📌 Question:", query)
    print("🤖 Answer:", result['result'])
    print("\n🔎 Sources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "Unknown"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask(question)
    else:
        # fallback interactive mode
        while True:
            query = input("\nAsk a question (or type 'exit'): ")
            if query.lower() in ["exit", "quit"]:
                break
            ask(query)
