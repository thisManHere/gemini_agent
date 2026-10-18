from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import get_embeddings

DOCS_DIR = Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "knowledge_base"

def load_documents() -> list[Document]:
    docs = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                                 metadata={"source": str(path)}))
    return docs

def get_vectorstore(rebuild: bool = False) -> Chroma:
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=DB_DIR,
    )
    if rebuild:
        store.delete_collection()
        store = Chroma(collection_name=COLLECTION,
                       embedding_function=get_embeddings(),
                       persist_directory=DB_DIR)
    if store._collection.count() == 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(load_documents())
        if chunks:
            store.add_documents(chunks)
            print(f"[rag] indexed {len(chunks)} chunks from {DOCS_DIR}/")
        else:
            print(f"[rag] no .md/.txt files found in {DOCS_DIR}/")
    return store

