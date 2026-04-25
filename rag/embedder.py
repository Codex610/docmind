import logging
import os
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a free HuggingFace embedding model.
    'all-MiniLM-L6-v2' is small, fast, and works great for semantic search.
    It downloads automatically on first run (~90MB) and is cached after that.
    """
    logger.info("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_vectorstore(chunks: list[Document], persist_dir: str = "vectorstore") -> Chroma:
    """
    Embed document chunks and store them in a local ChromaDB database.
    The store is persisted to disk so it survives between Streamlit reruns.
    """
    logger.info(f"Creating vectorstore with {len(chunks)} chunks...")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    logger.info(f"Vectorstore saved to '{persist_dir}'")
    return vectorstore


def load_vectorstore(persist_dir: str = "vectorstore") -> Chroma | None:
    """
    Load an existing ChromaDB vectorstore from disk.
    Returns None if nothing has been indexed yet.
    """
    if not os.path.exists(persist_dir):
        logger.warning(f"No vectorstore found at '{persist_dir}'")
        return None

    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    logger.info(f"Loaded existing vectorstore from '{persist_dir}'")
    return vectorstore


def add_documents(vectorstore: Chroma, chunks: list[Document]):
    """
    Add new document chunks to an existing vectorstore.
    Checks source metadata to skip files that are already indexed.
    """
    existing_sources = set()
    try:
        existing = vectorstore.get()
        for meta in existing.get("metadatas", []):
            if meta and "source" in meta:
                existing_sources.add(meta["source"])
    except Exception:
        pass

    new_chunks = [c for c in chunks if c.metadata.get("source") not in existing_sources]

    if not new_chunks:
        logger.info("All documents already indexed — nothing new to add.")
        return

    vectorstore.add_documents(new_chunks)
    logger.info(f"Added {len(new_chunks)} new chunks to vectorstore.")