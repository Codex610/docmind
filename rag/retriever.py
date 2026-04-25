import logging
from langchain_community.vectorstores import Chroma
from langchain.schema.retriever import BaseRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_retriever(vectorstore: Chroma, k: int = 4) -> BaseRetriever:
    """
    Return a retriever that fetches the top-k most relevant chunks
    for a given query using cosine similarity.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    logger.info(f"Retriever ready (top-{k} chunks per query)")
    return retriever