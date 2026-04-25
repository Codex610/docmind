import logging
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF file and return a list of Documents, one per page.
    Each document carries metadata: source filename, page number, total pages.
    """
    docs = []

    try:
        pdf = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF: {file_path} — {e}")
        return docs

    total_pages = len(pdf)
    logger.info(f"Loading PDF: {file_path} ({total_pages} pages)")

    for page_num in range(total_pages):
        page = pdf[page_num]
        text = page.get_text()

        if not text.strip():
            logger.warning(f"Page {page_num + 1} has no extractable text (possibly scanned).")
            continue

        doc = Document(
            page_content=text,
            metadata={
                "source": file_path,
                "page": page_num + 1,
                "total_pages": total_pages,
            }
        )
        docs.append(doc)

    pdf.close()
    logger.info(f"Loaded {len(docs)} pages with text out of {total_pages} total.")
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller overlapping chunks for better retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for doc in documents:
        split_docs = splitter.split_documents([doc])
        for i, chunk in enumerate(split_docs):
            chunk.metadata["chunk_index"] = i
            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages.")
    return chunks


def process_pdf(file_path: str) -> list[Document]:
    """
    Full pipeline: load PDF and return ready-to-embed chunks.
    """
    pages = load_pdf(file_path)
    if not pages:
        logger.error("No pages loaded. Check your PDF file.")
        return []

    chunks = chunk_documents(pages)
    return chunks


if __name__ == "__main__":
    # Quick test — replace with your own PDF path
    test_path = "uploads/sample.pdf"
    result = process_pdf(test_path)
    print(f"\nTotal chunks ready for embedding: {len(result)}")
    if result:
        print(f"Sample chunk:\n{result[0].page_content[:300]}")
        print(f"Metadata: {result[0].metadata}")