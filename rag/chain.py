import logging
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions strictly based on the provided document context.

Rules:
- Only use information from the context below to answer.
- If the answer is not in the context, say exactly: "I don't have enough information in the document to answer this."
- Keep your answer clear and concise.
- Mention the page number(s) where you found the information when possible.

Context:
{context}

Question: {question}

Answer:
""")


def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Set it in Streamlit Cloud → App settings → Secrets, "
            "or in a local .env file."
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=api_key,
        max_tokens=1024,
        streaming=True,   # ← enable streaming at the LLM level
    )


def format_docs(docs: list[Document]) -> str:
    formatted = []
    for doc in docs:
        page   = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        formatted.append(f"[Page {page} | {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever):
    """Build the full RAG pipeline (streaming-ready)."""
    llm = get_llm()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    logger.info("RAG chain ready (streaming / Groq / LLaMA 3.1)")
    return chain


def stream_answer(chain, retriever, question: str):
    """
    Generator that yields answer tokens one by one for streaming display,
    then yields a final dict with source documents.

    Usage in Streamlit:
        for chunk in stream_answer(chain, retriever, question):
            if isinstance(chunk, str):
                # token — write to st.write_stream
            else:
                # dict with "sources" key — display citations
    """
    # Fetch source docs first (fast — just a vector search)
    source_docs = retriever.invoke(question)
    sources = [
        {
            "page":    doc.metadata.get("page", "?"),
            "source":  doc.metadata.get("source", "unknown").split("/")[-1],
            "excerpt": doc.page_content[:300],
        }
        for doc in source_docs
    ]

    # Stream tokens from the chain
    try:
        for token in chain.stream(question):
            yield token          # plain string token
    except ValueError as e:
        yield f"Configuration error: {e}"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield "Something went wrong. Please check your Groq API key and try again."

    # After all tokens, yield sources as a sentinel dict
    yield {"sources": sources}


def ask_question(chain, retriever, question: str) -> dict:
    """Non-streaming fallback (used by evaluate.py)."""
    try:
        answer      = chain.invoke(question)
        source_docs = retriever.invoke(question)
        sources = [
            {
                "page":    doc.metadata.get("page", "?"),
                "source":  doc.metadata.get("source", "unknown").split("/")[-1],
                "excerpt": doc.page_content[:300],
            }
            for doc in source_docs
        ]
        return {"answer": answer, "sources": sources}
    except ValueError as e:
        return {"answer": f"Configuration error: {e}", "sources": []}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"answer": "Something went wrong. Please check your Groq API key.", "sources": []}
