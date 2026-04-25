import logging
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt that keeps the model strictly grounded in the document
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
    """
    Return Groq-hosted LLaMA 3.1 8B.
    Groq is free, extremely fast (200+ tokens/sec), and uses the same model
    as local Ollama — so quality is identical.
    Reads GROQ_API_KEY from environment (set in Streamlit Cloud secrets).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Set it in Streamlit Cloud → App settings → Secrets, "
            "or in a local .env file."
        )

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,        # low = more factual answers
        api_key=api_key,
        max_tokens=1024
    )


def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.
    Includes page number and filename so the model can cite them.
    """
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        formatted.append(f"[Page {page} | {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(retriever):
    """
    Build the full RAG pipeline using LangChain LCEL pipe syntax:
    question → retrieve chunks → format → prompt → Groq LLM → answer string
    """
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain ready (Groq / LLaMA 3.1)")
    return chain


def ask_question(chain, retriever, question: str) -> dict:
    """
    Run a question through the RAG chain.
    Returns both the generated answer and the source chunks used,
    so the UI can show which pages the answer came from.
    """
    try:
        answer = chain.invoke(question)

        source_docs = retriever.invoke(question)
        sources = []
        for doc in source_docs:
            sources.append({
                "page": doc.metadata.get("page", "?"),
                "source": doc.metadata.get("source", "unknown").split("/")[-1],
                "excerpt": doc.page_content[:300]
            })

        return {"answer": answer, "sources": sources}

    except ValueError as e:
        # Missing API key — show a clear message to the user
        logger.error(f"Configuration error: {e}")
        return {
            "answer": f"Configuration error: {e}",
            "sources": []
        }
    except Exception as e:
        logger.error(f"Error during chain invocation: {e}")
        return {
            "answer": "Something went wrong while generating the answer. Please check your Groq API key and try again.",
            "sources": []
        }


if __name__ == "__main__":
    # Quick local test — requires GROQ_API_KEY in environment
    from rag.loader import process_pdf
    from rag.embedder import create_vectorstore
    from rag.retriever import get_retriever

    print("Building test pipeline...")
    chunks = process_pdf("uploads/sample.pdf")
    vs = create_vectorstore(chunks)
    retriever = get_retriever(vs)
    chain = build_rag_chain(retriever)

    question = "What is the main topic of this document?"
    result = ask_question(chain, retriever, question)

    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    for s in result["sources"]:
        print(f"  Source → Page {s['page']} ({s['source']}): {s['excerpt'][:80]}...")