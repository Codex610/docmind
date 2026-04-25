import os
import streamlit as st
from rag.loader import process_pdf
from rag.embedder import create_vectorstore, load_vectorstore, add_documents
from rag.retriever import get_retriever
from rag.chain import build_rag_chain, ask_question

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="wide"
)

st.title("📄 DocMind — Chat with your PDFs")
st.caption("Upload PDFs on the left, then ask anything about them.")

# --- Session state defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


# --- Check API key early ---
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error(
        "**GROQ_API_KEY not set.**\n\n"
        "- On Streamlit Cloud: go to App settings → Secrets → add `GROQ_API_KEY = your_key`\n"
        "- Locally: create a `.env` file with `GROQ_API_KEY=your_key` and run with `streamlit run app.py`\n\n"
        "Get a free key at [console.groq.com](https://console.groq.com)"
    )
    st.stop()


# --- Sidebar: PDF upload ---
with st.sidebar:
    st.header("📁 Your Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("⚡ Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            new_files = []
            for f in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.read())
                if f.name not in st.session_state.loaded_files:
                    new_files.append((f.name, save_path))

            if not new_files:
                st.info("All selected PDFs are already loaded.")
            else:
                all_chunks = []
                with st.status("Processing PDFs...", expanded=True) as status:
                    for name, path in new_files:
                        st.write(f"Parsing: {name}")
                        chunks = process_pdf(path)
                        if chunks:
                            all_chunks.extend(chunks)
                            st.session_state.loaded_files.append(name)
                        else:
                            st.warning(f"{name} had no readable text — skipped.")

                    if all_chunks:
                        st.write("Building vector index...")
                        existing_vs = load_vectorstore(VECTOR_DIR)
                        if existing_vs:
                            add_documents(existing_vs, all_chunks)
                            vectorstore = existing_vs
                        else:
                            vectorstore = create_vectorstore(all_chunks, VECTOR_DIR)

                        st.write("Connecting to Groq LLM...")
                        retriever = get_retriever(vectorstore, k=4)
                        chain = build_rag_chain(retriever)
                        st.session_state.retriever = retriever
                        st.session_state.chain = chain
                        status.update(label="Ready! Start chatting below.", state="complete")
                    else:
                        status.update(label="No usable text found.", state="error")

    # Show loaded files list
    if st.session_state.loaded_files:
        st.markdown("**Loaded files:**")
        for name in st.session_state.loaded_files:
            st.markdown(f"- {name}")

    st.divider()

    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.messages = []
        st.session_state.loaded_files = []
        st.session_state.chain = None
        st.session_state.retriever = None
        import shutil
        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        st.rerun()

    st.caption("Powered by LLaMA 3.1 via Groq · ChromaDB · LangChain")


# --- Main chat area ---

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 View sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}** — `{s['source']}`")
                    st.caption(s["excerpt"])
                    st.divider()

# Placeholder when no PDFs loaded
if not st.session_state.chain:
    st.info("👈 Upload a PDF in the sidebar and click **Process Documents** to get started.")

# Chat input
user_input = st.chat_input("Ask something about your documents...")

if user_input:
    if not st.session_state.chain:
        st.warning("Please upload and process at least one PDF before asking questions.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Generate and stream answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(
                    st.session_state.chain,
                    st.session_state.retriever,
                    user_input
                )

            st.write(result["answer"])

            if result["sources"]:
                with st.expander("📚 View sources"):
                    for s in result["sources"]:
                        st.markdown(f"**Page {s['page']}** — `{s['source']}`")
                        st.caption(s["excerpt"])
                        st.divider()

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })