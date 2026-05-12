import os
import shutil
import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DocMind — Chat with your PDFs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── import RAG modules after page config ──────────────────────────────────────
from rag.loader    import process_pdf
from rag.embedder  import create_vectorstore, load_vectorstore, add_documents
from rag.retriever import get_retriever
from rag.chain     import build_rag_chain, stream_answer

# ── directory setup ───────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ── session state initialisation ─────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "loaded_files" not in st.session_state: st.session_state.loaded_files = []
if "chain"        not in st.session_state: st.session_state.chain        = None
if "retriever"    not in st.session_state: st.session_state.retriever    = None

# ── API key guard ─────────────────────────────────────────────────────────────
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    st.error(
        "**GROQ_API_KEY not found.**\n\n"
        "**Streamlit Cloud:** Settings → Secrets → paste `GROQ_API_KEY = \"gsk_...\"`\n\n"
        "**Local:** create `.env` with `GROQ_API_KEY=gsk_...`\n\n"
        "Get a free key at https://console.groq.com"
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 DocMind")
    st.caption("AI-powered PDF Q&A · RAG · LLaMA 3.1")
    st.divider()

    # ── PDF uploader ──────────────────────────────────────────────────
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # ── Process button ────────────────────────────────────────────────
    process_btn = st.button(
        "⚡ Process & Index",
        type="primary",
        use_container_width=True,
    )

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            # save uploaded files
            new_files = []
            for uf in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, uf.name)
                with open(save_path, "wb") as f:
                    f.write(uf.read())
                if uf.name not in st.session_state.loaded_files:
                    new_files.append((uf.name, save_path))

            if not new_files:
                st.info("All selected PDFs are already indexed.")
            else:
                # process new files
                progress = st.progress(0, text="Starting…")
                all_chunks = []
                total = len(new_files)

                for idx, (name, path) in enumerate(new_files):
                    progress.progress((idx) / total, text=f"Parsing {name}…")
                    chunks = process_pdf(path)
                    if chunks:
                        all_chunks.extend(chunks)
                        st.session_state.loaded_files.append(name)
                    else:
                        st.warning(f"Could not extract text from {name}")

                if all_chunks:
                    progress.progress(0.75, text="Building vector index…")
                    existing_vs = load_vectorstore(VECTOR_DIR)
                    if existing_vs:
                        add_documents(existing_vs, all_chunks)
                        vs = existing_vs
                    else:
                        vs = create_vectorstore(all_chunks, VECTOR_DIR)

                    progress.progress(0.90, text="Loading LLM…")
                    retriever = get_retriever(vs, k=4)
                    chain = build_rag_chain(retriever)
                    st.session_state.retriever = retriever
                    st.session_state.chain = chain
                    progress.progress(1.0, text="Done!")

                    st.success(
                        f"✅ {len(new_files)} document(s) indexed successfully! "
                        "Start chatting →"
                    )
                else:
                    st.error("No readable text found in uploaded PDFs.")

    st.divider()

    # ── Loaded files list ─────────────────────────────────────────────
    if st.session_state.loaded_files:
        st.subheader("📄 Loaded Documents")
        for name in st.session_state.loaded_files:
            st.markdown(f"- `{name}`")

        st.caption(
            f"📁 {len(st.session_state.loaded_files)} doc(s) · "
            f"💬 {len([m for m in st.session_state.messages if m['role']=='user'])} question(s)"
        )
        st.divider()

    # ── Clear button ──────────────────────────────────────────────────
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.loaded_files = []
        st.session_state.chain        = None
        st.session_state.retriever    = None
        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        st.rerun()

    st.divider()
    st.caption("🔒 Answers grounded in your docs only · No hallucination")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT PANEL
# ══════════════════════════════════════════════════════════════════════════════

# ── Welcome screen when no documents are loaded ───────────────────────────────
if not st.session_state.chain:
    st.markdown("## 👋 Welcome to DocMind")
    st.markdown(
        "Upload your PDF documents in the **left sidebar**, "
        "click **⚡ Process & Index**, then ask questions below."
    )
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📌 Summarise**\n\nWhat is this document about?")
    with col2:
        st.info("**🔍 Find Facts**\n\nWhat methodology was used in this paper?")
    with col3:
        st.info("**📊 Compare**\n\nHow do the results in Chapter 3 differ from Chapter 5?")
    st.stop()

# ── Chat header ───────────────────────────────────────────────────────────────
st.markdown("## 🧠 DocMind Chat")
st.caption(
    f"Chatting across **{len(st.session_state.loaded_files)} document(s)** · "
    "Powered by LLaMA 3.1 via Groq · Answers grounded in your PDFs"
)
st.divider()

# ── Render full chat history ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # show sources only for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**📄 Page {src['page']}** — `{src['source']}`\n\n"
                        f"> {src['excerpt']}"
                    )
                    st.divider()

# ── Chat input box ────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your documents…")

if user_input:
    # 1. show & save user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. stream assistant answer
    answer_tokens = []
    sources = []

    with st.chat_message("assistant"):
        # generator that yields only string tokens for st.write_stream
        def token_generator():
            for chunk in stream_answer(
                st.session_state.chain,
                st.session_state.retriever,
                user_input,
            ):
                if isinstance(chunk, str):
                    answer_tokens.append(chunk)
                    yield chunk
                else:
                    # final dict with sources
                    sources.extend(chunk.get("sources", []))

        # stream live to screen
        st.write_stream(token_generator())

        # 3. show sources after streaming finishes
        if sources:
            with st.expander("📚 Sources used", expanded=False):
                for src in sources:
                    st.markdown(
                        f"**📄 Page {src['page']}** — `{src['source']}`\n\n"
                        f"> {src['excerpt']}"
                    )
                    st.divider()

    # 4. save full answer + sources to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": "".join(answer_tokens),
        "sources": sources,
    })