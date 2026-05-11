import os
import time
import streamlit as st
from rag.loader   import process_pdf
from rag.embedder import create_vectorstore, load_vectorstore, add_documents
from rag.retriever import get_retriever
from rag.chain    import build_rag_chain, stream_answer

# ── dirs ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — dark glassmorphism with amber/gold accents
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0D0D0F;
    --surface:   #16161A;
    --surface2:  #1E1E24;
    --border:    rgba(255,255,255,0.07);
    --gold:      #F5C842;
    --gold-dim:  rgba(245,200,66,0.15);
    --gold-glow: rgba(245,200,66,0.08);
    --text:      #E8E8EC;
    --muted:     #7A7A8C;
    --user-bg:   #1E2535;
    --ai-bg:     #161B26;
    --radius:    14px;
    --font:      'DM Sans', sans-serif;
    --mono:      'DM Mono', monospace;
}

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.2rem !important; }

/* ── Main container ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Scrollable chat area ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1.5rem 2rem 6rem;
    min-height: calc(100vh - 160px);
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    animation: fadeUp 0.3s ease both;
}
.msg-row.user  { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0);    }
}

.avatar {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    border: 1px solid var(--border);
}
.avatar.ai   { background: linear-gradient(135deg,#1E2535,#252B3B); color: var(--gold); }
.avatar.user { background: linear-gradient(135deg,#252010,#2E2810); color: var(--gold); }

.bubble {
    max-width: 72%;
    padding: 0.85rem 1.1rem;
    border-radius: var(--radius);
    line-height: 1.65;
    font-size: 0.93rem;
    border: 1px solid var(--border);
    position: relative;
}
.bubble.ai   { background: var(--ai-bg);   border-radius: 4px var(--radius) var(--radius) var(--radius); }
.bubble.user { background: var(--user-bg); border-radius: var(--radius) 4px var(--radius) var(--radius); }

/* ── Source pills inside bubble ── */
.src-row { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.6rem; }
.src-pill {
    font-family: var(--mono);
    font-size: 0.72rem;
    padding: 2px 9px;
    background: var(--gold-dim);
    color: var(--gold);
    border-radius: 20px;
    border: 1px solid rgba(245,200,66,0.25);
    cursor: pointer;
}

/* ── Input bar ── */
.input-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, var(--bg) 70%, transparent);
    padding: 1rem 2rem 1.2rem;
    z-index: 100;
}

/* ── Streamlit chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-glow) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}
[data-testid="stChatInputSubmitButton"] svg { stroke: var(--gold) !important; }

/* ── Sidebar elements ── */
.sidebar-logo {
    display: flex; align-items: center; gap: 0.7rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.sidebar-logo .logo-icon {
    font-size: 1.6rem;
    width: 42px; height: 42px;
    background: var(--gold-dim);
    border: 1px solid rgba(245,200,66,0.3);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
}
.sidebar-logo .logo-text h2 {
    margin:0; font-size:1.15rem; font-weight:600; color:var(--text);
}
.sidebar-logo .logo-text span {
    font-size:0.72rem; color:var(--muted);
}

.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1rem 0 0.5rem;
}

.doc-chip {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.45rem 0.7rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.82rem; color: var(--text);
    margin-bottom: 0.35rem;
}
.doc-chip .dot { width:6px;height:6px;border-radius:50%;background:var(--gold);flex-shrink:0; }

.stat-row { display:flex; gap:0.5rem; margin-top:0.8rem; }
.stat-box {
    flex:1; padding:0.6rem 0.5rem; text-align:center;
    background:var(--surface2); border:1px solid var(--border); border-radius:8px;
}
.stat-box .val { font-size:1.1rem; font-weight:600; color:var(--gold); }
.stat-box .lbl { font-size:0.65rem; color:var(--muted); margin-top:1px; }

/* ── Buttons ── */
.stButton > button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    font-family: var(--font) !important;
    font-size: 0.88rem !important;
    transition: all 0.18s !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: var(--gold) !important;
    background: var(--gold-dim) !important;
    color: var(--gold) !important;
}

/* ── Primary process button ── */
.primary-btn > button {
    background: var(--gold) !important;
    color: #0D0D0F !important;
    border: none !important;
    font-weight: 600 !important;
}
.primary-btn > button:hover {
    background: #E5B830 !important;
    color: #0D0D0F !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed rgba(245,200,66,0.3) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; font-size:0.85rem !important; }

/* ── Spinner / status ── */
[data-testid="stStatusWidget"] { background: var(--surface2) !important; border-color: var(--border) !important; }

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: calc(100vh - 200px);
    text-align: center; padding: 2rem;
}
.welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.welcome-wrap h1 { font-size: 2rem; font-weight: 600; color: var(--text); margin: 0 0 0.4rem; }
.welcome-wrap p  { color: var(--muted); font-size: 0.95rem; max-width: 400px; line-height:1.6; }
.hint-grid { display:flex; gap:0.7rem; margin-top:1.5rem; flex-wrap:wrap; justify-content:center; }
.hint-card {
    background: var(--surface2); border:1px solid var(--border);
    border-radius:10px; padding:0.75rem 1rem;
    font-size:0.82rem; color:var(--muted); max-width:180px; text-align:left;
    cursor:pointer; transition: border-color .18s;
}
.hint-card:hover { border-color: var(--gold); color:var(--text); }
.hint-card strong { display:block; color:var(--text); margin-bottom:3px; font-size:0.85rem; }

/* ── Top header bar ── */
.top-bar {
    position: sticky; top: 0; z-index: 50;
    background: rgba(13,13,15,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0.7rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
}
.top-bar .title { font-size:1rem; font-weight:600; color:var(--text); display:flex;align-items:center;gap:0.5rem; }
.top-bar .badge {
    font-size:0.68rem; font-family:var(--mono);
    padding:2px 8px; border-radius:20px;
    background:var(--gold-dim); color:var(--gold);
    border:1px solid rgba(245,200,66,0.25);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--surface2); border-radius:10px; }

/* ── Streamlit markdown inside bubble ── */
.bubble p { margin:0 0 0.4em; }
.bubble p:last-child { margin-bottom:0; }
.bubble code { font-family:var(--mono); background:rgba(255,255,255,0.07); padding:1px 5px; border-radius:4px; }

/* ── Textarea autoresize focus ring ── */
textarea:focus { outline:none !important; }

/* ── Sidebar scrollbar ── */
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 3px; }
</style>
""", unsafe_allow_html=True)


# ── session state ─────────────────────────────────────────────────────────────
for key, val in {
    "messages":     [],
    "loaded_files": [],
    "chain":        None,
    "retriever":    None,
    "total_qs":     0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── API key guard ─────────────────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;
                min-height:100vh;flex-direction:column;gap:1rem;padding:2rem;">
        <div style="font-size:3rem;">🔑</div>
        <h2 style="color:#F5C842;margin:0;">GROQ_API_KEY not set</h2>
        <p style="color:#7A7A8C;text-align:center;max-width:400px;line-height:1.6;">
            Add your Groq API key to Streamlit Cloud Secrets or a local .env file.<br>
            Get a free key at <strong style="color:#E8E8EC;">console.groq.com</strong>
        </p>
        <code style="background:#16161A;border:1px solid rgba(255,255,255,0.07);
                     padding:0.5rem 1rem;border-radius:8px;color:#F5C842;font-size:0.85rem;">
            GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
        </code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">🧠</div>
        <div class="logo-text">
            <h2>DocMind</h2>
            <span>RAG · LLaMA 3.1 · Groq</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="section-label">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    col_proc, col_clr = st.columns([3, 1])

    with col_proc:
        process_clicked = st.button("⚡  Process", use_container_width=True,
                                    key="process_btn",
                                    help="Embed and index uploaded PDFs")
    with col_clr:
        clear_clicked = st.button("🗑", use_container_width=True,
                                   key="clear_btn",
                                   help="Clear all documents and chat")

    # Process documents
    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            new_files = []
            for f in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, f.name)
                with open(save_path, "wb") as out:
                    out.write(f.read())
                if f.name not in st.session_state.loaded_files:
                    new_files.append((f.name, save_path))

            if not new_files:
                st.info("All PDFs already loaded.")
            else:
                all_chunks = []
                with st.status("Processing…", expanded=True) as status:
                    for name, path in new_files:
                        st.write(f"📄 Parsing **{name}**")
                        chunks = process_pdf(path)
                        if chunks:
                            all_chunks.extend(chunks)
                            st.session_state.loaded_files.append(name)
                            st.write(f"   ✓ {len(chunks)} chunks")
                        else:
                            st.warning(f"No text found in {name}")

                    if all_chunks:
                        st.write("🔢 Building vector index…")
                        existing_vs = load_vectorstore(VECTOR_DIR)
                        if existing_vs:
                            add_documents(existing_vs, all_chunks)
                            vectorstore = existing_vs
                        else:
                            vectorstore = create_vectorstore(all_chunks, VECTOR_DIR)

                        st.write("🤖 Connecting to Groq LLM…")
                        retriever = get_retriever(vectorstore, k=4)
                        chain     = build_rag_chain(retriever)
                        st.session_state.retriever = retriever
                        st.session_state.chain     = chain
                        status.update(label="✅ Ready — start chatting!", state="complete")
                    else:
                        status.update(label="No usable text found.", state="error")

    # Clear
    if clear_clicked:
        import shutil
        st.session_state.messages     = []
        st.session_state.loaded_files = []
        st.session_state.chain        = None
        st.session_state.retriever    = None
        st.session_state.total_qs     = 0
        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR)
        os.makedirs(VECTOR_DIR, exist_ok=True)
        st.rerun()

    # Loaded docs list
    if st.session_state.loaded_files:
        st.markdown('<div class="section-label">Loaded Documents</div>', unsafe_allow_html=True)
        for name in st.session_state.loaded_files:
            short = name[:26] + "…" if len(name) > 28 else name
            st.markdown(f"""
            <div class="doc-chip">
                <div class="dot"></div>
                <span>{short}</span>
            </div>
            """, unsafe_allow_html=True)

        # Stats row
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="val">{len(st.session_state.loaded_files)}</div>
                <div class="lbl">Docs</div>
            </div>
            <div class="stat-box">
                <div class="val">{st.session_state.total_qs}</div>
                <div class="lbl">Queries</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position:absolute;bottom:1.2rem;left:1.2rem;right:1.2rem;
                border-top:1px solid rgba(255,255,255,0.06);padding-top:0.8rem;">
        <div style="font-size:0.7rem;color:#4A4A5C;line-height:1.6;">
            🔒 Answers grounded in your documents<br>
            ⚡ Streaming via Groq · LLaMA 3.1 8B<br>
            🗃 Semantic search · ChromaDB
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Top bar
if st.session_state.chain:
    status_badge = f'{len(st.session_state.loaded_files)} doc(s) loaded'
    badge_html   = f'<span class="badge">{status_badge}</span>'
else:
    badge_html   = '<span class="badge" style="background:rgba(255,255,255,0.05);color:#4A4A5C;">No docs loaded</span>'

st.markdown(f"""
<div class="top-bar">
    <div class="title">🧠 &nbsp;DocMind <span style="color:#4A4A5C;font-weight:400;font-size:0.85rem;">/ Chat</span></div>
    {badge_html}
</div>
""", unsafe_allow_html=True)


# ── Welcome screen (no docs loaded) ──────────────────────────────────────────
if not st.session_state.chain:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">📄</div>
        <h1>Chat with your documents</h1>
        <p>Upload PDFs in the sidebar, click <strong>⚡ Process</strong>, then ask anything about them.</p>
        <div class="hint-grid">
            <div class="hint-card"><strong>📌 Summarise</strong>What is this paper about?</div>
            <div class="hint-card"><strong>🔍 Find facts</strong>What methodology was used?</div>
            <div class="hint-card"><strong>📊 Compare</strong>How do the results compare?</div>
            <div class="hint-card"><strong>📖 Explain</strong>Define the key concepts.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Chat history ──────────────────────────────────────────────────────────
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar user">👤</div>
                <div class="bubble user">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Build source pills
            pills_html = ""
            if msg.get("sources"):
                pills_html = '<div class="src-row">'
                for src in msg["sources"]:
                    pills_html += f'<span class="src-pill">📄 p.{src["page"]} · {src["source"]}</span>'
                pills_html += "</div>"

            st.markdown(f"""
            <div class="msg-row">
                <div class="avatar ai">🧠</div>
                <div class="bubble ai">{msg["content"]}{pills_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    st.markdown('<div class="input-bar">', unsafe_allow_html=True)
    user_input = st.chat_input("Ask anything about your documents…")
    st.markdown('</div>', unsafe_allow_html=True)

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.total_qs += 1

        # Display user message immediately
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user">👤</div>
            <div class="bubble user">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)

        # Stream AI response
        answer_tokens = []
        sources       = []

        with st.chat_message("assistant", avatar="🧠"):
            # st.write_stream accepts a generator of strings
            def token_generator():
                for chunk in stream_answer(
                    st.session_state.chain,
                    st.session_state.retriever,
                    user_input
                ):
                    if isinstance(chunk, str):
                        answer_tokens.append(chunk)
                        yield chunk
                    else:
                        # sentinel dict with sources
                        sources.extend(chunk.get("sources", []))

            # Stream the answer word-by-word
            st.write_stream(token_generator())

            # Show source pills after streaming completes
            if sources:
                pills_html = '<div class="src-row" style="margin-top:0.5rem;">'
                for src in sources:
                    tip  = src["excerpt"][:120].replace('"', "'")
                    pg   = src["page"]
                    nm   = src["source"]
                    pills_html += f'<span class="src-pill" title="{tip}">📄 p.{pg} · {nm}</span>'
                pills_html += "</div>"
                st.markdown(pills_html, unsafe_allow_html=True)

                with st.expander("📚 View source excerpts"):
                    for src in sources:
                        st.markdown(
                            f"**Page {src['page']}** — `{src['source']}`\n\n"
                            f"> {src['excerpt']}"
                        )
                        st.divider()

        # Save assistant message
        full_answer = "".join(answer_tokens)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer,
            "sources": sources,
        })
        st.rerun()
