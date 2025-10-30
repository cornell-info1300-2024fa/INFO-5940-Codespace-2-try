import os
import io
import shutil
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)


from rag_chain import (
    RAGConfig,
    build_vector_store,
    load_vector_store,
    retrieve,
    generate_answer,
)

load_dotenv()

st.set_page_config(page_title="INFO 5940 A1 Chat",)
st.title("INFO 5940 Chat")

#sidebar
import os
st.sidebar.caption(f"API key prefix: {os.getenv('OPENAI_API_KEY','')[:3] or 'NONE'}")
with st.sidebar:
    st.header("Settings")
    persist_dir = st.text_input("Chroma persist dir", value="chroma_db")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
    k = st.slider("Top-K retrieved", min_value=2, max_value=10, value=4)
    chat_model = st.text_input("Chat model", value=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    embed_model = st.sidebar.text_input(
    "Embed model", value="local:sentence-transformers/all-MiniLM-L6-v2"
)
    if st.button("🗑️ Delete index"):
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
            st.success(f"Removed {persist_dir}")

cfg = RAGConfig(
    persist_dir=persist_dir,
    chunk_size=int(chunk_size),
    chunk_overlap=int(chunk_overlap),
    k=int(k),
    chat_model=chat_model,
    embed_model=embed_model,
)

#upload and index
st.markdown("**Upload .txt or .pdf (multiple allowed), then click _Index uploaded files_.**")
uploads = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

def parse_upload(upload):
    name = getattr(upload, "name", "uploaded")
    if name.lower().endswith(".txt"):
        text = upload.read().decode("utf-8", errors="ignore")
        return [(text, {"source": name})]
    if name.lower().endswith(".pdf"):
        out = []
        data = upload.read()
        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages):
            content = page.extract_text() or ""
            out.append((content, {"source": name, "page": i + 1}))
        return out
    return []

def index_files(files):
    texts_with_meta = []
    for f in files:
        texts_with_meta.extend(parse_upload(f))
    if not texts_with_meta:
        st.warning("No readable text found.")
        return None
    vs = build_vector_store(texts_with_meta, cfg)
    return vs

if "vs_ready" not in st.session_state:
    st.session_state.vs_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploads and st.button("Index uploaded files"):
    vs = index_files(uploads)
    if vs:
        st.session_state.vs_ready = True
        st.success("Index built")

# loading an existing index from disk
if not st.session_state.vs_ready and os.path.isdir(cfg.persist_dir) and os.listdir(cfg.persist_dir):
    try:
        _ = load_vector_store(cfg)
        st.session_state.vs_ready = True
        st.info("Loaded existing index from disk.")
    except Exception as e:
        st.warning(f"Could not load index: {e}")

#chat history-
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

#ask questions
q = st.chat_input("Ask a question...")
if q:
    if not st.session_state.vs_ready:
        st.warning("Please upload and index documents first.")
    else:
        st.session_state.messages.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            st.write("Retrieving")
            vs = load_vector_store(cfg)
            docs = retrieve(vs, q, cfg.k)
            st.write("Generating")
            try:
                ans = generate_answer(q, docs, cfg)
            except Exception as e:
                ans = f"Error: {e}\n\nCheck that `OPENAI_API_KEY` is set in `.env`."
            st.markdown(ans)
            st.session_state.messages.append(("assistant", ans))
