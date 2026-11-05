import os
import tempfile
from pathlib import Path

import streamlit as st

from rag import build_chain, build_vectorstore

st.set_page_config(page_title="TechStream PolicyBot", page_icon="🤖")
st.title("🤖 TechStream PolicyBot")
st.caption("Ask questions about our policies. Answers come only from the documents.")

DEFAULT_DOC = Path(__file__).parent / "data" / "company_policy.txt"

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    if not os.getenv("OPENAI_API_KEY"):
        key = st.text_input("OpenAI API key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key

    uploads = st.file_uploader(
        "Upload policy documents (.pdf / .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    k = st.slider("Chunks to retrieve", 1, 6, 3)
    if st.button("Clear chat"):
        st.session_state.messages = []

if not os.getenv("OPENAI_API_KEY"):
    st.info("Add your OpenAI API key in the sidebar or in a .env file to start.")
    st.stop()


# ---------- Index ----------
@st.cache_resource(show_spinner="Indexing documents...")
def get_chain(file_payloads, k):
    tmp = Path(tempfile.mkdtemp())
    paths = []
    for name, data in file_payloads:
        p = tmp / name
        p.write_bytes(data)
        paths.append(p)
    if not paths:
        paths = [DEFAULT_DOC]
    store, n_chunks = build_vectorstore(paths)
    return build_chain(store, k=k), n_chunks


payloads = tuple((u.name, u.getvalue()) for u in uploads or [])
chain, n_chunks = get_chain(payloads, k)
st.sidebar.success(f"Indexed {n_chunks} chunks")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write(s)

if question := st.chat_input("e.g. Can I return an opened laptop?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chain.invoke(question)
                answer = result["answer"]
                sources = [d.page_content for d in result["docs"]]
            except Exception as e:
                answer, sources = f"Error: {e}", []
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write(s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
