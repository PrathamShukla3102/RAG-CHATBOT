import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG Chatbot")
st.write("Upload documents and ask questions")

# ---------- Upload Section ----------
st.sidebar.header("Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    if st.sidebar.button("Process File"):
        with st.spinner("Processing..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())}
            )

            if response.status_code == 200:
                st.sidebar.success("File processed successfully")
            else:
                st.sidebar.error("Upload failed")


# ---------- Chat Section ----------
st.header("Ask Questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your question")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query}
            )

            if response.status_code == 200:
                data = response.json()

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": data["answer"],
                    "sources": data["sources"]
                })
            else:
                st.error("Query failed")


# ---------- Display Chat ----------
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"### User: {chat['question']}")
    st.markdown(f"### Answer:\n{chat['answer']}")

    with st.expander("Sources"):
        for s in chat["sources"]:
            st.write(f"• {s['source']} (chunk {s['chunk_id']})")