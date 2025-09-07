import streamlit as st
from rag_pipeline import rag_answer_stream

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("DailyMail from CNN")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask a question about news articles..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamer = rag_answer_stream(prompt)
        answer = ""
        for token in streamer:  # Stream tokens as they arrive
            answer += token
            placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
