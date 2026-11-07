import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ask import K, MODEL, SYSTEM_PROMPT, describe
from ingest import DB_DIR, EMBED_MODEL

load_dotenv()

st.set_page_config(page_title="docuagent", page_icon="📚")



@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


@st.cache_resource
def load_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


@st.cache_data
def load_courses():
    metas = load_db().get(include=["metadatas"])["metadatas"]
    return sorted({m["course"] for m in metas})


if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file.")
    st.stop()

db = load_db()
client = load_client()



st.title("docuagent")
st.caption("Ask about my lecture slides. Answers use only the slides and cite them.")

with st.sidebar:
    st.header("Settings")
    course = st.selectbox("Course", ["All courses"] + load_courses())
    k = st.slider("Slides to retrieve", min_value=1, max_value=10, value=K)

with st.form("ask_form"):
    question = st.text_input("Your question", placeholder="What is a masked autoencoder?")
    submitted = st.form_submit_button("Ask")



if submitted and question.strip():
    where = None if course == "All courses" else {"course": course}

    with st.spinner("Searching slides..."):
        docs = db.similarity_search(question, k=k, filter=where)

    if not docs:
        st.warning("No matching slides found.")
        st.stop()

    context = "\n\n".join(
        f"[{i}] ({describe(d)})\n{d.page_content}" for i, d in enumerate(docs, start=1)
    )

    with st.spinner("Writing answer..."):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
        except Exception as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    st.subheader("Answer")
    answer = response.choices[0].message.content
    answer = answer.replace("【", "[").replace("】", "]")
    st.markdown(answer)

    st.subheader("Sources")
    for i, d in enumerate(docs, start=1):
        with st.expander(f"[{i}] {describe(d)}"):
            st.text(d.page_content)