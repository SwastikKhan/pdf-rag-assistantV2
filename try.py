try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

import os
import streamlit as st
import pathlib
import nltk
import logging
import chromadb
import tempfile
import shutil
from typing import List
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import google.generativeai as genai
import pymupdf4llm

# Download NLTK resources quietly
nltk.download("punkt", quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define folder and model settings
PDF_FOLDER = pathlib.Path(r"D:\College_use\github\pdf-rag-assistant\document")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.chunk_size = 512
Settings.chunk_overlap = 50


def read_pdfs() -> List[str]:
    """Read and convert PDFs from a folder to markdown."""
    logging.info("Reading PDFs...")
    output_list = []
    for pdf_path in PDF_FOLDER.glob("*.pdf"):
        md_text = pymupdf4llm.to_markdown(str(pdf_path), write_images=False)
        output_list.append(md_text)
    return output_list


def chunk_documents(md_contents: List[str]) -> List[TextNode]:
    """Split markdown content into semantic chunks with metadata."""
    logging.info("Chunking documents...")
    config = LanguageConfig(language="english", spacy_model="en_core_web_md")
    semantic_parser = SemanticDoubleMergingSplitterNodeParser(
        language_config=config,
        initial_threshold=0.4,
        appending_threshold=0.5,
        merging_threshold=0.5,
        max_chunk_size=500
    )
    nodes = []
    for doc_id, content in enumerate(md_contents):
        doc = Document(text=content, metadata={"source": f"doc_{doc_id}"})
        nodes.extend(semantic_parser.get_nodes_from_documents([doc]))
    return nodes


def create_vector_store(nodes: List[TextNode], store_path: str = "./chroma_db"):
    logging.info("Creating vector store...")
    chroma_client = chromadb.PersistentClient(path=store_path)
    chroma_collection = chroma_client.get_or_create_collection("pdf_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )
    return vector_index


def process_uploaded_files(uploaded_files) -> List[str]:
    """Process uploaded PDF files and return a list of markdown content."""
    md_contents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.flush()
            md_text = pymupdf4llm.to_markdown(tmp_file.name, write_images=False)
            md_contents.append(md_text)
        os.remove(tmp_file.name)
    return md_contents


def initialize_vector_store_from_upload(uploaded_files):
    with st.spinner("Processing PDFs... Please wait."):
        md_contents = process_uploaded_files(uploaded_files)
        nodes = chunk_documents(md_contents)
        # Use a temporary folder for the uploaded PDFs vector store
        store_path = "./temp_chroma_db"
        # Clear any existing vector store at this path
        if os.path.isdir(store_path):
            shutil.rmtree(store_path)
        create_vector_store(nodes, store_path=store_path)
        st.session_state.vector_store_path = store_path
    st.success("Vector store created from uploaded PDFs!")


def get_llm_response_gemini(prompt: str) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    generation_config = {
        "temperature": 0.3,
        "response_mime_type": "text/plain"
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config
    )
    response = model.generate_content(prompt)
    return response.text


def get_context(question: str, store_path: str = "./chroma_db") -> List[str]:
    chroma_client = chromadb.PersistentClient(path=store_path)
    chroma_collection = chroma_client.get_collection("pdf_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=Settings.embed_model
    )
    retriever = vector_index.as_retriever(similarity_top_k=4)
    nodes = retriever.retrieve(question)
    contexts = []
    for node in nodes:
        content = node.node.get_content()
        if content and len(content.strip()) > 50:
            contexts.append(content)
    return contexts


def get_answer(question: str, context: List[str], history: List[dict]) -> str:
    prompt = f"""Below given is the question of the user and the context of a document based on the asked question.
Give me the answer to the question based on the given context.
Give me summarized text of the pdf when asked for.
Give me the summary of the pdf when asked for.
Note: Answer should have at least 50 characters if possible.

Question: {question}

Context: {str(context)}

History: {str(history)}

Answer:
"""
    final_output = get_llm_response_gemini(prompt)
    return final_output


def chat_interface():
    st.title("Chat with Your Personal PDF Assistant")

    st.header("Upload Your PDFs")
    uploaded_files = st.file_uploader("Upload your PDF file(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process PDFs"):
            initialize_vector_store_from_upload(uploaded_files)
    else:
        st.info("Please upload one or more PDF files to begin.")

    # if "vector_store_path" not in st.session_state:
    #     st.warning("No vector store found. Please upload and process your PDFs first.")
    #     return

    if "history" not in st.session_state:
        st.session_state.history = []

    st.header("Chat")
    # Display the conversation history using chat_message
    for chat in st.session_state.history:
        st.chat_message("user").markdown(chat["question"])
        st.chat_message("assistant").markdown(chat["answer"])

    # Use st.chat_input to capture new user input
    user_input = st.chat_input("Enter your question:")
    if user_input:
        store_path = st.session_state.vector_store_path
        context = get_context(user_input, store_path=store_path)
        answer = get_answer(user_input, context, st.session_state.history)
        st.chat_message("user").markdown(user_input)
        st.chat_message("assistant").markdown(answer)

        st.session_state.history.append({"question": user_input, "answer": answer})


if __name__ == "__main__":
    chat_interface()
