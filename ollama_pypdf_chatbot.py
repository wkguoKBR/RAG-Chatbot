import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from get_ollama_embedding import get_embedding_function
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from htmlTemplates import css, bot_template, user_template

CHROMA_PATH = "chroma"
DATA_PATH = "tempDir"
SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant specialized in providing answers based on the content of the uploaded PDF documents. 
You should only use the information available in the documents provided. Do not use any external knowledge.
If a question cannot be answered with the provided content, respond with "I am unable to answer the question with the provided context."
"""
PROMPT_TEMPLATE = """
Answer the question based only on the following context: 

{context}

---

Current chat history: 

{history}

---

Answer the question based on the above context: {question}
"""

def save_file(file):
    """Save uploaded file into tempDir directory"""
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, file.name), "wb") as f:
        f.write(file.getbuffer())

def load_documents(files):
    """Load documents from uploaded files with PyPDF"""
    for file in files:
        save_file(file)
    return PyPDFDirectoryLoader(DATA_PATH).load()

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100, 
        length_function=len
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks):
    """Add chunks into Chroma vector database"""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = set(db.get(include=[]).get("ids", []))
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
    else:
        print("✅ No new documents to add")
    return db

def calculate_chunk_ids(chunks: list[Document]):
    """Provide unique IDs to chunks"""
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source, page = chunk.metadata.get("source"), chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        current_chunk_index = (current_chunk_index + 1) if current_page_id == last_page_id else 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
    return chunks

def clear_database():
    """Clear the Chroma vector database"""
    if os.path.exists(CHROMA_PATH):
        print("✨ Clearing Database")
        shutil.rmtree(CHROMA_PATH)
    else:
        print("⚠️ Database not found.")

def query_rag(query_text: str):
    """Query the Chroma vector database for best matches and feed into LLM"""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    results = db.similarity_search_with_score(query_text, k=3)
    sources_content = [doc.page_content for doc, _score in results]
    context_text = "\n\n".join(sources_content)
    history_text = "\n\n".join([f"User: {msg['user_question']}\nAssistant: {msg['content']}" for msg in st.session_state.chat_history])
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE + PROMPT_TEMPLATE).format(
        context=context_text, 
        history=history_text, 
        question=query_text
    )
    print(prompt)
    response_text = Ollama(model="llama3").invoke(prompt)
    sources_id = [doc.metadata.get("id", None).replace("tempDir\\", "") for doc, _score in results]
    formatted_response = f"\n\n---\n\nQuestion: {query_text}\nResponse: {response_text}\nSources: {sources_id}"
    print(formatted_response)
    return response_text, sources_id, sources_content

def handle_userinput(user_question):
    """Take in user query and output LLM response"""
    if st.session_state.vectorstore is None:
        st.error("Vectorstore is not initialized. Please upload and process the documents first.")
        return
    response, sources_id, sources_content = query_rag(user_question)
    st.session_state.chat_history.append({'user_question': user_question, 'content': response, 'sources_id': sources_id, 'sources_content': sources_content})
    for message in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", message['user_question']), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        for i, source in enumerate(message['sources_id']):
            with st.expander(f"Source: {source}"):
                st.write(sources_content[i])

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        clear_database()

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_docs = load_documents(pdf_docs)
                chunks = split_documents(raw_docs)
                st.session_state.vectorstore = add_to_chroma(chunks)
                shutil.rmtree(DATA_PATH)

if __name__ == '__main__':
    main()