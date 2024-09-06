from langchain_community.embeddings.ollama import OllamaEmbeddings # yes

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings