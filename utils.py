import pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings

def initialize_pinecone(api_key, index_name):
    pinecone.init(api_key=api_key)
    return pinecone.Index(index_name)

def get_embeddings_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformerEmbeddings(model_name=model_name)
