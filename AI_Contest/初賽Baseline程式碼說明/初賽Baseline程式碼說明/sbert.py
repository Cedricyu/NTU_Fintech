import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import preprocess_text

bi_encoder = SentenceTransformer('hfl/chinese-roberta-wwm-ext')

def encode_text(text):
    return bi_encoder.encode(text, convert_to_numpy=True)

def build_faiss_index(embeddings, dimension):
    # Initialize a FAISS index for cosine similarity (normalized dot product)
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (cosine similarity when vectors are normalized)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def BiEncoder_with_FAISS(query, sources, corpus_dict, top_n=3):
    preprocessed_corpus_dict = {
        str(file): preprocess_text(corpus_dict.get(str(file), ""))
        for file in sources
    }
    
    document_embeddings = np.array([encode_text(text) for text in preprocessed_corpus_dict.values()])
    dimension = document_embeddings.shape[1]  # Embedding dimension
    
    index = build_faiss_index(document_embeddings, dimension)
    
    query_embedding = encode_text(preprocess_text(query)).reshape(1, -1)
    
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_n)
    
    top_files = [list(preprocessed_corpus_dict.keys())[i] for i in indices[0]]
    
    return int(top_files[0])