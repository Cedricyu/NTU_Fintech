from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import preprocess_text

def TFIDF_retrieve(query, sources, corpus_dict, top_n=1):
    """Retrieve relevant documents for the given query using TF-IDF and cosine similarity."""
    
    # Preprocess and tokenize corpus
    preprocessed_corpus_dict = {str(file): preprocess_text(corpus_dict.get(str(file), "")) for file in sources}
    
    # Convert corpus to list
    documents = list(preprocessed_corpus_dict.values())
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Transform query to the same TF-IDF space
    query_vec = vectorizer.transform([preprocess_text(query)])
    
    # Compute cosine similarities
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get the top N results
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_results = [int(list(preprocessed_corpus_dict.keys())[index]) for index in top_indices]
    
    return top_results[0]
