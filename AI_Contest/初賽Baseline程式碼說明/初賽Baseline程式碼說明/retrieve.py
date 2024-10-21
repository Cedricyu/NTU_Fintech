import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from preprocess import preprocess_text
import logging
import torch.nn.functional as F
from device import get_device
# Initialize device
device = get_device()

# Initialize the cross-encoder model for reranking
cross_encoder_model_name = 'maidalun1020/bce-reranker-base_v1'  # Replace with your model name
cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name).to(device)

def rerank_with_cross_encoder(query, top_docs):
    """
    Rerank the top documents using a cross-encoder model.

    Parameters:
    - query: The query text to search with.
    - top_docs: A list of top document texts to rerank.

    Returns:
    - The highest scored document based on its relevance to the query.
    """
    # Prepare inputs for the model
    inputs = cross_encoder_tokenizer(
        [[query, doc] for doc in top_docs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Get model predictions
    with torch.no_grad():
        logits = cross_encoder_model(**inputs).logits.cpu().numpy()
    # print("Logits:", logits)  # Check logits before softmax

    # Apply softmax to convert logits to probabilities
    scores = F.softmax(torch.tensor(logits), dim=0).numpy()

    # print("Scores:",scores)  # Debug: Print scores to verify softmax output

    # Extract the probabilities for the positive class and find the highest one
    highest_score_index = scores.argmax()  # Assuming the positive class is at index 1
    # print(highest_score_index)
    highest_scoring_doc = top_docs[highest_score_index]
    return highest_score_index

def retrieve(query, sources, corpus_dict, top_n=3):
    """
    Retrieve and rerank top N similar documents to the query using a cross-encoder.

    Parameters:
    - query: The query text to search with.
    - sources: A list of document IDs (numbers).
    - corpus_dict: A JSON dictionary mapping document IDs to lists of text chunks.
    - top_n: Number of top results to retrieve.

    Returns:
    - The most relevant document ID to the query, or None if no documents found.
    """
    all_chunks = []
    chunk_to_doc_map = {}

    # Collect all chunks and their corresponding document IDs
    chunk_id = 0
    for doc_id in sources:
        chunks = corpus_dict.get(str(doc_id), [])
        
        if not chunks:
            logging.warning(f"No chunks found for document ID: {doc_id}.")
            continue  # Skip to the next document ID if no chunks are found

        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_doc_map[chunk_id] = str(doc_id)
            chunk_id += 1  # Increment the chunk_id counter


    if not all_chunks:
        logging.warning("No chunks found across the provided document IDs. Returning empty result.")
        return None  # Return None if there are no chunks

    preprocessed_sources = [preprocess_text(chunk) for chunk in all_chunks]

    # Use only the cross-encoder to rerank
    top_documents = rerank_with_cross_encoder(query, preprocessed_sources)

    
    if top_documents:
        # print(chunk_to_doc_map[top_documents])
        return int(chunk_to_doc_map[top_documents])
    else:
        logging.warning("No matching document IDs found in the mapping.")
        return None  # Return None if no matches found
