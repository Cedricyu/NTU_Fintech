import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding for the entire sequence representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding

def BERT_retrieve(query, source, corpus_dict):
    # Encode the query
    query_embedding = encode_text(query).unsqueeze(0)  # Add batch dimension
    
    # Encode the corpus and compute similarities
    best_score = -1
    best_doc_id = None
    for doc_id in source:
        doc_text = corpus_dict[int(doc_id)]
        doc_embedding = encode_text(doc_text).unsqueeze(0)  # Add batch dimension
        similarity = cosine_similarity(query_embedding, doc_embedding).item()
        
        if similarity > best_score:
            best_score = similarity
            best_doc_id = doc_id
    
    return best_doc_id