import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Chinese BERT tokenizer and model, and move the model to the GPU if available
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese').to(device)

def encode_text(text):
    # Encode text and send the inputs to the GPU if available
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run the model on GPU
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the [CLS] token embedding for the entire sequence representation
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding.cpu()  # Move back to CPU for further processing if necessary

def BERT_retrieve(query, source, corpus_dict):
    # Encode the query and move it to GPU
    query_embedding = encode_text(query).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
    
    # Encode the corpus and compute similarities
    best_score = -1
    best_doc_id = None
    for doc_id in source:
        doc_text = corpus_dict[int(doc_id)]
        doc_embedding = encode_text(doc_text).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding.cpu(), doc_embedding.cpu()).item()
        
        if similarity > best_score:
            best_score = similarity
            best_doc_id = doc_id
    
    return best_doc_id
