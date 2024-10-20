import os
import json
import argparse
import logging
from tqdm import tqdm
import jieba  # Used for Chinese text segmentation
import pdfplumber  # Tool for extracting text from PDF files
from rank_bm25 import BM25Okapi  # Use BM25 algorithm for document retrieval
from collections import defaultdict
from preprocess import preprocess_text
from retrieve import retrieve
from  faq_retrieve import FAQ_retrieve
import pytesseract
from PIL import Image
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(source_path, chuck_size):
    """Load PDF files and return a dictionary with extracted text."""
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {
        str(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file), max_chars_per_chunk=chuck_size)
        for file in tqdm(masked_file_ls)
    }
    return corpus_dict

def split_text_into_chunks(text, max_chars_per_chunk):
    """
    Split text into chunks of complete sentences.

    Args:
        text (str): The text to split.
        max_chars_per_chunk (int): Maximum number of characters per chunk.

    Returns:
        list: A list of strings where each string contains up to `max_chars_per_chunk` characters.
    """
    # Split the text into sentences
    sentences = text.split('。')  # Assuming Chinese full stops as sentence delimiters
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars_per_chunk:  # +1 for the delimiter
            if current_chunk:
                current_chunk += "。"
            current_chunk += sentence.strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence.strip()

    if current_chunk:
        chunks.append(current_chunk)  # Add the last chunk if it exists

    return chunks

def read_pdf(pdf_loc, max_chars_per_chunk=512):
    """
    Extract text content from a PDF file and use OCR if the PDF contains no extractable text.
    
    Args:
        pdf_loc (str): Path to the PDF file.
        max_chars_per_chunk (int): Maximum number of characters per chunk in the returned list.

    Returns:
        list: A list of strings where each string contains up to `max_chars_per_chunk` characters.
    """
    try:
        with pdfplumber.open(pdf_loc) as pdf:
            pages = pdf.pages
            extracted_text = ''.join(page.extract_text() or '' for page in pages).strip()

            # If no text was extracted, apply OCR
            if not extracted_text:
                logging.info(f"No text extracted from PDF {pdf_loc}. Attempting OCR...")
                extracted_text = ''
                for page in pages:
                    # Convert the page to an image and perform OCR
                    page_image = page.to_image()
                    ocr_text = pytesseract.image_to_string(page_image.original, lang='chi_sim')  # Specify language for better accuracy
                    extracted_text += ocr_text

                extracted_text = extracted_text.strip()

                if not extracted_text:
                    logging.warning(f"OCR failed to extract any meaningful text from PDF {pdf_loc}.")
                    return []

            # Split extracted text into chunks of complete sentences
            return split_text_into_chunks(extracted_text, max_chars_per_chunk)

    except Exception as e:
        logging.error(f"Error reading PDF {pdf_loc}: {e}")
        return []
 
def extract_questions_and_texts(key_to_source_dict):
    """Extract questions and corresponding texts from the source dictionary."""
    questions = []
    texts = []
    for entries in key_to_source_dict.values():
        for entry in entries:
            question = entry['question']
            for answer in entry['answers']:
                questions.append(question)
                texts.append(answer)
    return questions, texts

def save_data(questions, texts, output_path):
    """Save questions and texts to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"questions": questions, "texts": texts}, f, ensure_ascii=False, indent=4)

def load_saved_data(file_path):
    """Load saved questions and texts from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['questions'], data['texts']
    return None, None

def save_corpus(corpus_dict, output_path):
    """Save the corpus dictionary to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus_dict, f, ensure_ascii=False, indent=4)

def load_corpus(file_path):
    """Load a corpus dictionary from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def BM25_retrieve(query, sources, corpus_dict, top_n=1):
    """Retrieve relevant documents for the given query using BM25."""
    
    # Preprocess corpus once and create a reverse lookup dictionary
    preprocessed_corpus_dict = {str(file): preprocess_text(corpus_dict.get(str(file), "")) for file in sources}
    reverse_lookup = {preprocessed_value: int(key) for key, preprocessed_value in preprocessed_corpus_dict.items()}
    
    # Tokenize preprocessed corpus
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in preprocessed_corpus_dict.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(query))
    
    # Retrieve the top N relevant documents
    ans = bm25.get_top_n(tokenized_query, list(preprocessed_corpus_dict.values()), n=top_n)
    
    # Map best matches to filenames using the reverse lookup
    results = [reverse_lookup[match] for match in ans if match in reverse_lookup]
    
    # Return the first result or an empty list if no matches
    return results[0] if results else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to the questions file.')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the reference data.')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the answers.')
    parser.add_argument('--chunk_size', type=int, required=False, default=512 ,help='Maximum chunk size for segmentation.')

    args = parser.parse_args()

    answer_dict = {"answers": []}

    with open(args.question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)

    # Load or create corpus dictionaries
    corpus_dict_insurance = load_corpus('corpus_dict_insurance.json') or load_data(os.path.join(args.source_path, 'insurance'),args.chunk_size)
    save_corpus(corpus_dict_insurance, 'corpus_dict_insurance.json')

    corpus_dict_finance = load_corpus('corpus_dict_finance.json') or load_data(os.path.join(args.source_path, 'finance'),args.chunk_size)
    save_corpus(corpus_dict_finance, 'corpus_dict_finance.json')

    # Load FAQ mapping and split
    try:
        with open(os.path.join('processed_pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # Read reference data file
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        logging.info("Loaded FAQ mapping.")
    except Exception as e:
        logging.error(f"Error reading FAQ mapping file: {e}")
        key_to_source_dict = {}
        faq_documents = []
        faq_doc_ids = []
        
    # Prepare FAQ documents
    faq_documents = []
    faq_doc_ids = []
    for key, value in key_to_source_dict.items():
        for q in value:
            combined = f"問題：{q['問題']} 答案：{' '.join(q['答案'])}"
            faq_documents.append(combined)
            faq_doc_ids.append(key)

    faq_dict = {}
    for doc_id, doc in zip(faq_doc_ids, faq_documents):
        if doc_id not in faq_dict:
            faq_dict[doc_id] = []
        faq_dict[doc_id].append(doc)

    for q_dict in tqdm(qs_ref['questions'], desc="Processing Questions"):
        retrieved = []
        if q_dict['category'] == 'finance':
            retrieved = retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance)

        elif q_dict['category'] == 'insurance':
            retrieved = retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)

        elif q_dict['category'] == 'faq':
            source = q_dict['source']
            corpus_dict = key_to_source_dict
            docs = faq_documents
            doc_ids = faq_doc_ids
            relevant_docs = []
            relevant_doc_ids = []
            for key in source:
                docs_for_key = corpus_dict.get(key, [])
                if not docs_for_key:
                    logging.warning(f"QID: {q_dict['qid']} - No documents found for faq key: {key}")
                relevant_docs.extend(docs_for_key)
                relevant_doc_ids.extend([key] * len(docs_for_key))
            
            relevant_doc_ids_u = list(dict.fromkeys(relevant_doc_ids))
            source_passage = [faq_dict[doc_id] for doc_id in relevant_doc_ids_u]
            retrieved = FAQ_retrieve(q_dict['query'], q_dict['source'], source_passage)

        else:
            logging.error(f"Unknown category: {q_dict['category']}")
            continue

        # Append results to answer dictionary
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

    # Save the answers dictionary as a JSON file
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)

    logging.info("Processing completed successfully.")
