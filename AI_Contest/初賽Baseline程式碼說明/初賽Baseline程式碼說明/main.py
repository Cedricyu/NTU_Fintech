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
from tdidf import TFIDF_retrieve
from sbert import BiEncoder_with_FAISS
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(source_path):
    """Load PDF files and return a dictionary with extracted text."""
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {
        int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file))
        for file in tqdm(masked_file_ls)
    }
    return corpus_dict

def read_pdf(pdf_loc, page_infos: list = None):
    """Extract text content from a PDF file."""
    try:
        with pdfplumber.open(pdf_loc) as pdf:
            pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
            return ''.join(page.extract_text() or '' for page in pages)
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_loc}: {e}")
        return ""

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

    args = parser.parse_args()

    answer_dict = {"answers": []}

    with open(args.question_path, 'r', encoding='utf-8') as f:
        qs_ref = json.load(f)

    # Load or create corpus dictionaries
    corpus_dict_insurance = load_corpus('corpus_dict_insurance.json') or load_data(os.path.join(args.source_path, 'insurance'))
    save_corpus(corpus_dict_insurance, 'corpus_dict_insurance.json')

    corpus_dict_finance = load_corpus('corpus_dict_finance.json') or load_data(os.path.join(args.source_path, 'finance'))
    save_corpus(corpus_dict_finance, 'corpus_dict_finance.json')

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'r', encoding='utf-8') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    # Load or extract questions and texts
    questions, texts = load_saved_data('saved_data.json')
    if questions is None or texts is None:
        questions, texts = extract_questions_and_texts(key_to_source_dict)
        save_data(questions, texts, 'saved_data.json')

    for q_dict in qs_ref['questions']:
        retrieved = []
        if q_dict['category'] == 'finance':
            retrieved = BiEncoder_with_FAISS(q_dict['query'], q_dict['source'], corpus_dict_finance, top_n=3)

        elif q_dict['category'] == 'insurance':
            retrieved = BiEncoder_with_FAISS(q_dict['query'], q_dict['source'], corpus_dict_insurance, top_n=3)

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {str(key): str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = BiEncoder_with_FAISS(q_dict['query'], q_dict['source'], corpus_dict_faq, top_n=3)

        else:
            logging.error(f"Unknown category: {q_dict['category']}")
            continue

        # Append results to answer dictionary
        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

    # Save the answers dictionary as a JSON file
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)

    logging.info("Processing completed successfully.")
