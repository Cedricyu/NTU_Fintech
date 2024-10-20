import os
import json
import argparse
from tqdm import tqdm
import jieba
import pdfplumber
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import logging
import pytesseract
from PIL import Image
from FlagEmbedding import FlagReranker  # 確保正確導入 FlagReranker
from collections import defaultdict
import re

# 設置日志記錄
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 基於正則表達式的中文分句函數
def split_sentences(text):
    """
    使用正則表達式根據中文句子結尾標點符號分割句子。
    """
    # 定義中文句子結尾的標點符號
    sentence_endings = re.compile(r'([。！？])')
    sentences = sentence_endings.split(text)
    combined_sentences = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip() + sentences[i+1]
        if sentence:
            combined_sentences.append(sentence)
    # 處理最後一個句子（如果沒有結尾標點）
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        combined_sentences.append(sentences[-1].strip())
    return combined_sentences

# 語義分塊函數
def semantic_chunking(text, max_sentences=10, overlap=2):
    """
    將文本分割成語義上連貫的塊，每個塊包含最多 `max_sentences` 句，並且有 `overlap` 句的重疊。
    """
    sentences = split_sentences(text)  # 使用正則表達式分句
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(' '.join(current_chunk))
            # 實現重疊
            if overlap > 0:
                current_chunk = current_chunk[-overlap:]
            else:
                current_chunk = []
    
    # 添加剩餘的句子
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logging.info(f"Total chunks created: {len(chunks)}")
    return chunks

# Function to read PDF and split into chunks with OCR fallback
def read_pdf(pdf_loc, page_infos: list = None, max_sentences=10, overlap=2):
    try:
        pdf = pdfplumber.open(pdf_loc)
    except Exception as e:
        logging.error(f"Error opening PDF file {pdf_loc}: {e}")
        return []
    
    try:
        pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    except IndexError:
        logging.warning(f"Page range {page_infos} out of bounds for file {pdf_loc}. Extracting all pages.")
        pages = pdf.pages
    
    pdf_text = ''
    for page_number, page in enumerate(pages, start=1):
        try:
            text = page.extract_text()
            if text:
                logging.info(f"Extracted {len(text)} characters from page {page_number} of {pdf_loc} using pdfplumber.")
                pdf_text += text + "\n\n"
            else:
                # 如果pdfplumber無法提取文本，嘗試使用OCR
                logging.info(f"No text found on page {page_number} of {pdf_loc}. Attempting OCR.")
                # 將頁面轉換為圖像
                image = page.to_image(resolution=300).original
                # 使用PIL將圖像轉換為RGB模式
                pil_image = image.convert("RGB")
                # 使用pytesseract進行OCR
                ocr_text = pytesseract.image_to_string(pil_image, lang='chi_tra')  # 使用繁體中文語言包
                if ocr_text.strip():
                    logging.info(f"Extracted {len(ocr_text)} characters from page {page_number} of {pdf_loc} using OCR.")
                    pdf_text += ocr_text + "\n\n"
                else:
                    logging.warning(f"OCR failed to extract text from page {page_number} of {pdf_loc}.")
        except Exception as e:
            logging.error(f"Error processing page {page_number} in {pdf_loc}: {e}")
    pdf.close()
    
    # 語義分塊
    splits = semantic_chunking(pdf_text, max_sentences, overlap)
    return splits

# Function to load data from source path
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {}
    all_documents = []
    all_doc_ids = []
    missing_pdfs = []
    for file in tqdm(masked_file_ls, desc=f"Loading data from {source_path}"):
        try:
            file_id = int(file.replace('.pdf', ''))
        except ValueError:
            logging.warning(f"Skipping non-PDF or improperly named file: {file}")
            continue
        file_path = os.path.join(source_path, file)
        splits = read_pdf(file_path)
        if not splits:
            logging.warning(f"No content extracted from file: {file_path}")
            missing_pdfs.append(file)
        corpus_dict[file_id] = splits
        all_documents.extend(splits)
        all_doc_ids.extend([file_id] * len(splits))
    if missing_pdfs:
        logging.info(f"Total missing PDFs: {len(missing_pdfs)}")
        for pdf in missing_pdfs:
            logging.info(f"Missing PDF: {pdf}")
    return corpus_dict, all_documents, all_doc_ids

# Function to load Reranker1 model (FlagReranker-based)
def load_reranker1(device):
    try:
        reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        logging.info("Reranker1 model (FlagReranker) loaded successfully.")
        return reranker
    except Exception as e:
        logging.error(f"Error loading Reranker1 model: {e}")
        return None

# Function to load Reranker2 model
def load_reranker2(model_name, device):
    try:
        reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        reranker.to(device)
        reranker.eval()
        logging.info(f"Reranker2 model '{model_name}' loaded successfully.")
        return reranker, tokenizer
    except Exception as e:
        logging.error(f"Error loading Reranker2 model {model_name}: {e}")
        return None, None

# Function to compute reranker2 scores using AutoModelForSequenceClassification
def compute_reranker2_scores(query, documents, reranker_model, reranker_tokenizer, device, batch_size=32):
    reranker_model.eval()
    sentence_pairs = [[query, doc] for doc in documents]
    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Computing Reranker2 Scores"):
            batch = sentence_pairs[i:i+batch_size]
            inputs = reranker_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = reranker_model(**inputs)
            logits = outputs.logits.view(-1).float()
            # 應用 sigmoid 轉換為概率分數
            probabilities = torch.sigmoid(logits)
            batch_scores = probabilities.cpu().numpy()
            scores.extend(batch_scores)
    return scores

def main(args):
    answer_dict = {"answers": []}

    # Read questions file
    try:
        with open(args.question_path, 'r', encoding='utf8') as f:
            qs_ref = json.load(f)
        logging.info(f"Loaded questions from {args.question_path}")
    except Exception as e:
        logging.error(f"Error reading question file {args.question_path}: {e}")
        return

    # Load reference data
    corpus_dict_finance, documents_finance, doc_ids_finance = load_data(os.path.join(args.source_path, 'finance'))
    corpus_dict_insurance, documents_insurance, doc_ids_insurance = load_data(os.path.join(args.source_path, 'insurance'))

    # Load FAQ mapping and split
    try:
        with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        logging.info("Loaded FAQ mapping.")
    except Exception as e:
        logging.error(f"Error reading FAQ mapping file: {e}")
        key_to_source_dict = {}
        faq_documents = []
        faq_doc_ids = []

    # 檢查是否包含所有需要的 FAQ doc_id
    required_faq_doc_ids = set()
    for q in qs_ref.get('questions', []):
        if 101 <= int(q.get('qid', 0)) <= 150 and q.get('category') == 'faq':
            required_faq_doc_ids.update(q.get('source', []))

    missing_faq_doc_ids = [doc_id for doc_id in required_faq_doc_ids if doc_id not in key_to_source_dict]
    if missing_faq_doc_ids:
        logging.warning(f"The following FAQ doc_ids are missing in 'faq/pid_map_content.json': {missing_faq_doc_ids}")
    else:
        logging.info("All required FAQ doc_ids are present in 'faq/pid_map_content.json'.")

    # Prepare FAQ documents
    faq_documents = []
    faq_doc_ids = []
    for key, value in key_to_source_dict.items():
        for q in value:
            combined = f"問題：{q['question']} 答案：{' '.join(q['answers'])}"
            faq_documents.append(combined)
            faq_doc_ids.append(key)

    # 建立 faq_corpus_dict，映射 doc_id 到 list of strings
    faq_corpus_dict = defaultdict(list)
    for doc_id, doc in zip(faq_doc_ids, faq_documents):
        faq_corpus_dict[doc_id].append(doc)
        logging.debug(f"Added to faq_corpus_dict: doc_id={doc_id}, doc='{doc[:30]}...'")  # 僅顯示前30個字符

    # 驗證 faq_corpus_dict 的內容
    for doc_id, docs in faq_corpus_dict.items():
        for doc in docs:
            if not isinstance(doc, str):
                logging.error(f"faq_corpus_dict 的 doc_id: {doc_id} 包含非字符串文檔: {doc}")
            else:
                logging.debug(f"faq_corpus_dict 的 doc_id: {doc_id} 包含字符串文檔: {doc[:30]}...")

    # Aggregate all documents
    all_documents = documents_finance + documents_insurance + faq_documents
    all_doc_ids = doc_ids_finance + doc_ids_insurance + faq_doc_ids

    # Build a mapping from doc_id to list of indices in all_documents
    doc_id_to_indices = {}
    for idx, doc_id in enumerate(all_doc_ids):
        if doc_id not in doc_id_to_indices:
            doc_id_to_indices[doc_id] = []
        doc_id_to_indices[doc_id].append(idx)

    # Initialize Reranker models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Initialize Reranker1 (FlagReranker)
    reranker1 = load_reranker1(device)
    if reranker1 is None:
        logging.error("Reranker1 model failed to load. Exiting.")
        return

    # Initialize Reranker2
    reranker2_model_name = 'maidalun1020/bce-reranker-base_v1'  # Existing Reranker model
    reranker2_model, reranker2_tokenizer = load_reranker2(reranker2_model_name, device)
    if reranker2_model is None:
        logging.error("Reranker2 model failed to load. Exiting.")
        return

    # Log loaded doc_ids for verification
    loaded_doc_ids = set(all_doc_ids)
    logging.info(f"Total loaded doc_ids: {len(loaded_doc_ids)}")

    # Process each question
    for q_dict in tqdm(qs_ref.get('questions', []), desc="Processing questions"):
        qid = q_dict.get('qid')
        query = q_dict.get('query')
        category = q_dict.get('category')
        source = q_dict.get('source')

        if not all([qid, query, category, source]):
            logging.warning(f"Skipping incomplete question entry: {q_dict}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Select the appropriate corpus
        if category == 'finance':
            corpus_dict = corpus_dict_finance
            docs = documents_finance
            doc_ids = doc_ids_finance
        elif category == 'insurance':
            corpus_dict = corpus_dict_insurance
            docs = documents_insurance
            doc_ids = doc_ids_insurance
        elif category == 'faq':
            corpus_dict = faq_corpus_dict  # 使用 faq_corpus_dict
            # docs 和 doc_ids 不再需要
        else:
            logging.warning(f"Unknown category '{category}' for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Get relevant documents based on source
        if category == 'faq':
            relevant_docs = []
            relevant_doc_ids = []
            for key in source:
                docs_for_key = corpus_dict.get(key, [])
                if not docs_for_key:
                    logging.warning(f"QID: {qid} - No documents found for faq key: {key}")
                relevant_docs.extend(docs_for_key)
                relevant_doc_ids.extend([key] * len(docs_for_key))
        else:
            relevant_docs = []
            relevant_doc_ids = []
            for doc_id in source:
                docs_for_id = corpus_dict.get(doc_id, [])
                if not docs_for_id:
                    logging.warning(f"QID: {qid} - No documents found for doc_id: {doc_id}")
                relevant_docs.extend(docs_for_id)
                relevant_doc_ids.extend([doc_id] * len(docs_for_id))

        if not relevant_docs:
            logging.warning(f"No valid documents found for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 進行數據驗證：確保所有相關文檔都是字符串
        valid_relevant_docs = []
        valid_relevant_doc_ids = []
        invalid_docs = []
        for doc, doc_id in zip(relevant_docs, relevant_doc_ids):
            if isinstance(doc, str) and doc.strip():
                valid_relevant_docs.append(doc)
                valid_relevant_doc_ids.append(doc_id)
            else:
                invalid_docs.append((qid, doc_id, doc))
        
        if invalid_docs:
            for invalid in invalid_docs:
                logging.error(f"QID: {invalid[0]}, Invalid Document ID: {invalid[1]}, Document: {invalid[2]}")
        
        if not valid_relevant_docs:
            logging.warning(f"No valid string documents found for QID: {qid} after filtering.")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 確認所有 relevant_docs 都是字符串
        for idx, doc in enumerate(valid_relevant_docs):
            if not isinstance(doc, str):
                logging.error(f"QID: {qid}, Relevant Doc Index: {idx}, Document is not a string: {doc}")
        
        # 直接使用所有相關文檔進行重排序

        # Reranker1: compute scores using FlagReranker
        try:
            # 準備 [query, passage] 的對列表
            score_inputs = [[query, doc] for doc in valid_relevant_docs]
            # 使用 FlagReranker 計算分數
            reranker1_scores = reranker1.compute_score(score_inputs)
            logging.info(f"QID: {qid}, Reranker1 scores: {reranker1_scores[:5]}")  # 只記錄前5個分數以簡化日誌
            
            # 標準化 Reranker1 分數到 -1 ~ 1
            normalized_reranker1_scores = [score / 10 for score in reranker1_scores]
            logging.info(f"QID: {qid}, Normalized Reranker1 scores: {normalized_reranker1_scores[:5]}")
        except Exception as e:
            logging.error(f"Error during reranking with Reranker1 for QID: {qid}: {e}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # Reranker2: compute scores using AutoModelForSequenceClassification
        try:
            reranker2_scores = compute_reranker2_scores(query, valid_relevant_docs, reranker2_model, reranker2_tokenizer, device)
            logging.info(f"QID: {qid}, Reranker2 scores: {reranker2_scores[:5]}")  # 只記錄前5個分數以簡化日誌
        except Exception as e:
            logging.error(f"Error during reranking with Reranker2 for QID: {qid}: {e}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 平均分數
        if len(normalized_reranker1_scores) != len(valid_relevant_docs) or len(reranker2_scores) != len(valid_relevant_docs):
            logging.error(f"Score length mismatch for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        average_scores = [(nr1 + r2) / 2 for nr1, r2 in zip(normalized_reranker1_scores, reranker2_scores)]

        # 將文檔與平均分數及 doc_id 組合
        doc_scores = list(zip(valid_relevant_docs, average_scores, valid_relevant_doc_ids))

        # 按平均分數降序排序
        top_n = 1  # 根據需求調整
        top_n_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_n]

        if not top_n_docs:
            logging.warning(f"No documents after reranking for QID: {qid}")
            answer_dict['answers'].append({"qid": qid, "retrieve": None})
            continue

        # 選擇最佳文檔
        best_doc, best_score, best_doc_id = top_n_docs[0]
        logging.info(f"QID: {qid}, Selected Document ID: {best_doc_id}, Average Score: {best_score:.4f}")

        # 將結果添加到 answer_dict
        answer_dict['answers'].append({"qid": qid, "retrieve": best_doc_id})

    # 保存結果
    try:
        with open(args.output_path, 'w', encoding='utf8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
        logging.info(f"Results saved to {args.output_path}")
    except Exception as e:
        logging.error(f"Error writing output file {args.output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Two-Reranker Retrieval System with FlagReranker and BCE Reranker.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to the questions JSON file.')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the source data directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output JSON file.')
    parser.add_argument('--max_sentences', type=int, default=10, help='Maximum number of sentences per chunk.')
    parser.add_argument('--overlap', type=int, default=2, help='Number of overlapping sentences between chunks.')

    args = parser.parse_args()

    main(args)
