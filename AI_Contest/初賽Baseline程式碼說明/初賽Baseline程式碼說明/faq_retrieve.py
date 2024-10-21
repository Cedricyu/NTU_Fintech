
from ckiptagger import WS, POS, NER, data_utils, construct_dictionary
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
import os
import torch
from torch.cuda.amp import autocast
from device import get_device

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer_l = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
def number_to_chinese(num):
    units = ["", "十", "百", "千", "萬", "億"]
    digits = "零一二三四五六七八九"
    result = ""
    str_num = str(num)
    length = len(str_num)
    
    for i, digit in enumerate(str_num):
        digit_value = int(digit)
        if digit_value != 0:
            result += digits[digit_value] + units[length - i - 1]
        elif not result.endswith("零"):
            result += "零"
    # 處理 "一十" -> "十" 的情況
    result = result.replace("一十", "十")
    # 去除末尾的 "零"
    result = result.rstrip("零")
    return result


custom_dict = {}
custom_dict["對帳單"]=1
custom_dict["刷臉"] = 1
custom_dict["綜合對帳單"] = 1
custom_dict["對帳單"] =1
custom_dict["支付寶"] =1
for i in range(1, 501):  # 假設需要生成「第1條」到「第500條」
    chinese_number = number_to_chinese(i)
    custom_dict[f"第{chinese_number}條"] = 1
    custom_dict[f"第{i}季"] =1

custom_dictionary = construct_dictionary(custom_dict)
stopwords = {"的", "了", "之", "在", "和", "也", "有", "是","於","\n","：","。","，","「","」","【","】","、","；"}
# data_utils.download_data_gdown("./") # gdrive-ckip
ws = WS("./data")


def remove_stopwords(words, stopwords):
    # 過濾掉不需要的詞
    return [word for word in words if word not in stopwords]

def get_ckip_tokenization(text):
    words = ws([text],sentence_segmentation=True,
            segment_delimiter_set={'：', '，', '\n', '。','【','】'},recommend_dictionary=custom_dictionary)
    tokenized_query = remove_stopwords(words[0], stopwords)
    return tokenized_query

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

model_l = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(get_device())

def embadding_e5_l(input_texts, relevant_doc_ids):
    # Tokenize input texts
    batch_dict = tokenizer_l(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    # Move tensors to the GPU
    for key in batch_dict.keys():
        batch_dict[key] = batch_dict[key].to(get_device())
    # Use no_grad for inference
    with torch.no_grad():
        outputs = model_l(**batch_dict)

    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity scores
    query_embedding = embeddings[0]  # First is the query embedding
    doc_embeddings = embeddings[1:]  # Remaining are document embeddings
    scores = (query_embedding @ doc_embeddings.T) * 100

    # Convert scores to a Python list
    scores = scores.tolist()

    # Store max scores for each document ID
    doc_score_dict = {}
    for score, doc_id in zip(scores, relevant_doc_ids):
        doc_score_dict[doc_id] = max(doc_score_dict.get(doc_id, float('-inf')), score)

    # Cleanup
    del outputs
    torch.cuda.empty_cache()

    return doc_score_dict

stopwords = {"的", "了", "之", "在", "和", "也", "有", "是","於","\n","：","。","，","「","」","【","】","、","；"}

def FAQ_retrieve(query, relevant_doc_ids, relevant_docs):
    tokenized_query = get_ckip_tokenization(query)
    filtered_query = remove_stopwords(tokenized_query, stopwords)
    question_str = ' '.join(filtered_query)
    # print("query: ",question_str)
    input_texts = [f"query: {question_str}"] + [f"passage: {doc}" for doc in relevant_docs]
    embadding_scores = embadding_e5_l(input_texts, relevant_doc_ids)
    max_key = max(embadding_scores, key=embadding_scores.get)
    print(max_key)
    return max_key