import os
import time
import logging
import numpy as np
import pandas as pd
import faiss # ---------- FAISS -----------
from datasets import load_dataset # ---------- HuggingFace / Datasets -----------
from sentence_transformers import SentenceTransformer # ---------- Sentence Embeddings -----------
from config_manager import ConfigManager
from typing import List, Dict
# ---------- Logging Setup -----------
logging.basicConfig(
    filename="app.log",filemode="a",level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s",
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import BaseNode
import json 
# ---------- GLOBAL VARIABLES -----------
config = ConfigManager.load_config("../configs/central.yaml") 
MODEL = SentenceTransformer("all-MiniLM-L6-v2")    


def get_embedding(text : list):
    """
    List of sentences
    Return embedding (can be used for both query and sentences(text from document))
    """
    embeddings = MODEL.encode(text)
    return embeddings

def chunk_text(text : str,
               chunk_size : int,
               document_name : str, 
               chunk_save_path : str,
               tokenizer 
               ):
    """
    Create textchunks of chunk_size from text. 
    Convert text to JSON save file at the chunk path.

    Args:
        text(str) - the string to divide into chunks.
        chunk_size - the size of chunks to divide the text.
        chunks_path - the path where json chunks are saved.

    Returns tokens length list
    """
    #load chunks from already created json
    os.makedirs(f"{chunk_save_path}{document_name}", exist_ok=True)
    chunk_file_path = f"{chunk_save_path}{document_name}/{document_name}-csize{chunk_size}.json"
        # Check if JSON file exists
    if os.path.exists(chunk_file_path):
        # Load existing JSON and return token lengths
        with open(chunk_file_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            token_lengths_list = [chunk["token_lengths"] for chunk in chunks_data]
        return token_lengths_list  # Return token lengths from the existing JSON
    document = Document(text = text)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents([document])
    chunks_data = []
    #convert to JSON friendly version
    token_lengths_list = []
    for i, node in enumerate(nodes):
        token_length = len(tokenizer.tokenize(node.text))
        token_lengths_list.append(token_length)
        chunks_data.append({
                "chunk_id" : f"{document_name}+chunk_{i}",
                "document_name": document_name,
                "chunk_index" : i, 
                "text" : node.text,
                "token_lengths" : token_length})
    #save JSON to chunk path
    with open(chunk_file_path, "w", encoding="utf=8") as f:
        json.dump(chunks_data, f , indent=4, ensure_ascii=False)

    return token_lengths_list

def build_embeddings(json_path, index_path:str) :
    """
    Read a json file and build embeddings from its text file.
    Args:
        json_path -  Path to the JSON file)
    Returns - index - index instance
    """
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    if not os.path.exists(index_path):
        index = faiss.IndexFlatL2(config["embedding_dim"])
        faiss.write_index(index, index_path)

    index = faiss.read_index(index_path)
    with open(json_path, "r" , encoding="utf-8") as f:
        data = json.load(f)

    texts = [chunk["text"] for chunk in data]
    embeddings = get_embedding(texts)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

# ========== 3. SIMILARITY SEARCH FUNCTION ==========

def get_query_indices(
    faiss_index: faiss.Index,
    k_example: int,
    query_text : str):
    """
    Get top 5 similar indexes(chunks) for query, from the embeddings created
    before.
    
    :param faiss_index: A pre-built or loaded FAISS index
    :param k_example: Number of neighbors to retrieve
    :param query_text: The query to encode & search

    Return: 
        indexes - top 5 similar indices
        query_indices_retrival_duration - time it takes to retrive indices, distances
    """
    t_query = time.time()
    query_embedding = np.array(get_embedding([query_text]), dtype=np.float32)  # Use model server functio
    distances, indices = faiss_index.search(query_embedding, k_example)
    query_indices_retreval_duration = time.time() - t_query
    return query_indices_retreval_duration, indices

# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
import torch

def load_model_tokenizer(model_path):
    """Loads a LLaMA causal language model and its tokenizer with no gradient tracking to save memory."""
    model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
            )  
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
    return model, tokenizer

def generate_with_kv_cache(token_ids : List[int],
                            max_gen_len : int, 
                            model1: AutoModelForCausalLM,
                            past_key_values = None):
    """
    Return past key values
       Args - 
            token_ids - List of tokens to change
            max_gen_len 
       """
    #view adds another dimension e.g (1, 364)
    input_tensor = torch.tensor(token_ids, dtype=torch.long).view(1, -1).to(model1.device)
    logging.info("Input tensor shape" + str(input_tensor.shape))
    res_tokens = []

    with torch.no_grad():
        for i in range(max_gen_len):
            output_dict = model1(input_tensor, past_key_values=past_key_values)
            logging.info("Logits shape " + str(output_dict["logits"].shape))
            logging.info(f"Logits shape for the last token: {output_dict['logits'][:, -1, :].shape}")
            logging.info(f"First 5 logits for last token: {output_dict['logits'][:, -1, :][0, :5]}")

            tok = torch.argmax(output_dict["logits"][:, -1, :])
        
            logging.info("Input tensor shape" + str(input_tensor.shape))
            past_key_values = output_dict["past_key_values"]

            # logging.info("Past key values" + str(past_key_values.shape))
            if int(tok) in [128001, 128009, 128000]:
                break
            res_tokens.append(int(tok))
            input_tensor = tok.view(1, -1)

    return past_key_values, tokenizer.decode(res_tokens)

def compute_kv_cache_size(past_key_values):
    if past_key_values is None:
        return 0  # No KV cache stored yet
    total_size = 0
    for layer_idx, (key_tensor, value_tensor) in enumerate(past_key_values):
        # Compute memory usage per tensor
        key_size = key_tensor.numel() * key_tensor.element_size()
        value_size = value_tensor.numel() * value_tensor.element_size()
        # Accumulate total KV cache size
        total_size += key_size + value_size
    logging.info(f"Total KV Cache Size: {total_size / (1024 * 1024):.2f} MB")
    return total_size


def get_kv_cache(input_dict : Dict[int, torch.Tensor]):
    """
    Get the KV cache of already generated text.
    
    Args:
        input_dict - dictionary containing input id and attention masks tensors.

    Returns - kv cache of the past values.
    """
    with torch.no_grad():
        outputs = model(input_dict['input_ids'], use_cache=True)
        kv_cache = outputs.past_key_values
    
    return kv_cache

def measure_retained_indexes(query_text,  faiss_index, query_index):
    """
    Measures the # of indexes (chunks) retained by each prefix of a query wrt
    the top 5 indexes retrieved by the full query.

    Args:
        query_index (list[int]): The top 5 most similar indexes retrieved from the FAISS index for the full query.
        faiss_index (faiss.Index): The FAISS index storing chunk embeddings.

    Returns:
        dict: `partial_query_retained_index`, a mapping where:
            - Keys are partial queries.
            - Values represent the number of indexes retained by the partial query,
              relative to the top 5 indexes retrieved by the full query."""
    words = query_text.split()
    partial_query_retained_index = {}
    for i in range(len(words)):
        partial_query = " ".join(words[: len(words) - i])
        partial_query_embedding = get_embedding(partial_query).reshape(1, -1)
        distances, partial_indices = faiss_index.search(partial_query_embedding, 5)
        missing_indexes = query_index - partial_indices
        partial_query_retained_index[partial_query] = np.count_nonzero(missing_indexes == 0)
    return partial_query_retained_index

def measure_rate(query_text, faiss_index, query_index):
    """
    Return:
        prefix_query_to_retrieved_indexes : dict
            A dictionary where:
            - Keys are partial queries.
            - Values are lists containing:
                - The number of retrieved indexes.
                - The retrieval time.
            Ensures that at least the required number of indexes are retrieved to include the top 5 results.
    """
    words = query_text.split()
    prefix_query_to_retrieved_indexes = {}

    for i in range(len(words)):
        partial_query = " ".join(words[: len(words) - i])
        partial_query_embedding = get_embedding(partial_query).reshape(1, -1)
        start_time = time.time()
        for j in range(0, 60000, 10):
            _ , prefix_index = faiss_index.search(partial_query_embedding, j+5)
            prefix_index.flatten()
            # logging.info(f"Query Indexes : {query_index} - Partial {prefix_index} = Missing indexes {np.setdiff1d(query_index, prefix_index)}")
            if np.all(np.isin(query_index, prefix_index)):
                prefix_query_to_retrieved_indexes[partial_query] = [j+5]
                retrieval_time = time.time() - start_time
                prefix_query_to_retrieved_indexes.setdefault(partial_query, []).append(retrieval_time)
                break

    return prefix_query_to_retrieved_indexes

def measure_typing_time(query_text, tokenizer,  one_word_time=0.66):
    """
    Return a dictionary for the total typing of for each prefix of the query  

    Args:
        one_word_time - time it takes to type a single word.
        query_text - the query to calculate the typing time for.
    
    Returns - dictionary with prefix and the time it takes to type the prefix
    """
    prefix_time = {}
    tokens = tokenizer.tokenize(query_text)
    for i, token in enumerate(tokens):
        if i == 0:
            prefix_query = token
        else:
            prefix_query += " " + token
        prefix_time[prefix_query] = (i + 1) * one_word_time
    return prefix_time

#TODO - remove main demo after copying code to run_pipeline.
# # ========== 4. MAIN DEMO ==========

# from itertools import chain
# from utils import load_pdf
# from utils import generate_retained_chunks_graph
# from data.queries import queries_list
# from utils import generate_match_rate
# from utils import generate_match_rate_words
# from utils import plot_token_length_distribution
# from utils import plot_typing_time

# if __name__ == "__main__":

#     model, tokenizer = load_model_tokenizer(model_path="meta-llama/Llama-3.1-8B-Instruct")
#     pdf_to_text = load_pdf(pdf_path="./data/pdf_files/paul-graham-essays.pdf", 
#                             output_txt_path="./data/text_files/paul-graham-essays.txt")

#     token_length_list = chunk_text(pdf_to_text, 
#                 chunk_size= 200, 
#                 document_name="paul-graham-essay",
#                 chunk_save_path="./data/json_files/",
#                 tokenizer=tokenizer
#                 )
#     plot_token_length_distribution(token_length_list=token_length_list, 
#                                    append_suffix="distribution",
#                                    )
#     #get the distribution of lengths
    
#     paul_g_index = build_embeddings(json_path="./data/json_files/paul-graham-essays.json", 
#                      index_path = "./index_store/paul-graham-essays.faiss")
    
#     complex_qa_20 = queries_list.complex_pg_20
#     simple_qa = queries_list.simple_pg
#     complex_qa_50 = queries_list.complex_pg_50
    

#     # 3) Prepare a DataFrame for logging
#     columns = ["k_example", "embed_time", "search_time", "query_text", "retrival_time"]
#     log_df = pd.DataFrame(columns=columns)

#     k_values = [5]  # Different k-values to test
#     retrieved_dictionary = {}
#     for k_example in k_values:
#         for query_text in complex_qa_20:
#             log_df, retrieved_text = get_indices(
#                 faiss_index=paul_g_index,
#                 k_example=k_example,  # Change k dynamically
#                 query_text=query_text,
#                 log_df=log_df,
#             )

#             query_index_map = measure_retention(query_text, faiss_index=paul_g_index)

#             query_text_retained = next(iter(query_index_map))
#             generate_retained_chunks_graph(query_index_map=query_index_map, query_text=query_text_retained, append_suffix = f"simple_qa")

            
#     for k_example in k_values:
#         for query_text in complex_qa_20:
#             query_to_required_prefixes = measure_rate(query_text=query_text, faiss_index=paul_g_index)
#             query_text_match= next(iter(query_to_required_prefixes))
#             generate_match_rate(query_map = query_to_required_prefixes,
#                                 append_suffix = "complex_match_rate")
#             generate_match_rate_words(query_map=query_to_required_prefixes,
#                                       append_suffix="complex_match_rate_words")

#     for query_text in complex_qa_50:
#         prefix_time_dictionary = measure_typing_time(query_text=query_text, tokenizer=tokenizer)
#         plot_typing_time(prefix_time_dictionary, append_suffix = "typing")

