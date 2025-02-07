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

# ---------- GLOBAL VARIABLES -----------
config = ConfigManager.load_config("../configs/central.yaml") 


from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import BaseNode
import json 

def chunk_text(text : str,
               chunk_size : int,
               document_name : str, 
               chunk_path : str,
               ):
    """
    Create chunks of size chunk_size from text file.

    Args:
        text(str) - the string to divide into chunks.
        chunk_size - the size of chunks to divide the text.
        chunks_path - the path where json chunks are saved.
    """
    #load chunks from already created json
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as file:
            chunks_data = json.load(file)
        return chunks_data
            
    document = Document(text = text)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents([document])
    chunks_data = []
    #convert to JSON friendly version
    for i, node in enumerate(nodes):
        chunks_data.append({
                "chunk_id" : f"{document_name}+chunk_{i}",
                "document_name": document_name,
                "chunk_index" : i, 
                "text" : node.text})
    #save to JSON
    with open(chunk_path, "w", encoding="utf=8") as f:
        json.dump(chunks_data, f , indent=4, ensure_ascii=False)

from model_server import get_embedding, start_model_server
start_model_server()
def build_embeddings(json_path, index_path:str) :
    """
    Read a json file and build embeddings from its text file.
    Args:
        json_path -  Path to the JSON file)
    Returns - index - index instance
    """
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Index exist. Contains {index.ntotal} total embeddings with dim {index.d}")
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

def check_scores(
    faiss_index: faiss.Index,
    k_example: int,
    query_text: str,
    log_df: pd.DataFrame,
):
    """
    Perform a similarity search using the provided FAISS index and log execution times.
    
    :param faiss_index: A pre-built or loaded FAISS index
    :param k_example: Number of neighbors to retrieve
    :param query_text: The query to encode & search
    :param log_df: DataFrame for logging results

    :return: (updated log_df, retrieved_examples)
    """
    
    with open("./data/json_files/paul-graham-essays.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    t_query = time.time()
    query_embedding = np.array(get_embedding([query_text]), dtype=np.float32)  # Use model server functio
    print(f"Query embedding shape: {query_embedding.shape}")
    embed_time = time.time() - t_query
    # 2) Perform the FAISS search
    t_search = time.time()
    distances, indices = faiss_index.search(query_embedding, k_example)
    search_time = time.time() - t_search
    retrieved_chunks = [data[i]["text"] for i in indices[0] if i < len(data)]
    logging.info("Retreived chunks " + str(retrieved_chunks))
    

    # 4) Logging
    logging.info(f"Query: {query_text}")
    logging.info(f"Embed time: {embed_time:.4f} sec | Search time: {search_time:.4f} sec")
    logging.info(f"Top {k_example} indices: {indices[0]}")
    logging.info(f"Top {k_example} distances: {distances[0]}")

    # 5) Append row to log_df
    log_df.loc[len(log_df)] = [
        k_example,
        embed_time,
        search_time,
        query_text
    ]

    return log_df, retrieved_chunks

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

import torch
def encode_input(text: str, tokenizer, chunk_size, pad_token_id=0):
    """
    Convert string to tokenized chunks of fixed size.

    Args:
        text (str): The input string to tokenize.
        tokenizer: Tokenizer object to convert text to token-ids.
        chunk_size (int, optional): Size of each chunk. Defaults to 200.
        pad_token_id (int, optional): Token ID used for padding. Defaults to 0.

    Returns:
        dict: Dictionary with "input_ids" and "attention_mask" as tensors.
    """
    
    # Tokenize the text
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    
    # Create chunks of fixed size
    input_ids = []
    attention_mask = []
    
    for i in range(0, len(tokenized_text), chunk_size):
        chunk = tokenized_text[i:i + chunk_size]
        
        # Pad if the chunk is smaller than chunk_size
        pad_length = chunk_size - len(chunk)
        
        if pad_length > 0:
            chunk.extend([pad_token_id] * pad_length)  # Add padding
        
        input_ids.append(chunk)
        attention_mask.append([1] * len(chunk) + [0] * pad_length)  # Mask padding
        
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }

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

def measure_retention(query_text,  faiss_index):
    """
    Measure the number of index(chunks) retained by prefix of a query.

    Args:
        query_text : the full query text 
        faiss_index : index created for storing chunk embeddings
    Returns : 
        partial_query_retained_index : mapping of key(partial query) and number of retained indexes
        (currently both )
    """
    with open("./data/json_files/paul-graham-essays.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    full_query_embedding = np.array(get_embedding([query_text]), dtype=np.float32)  # Use model server functio
    distances, full_indices = faiss_index.search(full_query_embedding, 5)
    retrieved_embeddings = np.array([faiss_index.reconstruct(int(idx)) for idx in full_indices[0]])
    retrieved_chunks = [data[i]["text"] for i in full_indices[0] if i < len(data)]

    words = query_text.split()
    partial_query_retained_index = {}
    for i in range(len(words)):
        partial_query = " ".join(words[: len(words) - i])
        logging.info(f"Partial Query is {partial_query}")
        partial_query_embedding = get_embedding(partial_query).reshape(1, -1)
        logging.info(f"Partial Query embedding shape {partial_query_embedding.shape}")
        distances, partial_indices = faiss_index.search(partial_query_embedding, 5)
        missing_indexes = full_indices - partial_indices
        logging.info(f"Full : {full_indices} - Partial {partial_indices} = Missing indexes {missing_indexes}")
        partial_query_retained_index[partial_query] = np.count_nonzero(missing_indexes == 0)
    return partial_query_retained_index

def measure_rate(query_text, faiss_index):
    with open("./data/json_files/paul-graham-essays.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    query_embedding = np.array(get_embedding([query_text]), dtype=np.float32)
    _ , query_indexes = faiss_index.search(query_embedding, 5)
    retreived_embeddings = np.array([faiss_index.reconstruct(int(idx)) for idx in query_indexes[0]])
    query_indexes.flatten()
    prefix_query_to_retreived_indexes = {}
    words = query_text.split()
    for i in range(len(words)):
        partial_query = " ".join(words[: len(words) - i])
        partial_query_embedding = get_embedding(partial_query).reshape(1, -1)
        for j in range(0, 6000, 10):
            _ , prefix_index = faiss_index.search(partial_query_embedding, j+5)
            prefix_index.flatten()
            logging.info(f"Query Indexes : {query_indexes} - Partial {prefix_index} = Missing indexes {np.setdiff1d(query_indexes, prefix_index)}")
            if np.all(np.isin(query_indexes, prefix_index)):
                prefix_query_to_retreived_indexes[partial_query] = j+5
                break
    print("Hello World")
    return prefix_query_to_retreived_indexes
    # words = query_text.split()
    # for i in range(len(words)):
    #     partial_query = " ".join(words[: len(words) - 1])
    #     partial_query_embedding = get_embedding(partial_query).reshape(1, -1)
    #     partial_distance , partial_index = faiss_index.search(partial_query_embedding, 5)
    #     missing_indices = full_indices - partial_indices
    #     partial_index = partial_index.flatten()
    #     full_indices = full_indices.flatten()
    #     for j in range(1):
    #         distances_cp, partial_index = faiss_index.search(partial_query_embedding, j)
    #         retreived_embeddings = np.array([fais])



# ========== 4. MAIN DEMO ==========

from itertools import chain
from utils import load_pdf
from utils import generate_retained_chunks_graph
from data.queries import queries_list
from utils import generate_match_rate
if __name__ == "__main__":
    pdf_to_text = load_pdf(pdf_path="./data/pdf_files/paul-graham-essays.pdf", 
                            output_txt_path="./data/text_files/paul-graham-essays.txt")

    chunked_text = chunk_text(pdf_to_text, 
                              chunk_size= 200, 
                              document_name="paul-graham-essay",
                              chunk_path="./data/json_files/paul-graham-essays.json")
    
    paul_g_index = build_embeddings(json_path="./data/json_files/paul-graham-essays.json", 
                     index_path = "./index_store/paul-graham-essays.faiss")
    
    # complex_qa = queries_list.complex_pg
    simple_qa = queries_list.simple_pg
    

    # 3) Prepare a DataFrame for logging
    columns = ["k_example", "embed_time", "search_time", "query_text"]
    log_df = pd.DataFrame(columns=columns)


    k_values = [5]  # Different k-values to test
    retrieved_dictionary = {}
    model, tokenizer = load_model_tokenizer(model_path="meta-llama/Llama-3.1-8B-Instruct")

    for k_example in k_values:
        for query_text in simple_qa:
            log_df, retrieved_text = check_scores(
                faiss_index=paul_g_index,
                k_example=k_example,  # Change k dynamically
                query_text=query_text,
                log_df=log_df,
            )

            query_index_map = measure_retention(query_text, faiss_index=paul_g_index)

            query_text_retained = next(iter(query_index_map))
            generate_retained_chunks_graph(query_index_map=query_index_map, query_text=query_text_retained, append_suffix = f"simple_qa")

            
    for k_example in k_values:
        for query_text in simple_qa:
            query_to_required_prefixes = measure_rate(query_text=query_text, faiss_index=paul_g_index)
            print(query_to_required_prefixes)
            query_text_match= next(iter(query_to_required_prefixes))
            generate_match_rate(query_map = query_to_required_prefixes,
                                append_suffix = "simple_match_rate"
                                  )


            
