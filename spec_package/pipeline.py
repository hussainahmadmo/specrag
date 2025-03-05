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
    filename="baseline_retrival.log",filemode="a",level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s",
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import BaseNode
import json 
# ---------- GLOBAL VARIABLES -----------
from transformers import AutoModelForCausalLM, AutoTokenizer
config = ConfigManager.load_config("../configs/central.yaml") 
EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")    
device = "cuda:0"  # Change to your desired GPU
# GEN_MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct").to(device)
# tokenizer_gen_model = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")


def get_embedding(text : list):
    """
    List of sentences
    Return embedding (can be used for both query and sentences(text from document))
    """
    embeddings = EMB_MODEL.encode(text)
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
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
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
    with open("./data/json_files/paul-graham-essay/paul-graham-essay-csize200.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    # Create a mapping from chunk index to text
    id_to_text = {entry["chunk_index"]: entry["text"] for entry in data}

    for i in range(len(words)):
        partial_query = " ".join(words[: len(words) - i])
        partial_query_embedding = get_embedding(partial_query).reshape(1, -1)
        start_time = time.time()
        for j in range(0, 60000, 10):
            _ , prefix_index = faiss_index.search(partial_query_embedding, j+5)
            prefix_index = prefix_index.flatten().tolist()  # Convert to a standard list
            retrieved_chunks = [id_to_text[int(idx)] for idx in prefix_index if int(idx) in id_to_text]
            logging.info(f"\nPartial Query: '{partial_query}'")
            logging.info("Length of Retrieved Chunks:" + len(retrieved_chunks))
            for idx, chunk in zip(prefix_index, retrieved_chunks):
                logging.info(f"- Chunk {idx}: {chunk[:200]}...")  # Limit output to first 200 chars
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
    with open("./data/json_files/paul-graham-essay/paul-graham-essay-csize200.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    # Create a mapping from chunk index to text
    id_to_text = {entry["chunk_index"]: entry["text"] for entry in data}

    for i in range(len(words)):
        partial_query = " ".join(words[: len(words) - i])
        partial_query_embedding = get_embedding(partial_query).reshape(1, -1)
        start_time = time.time()
        for j in range(0, 60000, 10):
            _ , prefix_index = faiss_index.search(partial_query_embedding, j+5)
            prefix_index = prefix_index.flatten().tolist()  # Convert to a standard list
            retrieved_chunks = [id_to_text[int(idx)] for idx in prefix_index if int(idx) in id_to_text]
            # logging.info(f"\nPartial Query: '{partial_query}'")
            # # logging.info("Length of Retrieved Chunks:" + len(retrieved_chunks))
            # for idx, chunk in zip(prefix_index, retrieved_chunks):
            #     logging.info(f"- Chunk {idx}: {chunk[:200]}...")  # Limit output to first 200 chars
            if np.all(np.isin(query_index, prefix_index)):
                prefix_query_to_retrieved_indexes[partial_query] = [j+5]
                retrieval_time = time.time() - start_time
                prefix_query_to_retrieved_indexes.setdefault(partial_query, []).append(retrieval_time)
                break

    return prefix_query_to_retrieved_indexes

def establish_baseline(query_text, faiss_index, df, id_to_text, top_k):
    """
    Return:
        prefix_query_to_retrieved_indexes : dict
            A dictionary where:
            - Keys are partial queries.
            - Values are lists containing:
                - The number of retrieved indexes.
                - The retrieval time.
        Args:
            - top_k - the number of K similar elements 
                        to retrieve
    """
    query_embedding = get_embedding([query_text])
    _ , prefix_index = faiss_index.search(query_embedding, top_k)
    prefix_index = prefix_index.flatten().tolist()  # Convert to a standard list
    retrieved_chunks = [id_to_text[int(idx)] for idx in prefix_index if int(idx) in id_to_text]

    retrieval_results = {"query_text": query_text}

    for i in range(1, top_k + 1):
        retrieval_results[f"retrieved_chunk_{i}"] = retrieved_chunks[i - 1] if i <= len(retrieved_chunks) else None

    df = pd.concat([df, pd.DataFrame([retrieval_results])], ignore_index=True)
    df.to_csv("./csv_files/retrieved_chunks.csv", index=False)
    
    return df

def measure_bandwidth_delay(query_text, 
                      faiss_index, 
                      query_index,
                      no_of_chunks_to_retrieve,
                      percentage_of_query,
                      ):
    """
    
    Return bandwidth used to fetch m chunks, and the 
    delay incurred for fetching 

    Args - 
        query_text : full query
        faiss_index : index to search for
        query_index : the original query_index 

    Returns 
    """
    truncated_query_length = int(len(query_text.split()) * percentage_of_query)
    truncated_query= query_text.split()[:truncated_query_length]
    truncated_query= " ".join(truncated_query)  # Convert back to string


    _, truncated_query_index = get_query_indices(faiss_index=faiss_index, 
                                            k_example=no_of_chunks_to_retrieve, 
                                            query_text=truncated_query)
    
    extra_indexes = len(np.setdiff1d(truncated_query_index.flatten(), 
                                      query_index.flatten()))
    shared_index_count = len(np.intersect1d(query_index,truncated_query_index))
    #assuming KV cache size is 20 MB for a 
    extra_delay = ((5-shared_index_count) * 20) / 125
    #calculate the delay using the found indexes
    bandwidth_used = no_of_chunks_to_retrieve * 20

    return bandwidth_used, extra_delay, truncated_query_length



from sklearn.manifold import TSNE as SklearnTSNE
from cuml.manifold import TSNE as CuMLTSNE

# from sklearn.decomposition import PCA
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
import cupy as cp  # GPU-based NumPy alternative
from sklearn.preprocessing import normalize


from sklearn.preprocessing import normalize
import numpy as np
import os
import json
from sklearn.manifold import TSNE as SklearnTSNE

def compute_tsne(json_path, id_query_map, perplexity=50):
    """
    Compute or load embeddings using t-SNE.
    
    Args:
        json_path (str): Path to the JSON file containing text data.
        id_query_map (dict): A dictionary mapping unique query IDs (str) to their corresponding queries (str).
        perplexity (int): Perplexity parameter for t-SNE.

    Returns:
        prefix_2d_emb_map (dict): A dictionary mapping query IDs to their text and a dictionary of prefix embeddings.
        embeddings_2d_copy[:-1]: All embeddings except the last row (which is the last query's tsne-2d representation).
    """
    npz_file_path = "./data/computed_emb/prefix_embedding.npz"
    
    # Check if .npz file exists and load cached embeddings
    if os.path.exists(npz_file_path):
        data = np.load(npz_file_path, allow_pickle=True)
        prefix_2d_emb_map = data["prefix_2d_emb_map"].item()
        embeddings_2d_copy = data["embeddings_2d_copy"]
        return prefix_2d_emb_map, embeddings_2d_copy

    # Load text data from JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [chunk["text"] for chunk in data]
    data_embeddings = np.array(get_embedding(texts))
    prefix_2d_emb_map = {}
    embeddings_2d_copy = None

    for query_id, query_text in id_query_map.items():
        words = query_text.split()
        query_prefixes = [" ".join(words[: len(words) - i]) for i in range(len(words)) if words[: len(words) - i]]
        
        # Initialize structure for storing prefixes
        prefix_2d_emb_map[query_id] = {
            "query_text": query_text,
            "prefixes": {}
        }

        for prefix in query_prefixes:
            query_embedding = np.array(get_embedding(prefix)).reshape(1, -1)  # Ensure it's 2D
            # Stack normalized embeddings
            all_embeddings = np.vstack([data_embeddings, query_embedding])
            # Compute t-SNE
            embeddings_2d_slow = SklearnTSNE(n_components=2, perplexity=perplexity, init="random", random_state=42, n_jobs=-1).fit_transform(all_embeddings)
            query_2d = embeddings_2d_slow[-1]
            embeddings_2d_copy = embeddings_2d_slow
            # Store prefix 2D embedding
            prefix_2d_emb_map[query_id]["prefixes"][prefix] = query_2d

    # Save results in a compressed file
    np.savez_compressed(npz_file_path, prefix_2d_emb_map=prefix_2d_emb_map, embeddings_2d_copy=embeddings_2d_copy)
    return prefix_2d_emb_map, embeddings_2d_copy[:-1]


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def plot_tsne_per_query(prefix_2d_emb_map, data_embedding, save_dir="saved_graphs/tsne_graphs/"):
    """Plot t-SNE embeddings for each full query separately, showing prefix length variations."""

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over each full query and plot its prefixes separately
    for query_id, prefix_data in prefix_2d_emb_map.items():
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # Plot dataset embeddings in gray
        ax.scatter(data_embedding[:, 0], data_embedding[:, 1], 
                   c='gray', alpha=0.3, s=20, label="Dataset Embeddings")

        # Get unique prefix lengths
        prefix_lengths = sorted(set(len(prefix.split()) for prefix in prefix_data["prefixes"]))

        # Generate colors for prefix length shading
        cmap = sns.color_palette("Blues", len(prefix_lengths))  # Shades of blue for prefix length

        # Create a mapping from prefix length to color
        prefix_length_to_color = {length: cmap[i] for i, length in enumerate(prefix_lengths)}

        # Dictionary to store legend handles
        legend_handles = {}

        # Plot each prefix for this query
        for prefix, prefix_embedding in prefix_data["prefixes"].items():
            prefix_embedding = np.array(prefix_embedding).reshape(-1, 2)
            prefix_length = len(prefix.split())  # Count words in the prefix
            color = prefix_length_to_color[prefix_length]

            scatter = ax.scatter(prefix_embedding[:, 0], prefix_embedding[:, 1], 
                                 c=[color], s=100, edgecolors='black')

            # Store one entry per prefix length for the legend
            if prefix_length not in legend_handles:
                legend_handles[prefix_length] = mpatches.Patch(color=color, label=f"Prefix Length {prefix_length}")

        # Labels and title
        ax.set_title(f"t-SNE Projection for Query: {query_id}", fontsize=14)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

        # Manually create the legend
        legend_labels = [f"Prefix Length {length}" for length in legend_handles.keys()]
        if legend_handles:  # Only add legend if there are prefixes
            ax.legend(handles=legend_handles.values(), 
                      loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10, frameon=False, title="Prefix Lengths")

        # Save figure for this query
        save_path = os.path.join(save_dir, f"tsne_query_{query_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot for Query {query_id} at: {save_path}")

        plt.close(fig)  # Close figure to avoid memory issues

def prefix_index_map(query_map, faiss_index: faiss.Index):
    """Return the top 5 similar indexes retreived for a prefix."""
    prefix_index_map = {}
    for query_id, query_text in query_map.items():
            words = query_text.split()
            query_prefixes = [" ".join(words[: len(words) - i]) for i in range(len(words)) if words[: len(words) - i]]
            # Initialize structure for storing prefixes
            prefix_index_map[query_id] = {
                "query_text": query_text,
                "prefixes": {}
            }

            for prefix in query_prefixes:
                prefix_embedding = np.array(get_embedding([prefix]), dtype=np.float32)
                _ , prefix_index = faiss_index.search(prefix_embedding, 5)
                prefix_index_map[query_id]["prefixes"][prefix] = prefix_index

    return prefix_index_map

def follows_strict_order(list1, list2):
    """
    Returns stru

    Args -  list1 - a list type 
            list2 - a list type
    """
    if not isinstance(list2, list):  # Ensure list2 is a list
        raise TypeError("list2 must be a list")
    if not isinstance(list1, list):
        raise TypeError("list1 must be a list")
    
    matched_count = 0
    for i, val in enumerate(list1):
        if val == list2[i]:
            matched_count = matched_count + 1

    if matched_count <= 0:
        return (False, 0)
    
    return (True, matched_count)

import copy
def prev_prefix_match_ordered(prefix_index_map):
    """
    Return the prefix index map, with the common prefix count between the current prefix and the other prefix.
    The ordering follows a strict ordering.
    """
    new_prefix_index_map = copy.deepcopy(prefix_index_map)

    for query_id, query_text in new_prefix_index_map.items():
        next_indices = None
        next_prefix = None
        for prefix, indices in reversed(query_text["prefixes"].items()):
            current_prefix = prefix
            current_indices = indices.flatten().tolist()
            if next_indices is not None and isinstance(next_indices, np.ndarray):
                next_indices = next_indices.flatten().tolist()
                current_indices = current_indices.flatten().tolist()
                res = follows_strict_order(current_indices, next_indices)
                new_prefix_index_map[query_id]["prefixes"][current_prefix] = {"index_count" : 0 }
                new_prefix_index_map[query_id]["prefixes"][current_prefix]["index_count"] = res[1]
            if next_indices is not None and isinstance(next_indices, list):
                res = follows_strict_order(current_indices, next_indices)
                #TODO - now update count for next index and its prefix
                new_prefix_index_map[query_id]["prefixes"][current_prefix] = np.append(new_prefix_index_map[query_id]["prefixes"][current_prefix], res[1])
                
            # Update previous values
            next_indices = current_indices
            next_prefix = current_prefix
    
    return new_prefix_index_map

def prev_prefix_match_intersect(prefix_index_map):
    """
    Return the prefix index map, with the intersection(common indexes bewtween the
    current prefix and the previous prefix, and the common indexes between the 
    current prefix and the full query prefix.


    Returns - query_index_map
        In the map, each query has a key. 
        
        The second last element of the list is the count of set of 
        indexes similar between the previous prefix, and the current prefix.
        The last element in the list is the count of set of indexes similar 
        between the current prefix.
        
        The last element of the list is the count of set of indexes similar 
        between the full query, and the current prefix.

    """
    new_prefix_index_map = copy.deepcopy(prefix_index_map)

    for query_id, query_text in new_prefix_index_map.items():
        next_indices = None
        next_prefix = None
        last_query_text = next(iter(new_prefix_index_map[query_id]["prefixes"]))
        last_index = new_prefix_index_map[query_id]["prefixes"][last_query_text]
        last_indices = last_index.flatten().tolist()


        for prefix, indices in reversed(query_text["prefixes"].items()):
            current_prefix = prefix
            current_indices = indices.flatten().tolist()
            if next_indices is not None and isinstance(next_indices, list):
                count = len(list(set(current_indices) & set(next_indices)))
                final_index_intersection_count = len(list(set(current_indices) & set(last_indices)))
                new_prefix_index_map[query_id]["prefixes"][current_prefix] = np.append(new_prefix_index_map[query_id]["prefixes"][current_prefix], count)
                new_prefix_index_map[query_id]["prefixes"][current_prefix] = np.append(new_prefix_index_map[query_id]["prefixes"][current_prefix], final_index_intersection_count)

            next_indices = current_indices
            next_prefix = current_prefix

    return new_prefix_index_map


def total_bandwidth(prefix_index_map):
    """
    Return the total bandwidth used for each query with query ID.
    """

    new_prefix_index_map = copy.deepcopy(prefix_index_map)

    query_id_to_total_bandwidth = {}
    for query_id, querytext in new_prefix_index_map.items():

        query_id_to_total_bandwidth[query_id] = []
        prev_indices = None
        last_index = None
        next_indices_list = []
        total_bandwidth = []
        consecutive = False
        for prefix, indices in reversed(querytext["prefixes"].items()):
            current_prefix = prefix
            current_indices = indices.flatten().tolist()

            if prev_indices is not None:
                delta_change = prev_indices[-2] - current_indices[-2]

                if delta_change == 0:
                    consecutive = True
                if (delta_change < 0 or delta_change > 0) and consecutive == True:
                    query_id_to_total_bandwidth[query_id].append(prev_indices[-2] * 20)
                    consecutive = False

            last_prev_index = prev_indices
            prev_indices = current_indices

        delta_change = last_prev_index[-1] - prev_indices[-1]

        if delta_change == 0:
            query_id_to_total_bandwidth[query_id].append(last_prev_index[-1] * 20)
        query_id_to_total_bandwidth[query_id].append(last_prev_index[-1])
        
    return query_id_to_total_bandwidth



                




        

        

        

# def plot_index_match_graph(prefix_index_map):

#     for queries_text in prefix_index_map:

#         for 


            

#TODO
# def average_chunks(df):
#     """
#     Return average no of chunks required to answer the queries .
#     Args:
#         dataframe : dataframe containing the columns query_text 
#         and the retrieved top 10 chunks.
#     """

#     # Process each row into a separate formatted string
#     query_context_texts = df.apply(
#         lambda row: " | ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notna(row[col])]), 
#         axis=1
#     ).tolist()
#     query_text = df.iloc[0,0]
#     chat1 = [
#         {
#             "role": "system",
#             "content": "You are an AI assistant that selects the minimal number of text chunks required to accurately answer a given query. \
#                         Return only the relevant chunks from the provided list, ensuring that they fully support the answer to the query.",
#         },
#         {"role": "user", "content": f"Query: {query_text}\n\nChunks:\n {query_context_texts[0]}"}
#     ]
#     chat1.append({"role": "user", "content": "Return the relevant chunks as a JSON list."})
#     tokenized_chat = tokenizer_gen_model.apply_chat_template(chat1, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
#     outputs = GEN_MODEL.generate(tokenized_chat, max_new_tokens=2000)
#     decoded_output = tokenizer_gen_model.decode(outputs[0])

#     print(decoded_output)






        












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

