from transformers import AutoTokenizer, AutoConfig,AutoModel
import torch
import datetime
import logging
import re

# Map torch.dtype to its byte size
dtype_to_size = {
    torch.float32: 4,  # FP32
    torch.float16: 2,  # FP16
    torch.bfloat16: 2,  # BF16
    torch.int8: 1,      # INT8
}

def compute_kv(sequence_length, model_name : str) -> str:
    """
    Compute kv size which is equal to 
        memory = 2 * L * H * N * D
    L(sequence length)
    H(Hidden Dimension)
    N(Number of Layers)
    D(Precision used for cache)
    Returns - dictionary with kv_size in byte, mb, gb
    """
    config = AutoConfig.from_pretrained(model_name)
    # Load the model
    model = AutoModel.from_pretrained(model_name)
    dtype = next(model.parameters()).dtype
    size = dtype_to_size[dtype]
    kv_size_in_bytes = 2 * sequence_length * config.hidden_size*config.num_attention_heads*size
    kv_size_in_mb = kv_size_in_bytes / (1024 ** 2)
    kv_size_in_gb = kv_size_in_bytes / (1024 ** 3)  
    # Return the sizes as a dictionary
    return {
        "kv_size_bytes": kv_size_in_bytes,
        "kv_size_mb": kv_size_in_mb,
        "kv_size_gb": kv_size_in_gb
    }

def measure_inference(llm_instance, 
                        context: bool, 
                        prompt: str,
                        context_text : str = None,
                        position: int = None):
    """
    Measures the inference time for invoking the language model, optionally inserting user-specified context
    into the prompt at a specific position.

    Args:
        context (str): The user-specified context to add to the prompt.
        position (int): The index at which to insert the context in the prompt.
        prompt (str): The input prompt for the language model.

    Returns:
        result: The output of the language model invocation.
        duration: Time taken for inference.
    """
    if context:
        if position < 0 or position > len(prompt.split()):
            raise ValueError("Position is out of range for the prompt length.")
        
        prompt_list = prompt.split()
        prompt_list.insert(position, context_text)
        prompt = " ".join(prompt_list)

    logging.info("Prompt : " + str(prompt) )
    start_time = datetime.datetime.now()
    # Invoke the LLM
    result = llm_instance.invoke(prompt)
    # Print the result
    logging.info("Result : " + str(result.content))
    # Measure and log inference time
    end_time = datetime.datetime.now()
    inference_time = (end_time - start_time).total_seconds()
    logging.info(f"Inference time: {inference_time:.6f} seconds")


from bs4 import BeautifulSoup

# Custom parser to extract text from HTML using BeautifulSoup
class HTMLTextExtractor:
    def __call__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(strip=True)


import pypdf
from llama_index.core import Document
import os
def load_pdf(pdf_path, output_txt_path):
    """Extract text from a PDF file and return as a single string."""
    if os.path.exists(output_txt_path):
        with open(output_txt_path, "r", encoding="utf-8") as file:
            text = file.read()
            return text
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found in {pdf_path}")
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)
    return text

import matplotlib.pyplot as plt 
def generate_retained_chunks_graph(query_index_map : dict, query_text, append_suffix : str):
    # Create a relative directory to store the graph
    save_dir = "saved_graphs/chunk_graphs"  # Folder where graphs will be saved
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    exp_dir = f"saved_graphs/chunk_graphs/{append_suffix}"
    os.makedirs(exp_dir, exist_ok=True)

    base_file_name = f"{append_suffix}"
    counter = 1

    while os.path.exists(os.path.join(exp_dir, f"{base_file_name}_{counter}.png")):
        counter += 1
    save_path = os.path.join(exp_dir, f"{base_file_name}_{counter}.png")

    queries =list(query_index_map.keys())
    num_retained_indexes = [v for v in query_index_map.values()]

    plt.figure(figsize=(16,8))
    plt.plot(queries, num_retained_indexes, marker="o", color="orange", linestyle="-", markersize=6)

    # Labels and Title
    plt.xlabel("Queries", fontsize=4)
    plt.ylabel("Number of Retained Indexes", fontsize=12)
    plt.title("Queries vs. Retained Index Count", fontsize=14)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Query vs retained graph saved at : {save_path}")

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import textwrap

def generate_match_rate(query_map: dict, append_suffix: str, wrap_width=50):
    """
    Generates a match rate graph with improved Y-axis text fitting.

    - Ensures Y-axis text fits properly without overlap.
    - Increases figure height to prevent squeezing.
    - Uses text wrapping for long queries.
    """


    # Create directories
    save_dir = "saved_graphs/chunk_graphs"
    exp_dir = os.path.join(save_dir, append_suffix)
    os.makedirs(exp_dir, exist_ok=True)

    # Generate unique filename
    base_file_name = append_suffix
    counter = 1
    while os.path.exists(os.path.join(exp_dir, f"{base_file_name}_{counter}.png")):
        counter += 1
    save_path = os.path.join(exp_dir, f"{base_file_name}_{counter}.png")

    orig_query_length = len(next(iter(query_map)))

    # Convert data to NumPy arrays for speed
    queries = list(query_map.keys())[::-1]
    values = list(query_map.values())[::-1]

    fraction_query = np.array([
        round((len(query) / orig_query_length) * 100, 0)
        for query in queries])

    # Wrap queries for better Y-axis fitting
    wrapped_queries = ["\n".join(textwrap.wrap(q, width=wrap_width)) for q in queries]

    # Increase figure size for better spacing
    plt.figure(figsize=(18, 15))  

    # Plot valid points efficientl
    plt.plot(fraction_query , values,  marker="o", color="orange", linestyle="-",
                 markersize=8, linewidth=3)

    # for x, y in zip(values, fraction_query):
    #         plt.text(x, y, f"{x}", fontsize=10, ha="right", va="bottom", color="black")

    plt.xlim(0, 101)
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, 10), fontsize=12, fontweight="bold")  # 10 intervals at [0, 10, 20, ..., 100]

    # Adjust labels with padding to avoid overlap
    plt.ylabel("Top-5 Similar Chunks Retrieved in Top-K Chunks", fontsize=14, fontweight="bold", labelpad=10)
    plt.xlabel("Prefix Length as % of Full Query", fontsize=14, fontweight="bold", labelpad=12)
    plt.title("Match Rate Graph", fontsize=16, fontweight="bold")

    # Create more space on the left for queries
    plt.subplots_adjust(left=0.3)
    # Thicker grid lines
    plt.grid(True, linestyle="--", alpha=0.7, linewidth=1.5)
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"FAISS retrieval evaluation saved: {save_path}")

def generate_match_rate_words(query_map: dict, append_suffix: str, wrap_width=50):
    """
    Generates a match rate graph with improved Y-axis text fitting.

    - Ensures Y-axis text fits properly without overlap.
    - Increases figure height to prevent squeezing.
    - Uses text wrapping for long queries.
    """
    # Create directories
    save_dir = "saved_graphs/chunk_graphs"
    exp_dir = os.path.join(save_dir, append_suffix)
    os.makedirs(exp_dir, exist_ok=True)
    # Generate unique filename
    base_file_name = append_suffix
    counter = 1
    while os.path.exists(os.path.join(exp_dir, f"{base_file_name}_{counter}.png")):
        counter += 1
    save_path = os.path.join(exp_dir, f"{base_file_name}_{counter}.png")
    
    orig_query = next(iter(query_map))
    # Convert data to NumPy arrays for speed
    words_left_behind = [len(orig_query.split()) - len(query.split()) for query in query_map.keys()]
    num_retrieved_chunks = list(query_map.values())

    assert len(words_left_behind) == len(num_retrieved_chunks)

    # Increase figure size for better spacing
    plt.figure(figsize=(18, 15))  

    # Plot valid points efficientl
    plt.plot(words_left_behind, num_retrieved_chunks,  marker="o", color="orange", linestyle="-",
                 markersize=8, linewidth=3)

    # for x, y in zip(values, fraction_query):
    #         plt.text(x, y, f"{x}", fontsize=10, ha="right", va="bottom", color="black")
    # 
    plt.xlim(0, len(orig_query.split()) + 1)  # Limit X-axis to full query length
    plt.ylim(0, 30)
    full_query_length = len(orig_query.split())

    # Add a dotted vertical line at the full query length
    plt.axvline(x=full_query_length, color="red", linestyle="dashed", linewidth=2)
    # Adjust labels with padding to avoid overlap
    plt.ylabel("Top-5 Similar Chunks Retrieved in Top-K Chunks", fontsize=14, fontweight="bold", labelpad=10)
    plt.xlabel("# of words left behind", fontsize=14, fontweight="bold", labelpad=12)
    plt.title("Match Rate Graph", fontsize=16, fontweight="bold")

    # Create more space on the left for queries
    plt.subplots_adjust(left=0.3)
    # Thicker grid lines
    plt.grid(True, linestyle="--", alpha=0.7, linewidth=1.5)
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logging.info(f"FAISS retrieval evaluation saved: {save_path}")

import numpy as np
import matplotlib.pyplot as plt

def analyze_query_distribution(query_map):
    """
    Analyzes and visualizes the distribution of queries with prefix length > 50%
    that require at most the top 15 chunks.
    """

    orig_query_length = len(next(iter(query_map)))
    queries = list(query_map.keys())[::-1]  # Reverse for better visualization
    values = np.array(list(query_map.values())[::-1])  # Chunks retrieved (Y-axis)
    
    fraction_query = np.array([
        round((len(query) / orig_query_length) * 100, 0)
        for query in queries
    ])

    high_prefix_mask = fraction_query > 50
    high_prefix_queries = fraction_query[high_prefix_mask]
    high_prefix_values = values[high_prefix_mask]

    top_15_mask = high_prefix_values <= 15
    count_top_15 = np.sum(top_15_mask)  # Total queries satisfying the condition
    total_high_prefix = len(high_prefix_queries)  # Total queries with >50% prefix

    if total_high_prefix > 0:
        percentage_top_15 = (count_top_15 / total_high_prefix) * 100
    else:
        percentage_top_15 = 0

    print(f"Total queries with >50% prefix: {total_high_prefix}")
    print(f"Queries where top-15 chunks were sufficient: {count_top_15}")
    print(f"Percentage of high-prefix queries needing ≤15 chunks: {percentage_top_15:.2f}%")

    # **Group by prefix length and calculate percentages**
    unique_prefixes = np.unique(high_prefix_queries)
    prefix_percentages = []

    for prefix in unique_prefixes:
        prefix_mask = high_prefix_queries == prefix
        prefix_total = np.sum(prefix_mask)
        prefix_top_15 = np.sum(high_prefix_values[prefix_mask] <= 15)
        prefix_percentage = (prefix_top_15 / prefix_total) * 100 if prefix_total > 0 else 0
        prefix_percentages.append(prefix_percentage)

    # **Plot distribution**
    plt.figure(figsize=(10, 6))
    plt.bar(unique_prefixes, prefix_percentages, color="blue", alpha=0.7, edgecolor="black")
    
    plt.xlabel("Prefix Length as % of Query")
    plt.ylabel("Percentage of Queries Needing ≤ 15 Chunks")
    plt.title("Percentage of Queries Needing ≤ 15 Chunks for Prefixes > 50%")
    plt.xticks(unique_prefixes, rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    

def plot_token_length_distribution(token_length_list, append_suffix):
    """
    
    """

    # Create directories
    save_dir = "saved_graphs/distribution"
    exp_dir = os.path.join(save_dir, append_suffix)
    os.makedirs(exp_dir, exist_ok=True)
    # Generate unique filename
    base_file_name = append_suffix
    counter = 1
    while os.path.exists(os.path.join(exp_dir, f"{base_file_name}_{counter}.png")):
        counter += 1
    save_path = os.path.join(exp_dir, f"{base_file_name}_{counter}.png")


    # Compute statistics
    min_length = min(token_length_list)
    max_length = max(token_length_list)
    mean_length = np.mean(token_length_list)
    median_length = np.median(token_length_list)


    # Plot histogram of token length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(token_length_list, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel("Token Lengths")
    plt.ylabel("Frequency")
    plt.title("Distribution of Chunk Token Lengths")
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches = "tight")


import os
import matplotlib.pyplot as plt

def plot_typing_time(prefix_time_dictionary, append_suffix):
    """
    Plots a line graph for typing time against words left behind.
    
    Args:
        prefix_time_dictionary (dict): Dictionary with query prefixes as keys and typing times as values.
        append_suffix (str): Suffix for naming the saved graph.
    
    Saves the plot as a PNG file in the 'saved_graphs/typing_time' directory.
    """
    # Define save directory
    save_dir = "saved_graphs/typing_time"
    exp_dir = os.path.join(save_dir, append_suffix)
    os.makedirs(exp_dir, exist_ok=True)

    # Generate unique filename
    base_file_name = append_suffix
    counter = 1
    while os.path.exists(os.path.join(exp_dir, f"{base_file_name}_{counter}.png")):
        counter += 1
    save_path = os.path.join(exp_dir, f"{base_file_name}_{counter}.png")

    # Extract data for plotting
    full_query = list(prefix_time_dictionary.keys())[-1]
    words_list = [len(full_query.split()) - len(query.split()) for query in prefix_time_dictionary.keys()][::-1]
    time_taken_for_key = list(prefix_time_dictionary.values())  # Reverse to match words_list

    # Plot the data
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(words_list, time_taken_for_key, marker="o", color="orange", linestyle="-",
             markersize=8, linewidth=3)

    # Label axes and title
    plt.ylabel("Time elapsed (seconds)", fontsize=14, fontweight="bold", labelpad=10)
    plt.xlabel("Words left behind", fontsize=14, fontweight="bold", labelpad=12)
    plt.title("Typing Time vs. Words Left Behind", fontsize=16, fontweight="bold")

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")