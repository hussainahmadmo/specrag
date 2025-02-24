
import os
import json
import argparse
import pandas as pd
import faiss
import numpy as np
import time
import logging
from utils import (
    load_pdf, plot_token_length_distribution, generate_retained_chunks_graph, 
    generate_match_rate, generate_match_rate_words, plot_typing_time, generate_end2end_graph,
    generate_bandwidth_delay_graph
)
from data.queries import queries_list
from pipeline import load_model_tokenizer, chunk_text, build_embeddings, get_query_indices, measure_retained_indexes, measure_rate, measure_typing_time
from pipeline import measure_bandwidth_delay
from collections import defaultdict
# ---------- Logging Setup ----------
logging.basicConfig(filename="pipeline.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- GLOBAL SETTINGS ----------
chunk_save_path = "./data/json_files/"

# Define different dataset configurations
datasets = {
    "paul_graham": {
        "pdf_path": "./data/pdf_files/paul-graham-essays.pdf",
        "document_name": "paul-graham-essay",
        "text_output_path": "./data/text_files/paul-graham-essays.txt",
        "json_output_path": "./data/json_files/paul-graham-essay/paul-graham-essay-csize200.json",
        "index_storage_path": "./index_store/paul-graham-essays.faiss",
    },
}

# Define different query types
query_types = {
    "simple": queries_list.simple_pg,
    "complex_20": queries_list.complex_pg_20,
    "complex_50": queries_list.complex_pg_50,
}

from utils import generate_mean_statistics
from pipeline import establish_baseline
# from pipeline import average_chunks
def run_pipeline(dataset_name, model_path, chunk_size, query_type):
    """
    Runs the complete pipeline for a given dataset and query type.

    Args:
        dataset_name (str): The dataset to process.
        model_path (str): Path to the model to be used.
        chunk_size (int): Size of text chunks.
        query_type (str): Type of query ('simple', 'complex_20', 'complex_50').
    """

    # Validate dataset and query type
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} is not defined. Choose from {list(datasets.keys())}")
    if query_type not in query_types:
        raise ValueError(f"Query type {query_type} is not defined. Choose from {list(query_types.keys())}")
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(model_path)
    # Load dataset configuration
    dataset = datasets[dataset_name]
    # Extract text from the dataset
    text = load_pdf(pdf_path=dataset["pdf_path"], output_txt_path=dataset["text_output_path"])
    # total length of tokens per chunk
    token_length_per_chunk = chunk_text(
        text=text,
        chunk_size=chunk_size,
        document_name=dataset["document_name"],
        chunk_save_path=chunk_save_path,
        tokenizer=tokenizer
    )
    # Plot the distribution of token lengths of chunks
    plot_token_length_distribution(token_length_per_chunk, append_suffix=dataset["document_name"])
    # Build embeddings
    index = build_embeddings(json_path=dataset["json_output_path"], index_path=dataset["index_storage_path"])
    # Run similarity queries
    query_set = query_types[query_type]
    
    with open("./data/json_files/paul-graham-essay/paul-graham-essay-csize200.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    # Create a mapping from chunk index to text
    id_to_text = {entry["chunk_index"]: entry["text"] for entry in data}
    # log_df = pd.DataFrame(columns=["k_example", "embed_time", "search_time", "query_text", "retrival_time"])
    for query_text in query_set:
        try:
            df = pd.read_csv("./csv_files/retrieved_chunks.csv")
        except FileNotFoundError:
            df = pd.DataFrame(columns=["query_text"])  # Initialize empty DataFrame
        establish_baseline(query_text, index, df, id_to_text, top_k=10)

    df = pd.read_csv("./csv_files/queries50_top10chunks.csv")
    # average_chunks(df)

    import sys
        
    for query_text in query_set:
        query_index_retrival_duration, query_index = get_query_indices(faiss_index=index, k_example=5, query_text=query_text)
        partial_query_retained_index_map = measure_retained_indexes(query_text=query_text, query_index=query_index, faiss_index=index)
        # Generate graph for retained indexes
        # generate_retained_chunks_graph(query_index_map=partial_query_retained_index_map, query_text=query_text, append_suffix=f"{dataset_name}_retained")
        query_to_required_prefixes = measure_rate(query_text=query_text, faiss_index=index, query_index=query_index)
        # generate_match_rate(query_map=query_to_required_prefixes, append_suffix=f"{query_type}_match_rate")
        # generate_match_rate_words(query_map=query_to_required_prefixes, append_suffix=f"{query_type}_match_rate_words")
        generate_end2end_graph(query_to_required_prefixes)

    all_queries_delay_map = defaultdict(list)


    for query_text in query_set:

        query_index_retrival_duration, query_index = get_query_indices(faiss_index=index, k_example=5, query_text=query_text)
        partial_query_retained_index_map = measure_retained_indexes(query_text=query_text, query_index=query_index, faiss_index=index)
            
        bandwidth_list = []
        percentage_to_chunk_map = defaultdict(list)
        truncated_query_length = None
        percentages_of_query = [0.25, 0.5, 0.75, 0.90]
        chunks_sizes = [5, 10, 15, 20]
        for percentage in percentages_of_query:
            for chunk_size in chunks_sizes:
                bandwidth_used, extra_delay, truncated_query_length = measure_bandwidth_delay(query_text,
                                                                index,
                                                                query_index,
                                                                no_of_chunks_to_retrieve=chunk_size,
                                                                percentage_of_query=percentage
                                                                )
                truncated_query_length = truncated_query_length
                percentage_to_chunk_map[percentage].append(extra_delay)
                bandwidth_list.append(bandwidth_used)
        #only use the first 4 elements
        # generate_bandwidth_delay_graph(bandwidth_list[:4], percentage_to_chunk_map, query_text)
        all_queries_delay_map[query_text].append(percentage_to_chunk_map)
        # mean_values_map = generate_mean_statistics(all_queries_delay_map)
        # from utils import generate_mean_graph
        # generate_mean_graph(bandwidth_list[:4], mean_values_map)

        from pipeline import compute_tsne
        from pipeline import plot_tsne

        q_len = len(query_text.split())
        prefix_map, data_emb = compute_tsne(json_path=dataset["json_output_path"], 
                         query_text=query_text)
        plot_tsne(prefix_map, data_embedding=data_emb, append_suffix=f"qrylen_{q_len}")



# This makes the script executable
if __name__ == "__main__":
    runs = [
        {"dataset": "paul_graham", "model_path": "meta-llama/Llama-3.1-8B-Instruct", "query_type": "complex_20", "chunk_size": 200},
    ]

    # Loop through different configurations and execute
    for run in runs:
        run_pipeline(dataset_name=run["dataset"],
                     model_path=run["model_path"],
                     query_type=run["query_type"],
                     chunk_size=run["chunk_size"]
                     )

