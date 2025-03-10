
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
        "pdf_paths": 
            ["./data/pdf_files/paul-graham/paul-graham-essays.pdf"],
        "document_name": "paul-graham-essay",
        "text_output_path": "./data/text_files/paul-graham-essays.txt",
        "json_output_path": "./data/json_files/paul-graham-essay/paul-graham-essay-csize200.json",
        "index_storage_path": "./index_store/paul-graham-essays.faiss",
        # "query_path": "./data/queries/sec-10/questions.csv"

    },
    "sec-10": {
        "pdf_paths": [
            "./data/pdf_files/sec-10/2022 Q3 AAPL.pdf",
            "./data/pdf_files/sec-10/2022 Q3 AMZN.pdf",
            "./data/pdf_files/sec-10/2022 Q3 INTC.pdf",
            "./data/pdf_files/sec-10/2022 Q3 MSFT.pdf",
            "./data/pdf_files/sec-10/2022 Q3 NVDA.pdf",
            "./data/pdf_files/sec-10/2023 Q1 AAPL.pdf",
            "./data/pdf_files/sec-10/2023 Q1 AMZN.pdf",
            "./data/pdf_files/sec-10/2023 Q1 INTC.pdf",
            "./data/pdf_files/sec-10/2023 Q1 MSFT.pdf",
            "./data/pdf_files/sec-10/2023 Q1 NVDA.pdf",
            "./data/pdf_files/sec-10/2023 Q2 AAPL.pdf",
            "./data/pdf_files/sec-10/2023 Q2 AMZN.pdf",
            "./data/pdf_files/sec-10/2023 Q2 INTC.pdf",
            "./data/pdf_files/sec-10/2023 Q2 MSFT.pdf",
            "./data/pdf_files/sec-10/2023 Q2 NVDA.pdf",
            "./data/pdf_files/sec-10/2023 Q3 AAPL.pdf",
            "./data/pdf_files/sec-10/2023 Q3 AMZN.pdf",
            "./data/pdf_files/sec-10/2023 Q3 INTC.pdf",
            "./data/pdf_files/sec-10/2023 Q3 MSFT.pdf",
            "./data/pdf_files/sec-10/2023 Q3 NVDA.pdf",
        ],
        "document_name": "sec-10", 
        "text_output_path": "./data/text_files/sec-10/combined.txt",
        "json_output_path": "./data/json_files/sec-10/sec-10-csize200.json",
        "index_storage_path": "./index_store/sec-10.faiss",
        "query_path": "./data/queries/sec-10/qna_data.csv"
    }
}


from utils import generate_mean_statistics
from pipeline import establish_baseline
from utils import filter_questions
# from pipeline import average_chunks
def run_pipeline(dataset_name, 
                model_path, 
                chunk_size):
    """
    Runs the complete pipeline for a given dataset and query type.

    Args:
        dataset_name (str): The dataset to process.
        model_path (str): Path to the model to be used.
        chunk_size (int): Size of text chunks.
    """
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(model_path)
    # Load dataset configuration
    dataset = datasets[dataset_name]
    # Extract text from the dataset
    text = load_pdf(pdf_paths=dataset["pdf_paths"], 
                    output_txt_path=dataset["text_output_path"])
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
    index = build_embeddings(json_path=dataset["json_output_path"], 
                             index_path=dataset["index_storage_path"])
    
    queries_list = None
    if dataset["query_path"].endswith(".csv"):
        query_path = dataset['query_path']
        queries_list = filter_questions(query_path=query_path,
                                        rag_type=None
                                               )

    from pipeline import compute_tsne
    # from pipeline import plot_tsne
    from pipeline import plot_tsne_per_query
    pg_query_map = {f"Q{i+1}": query for i, query in enumerate(queries_list)}
    # prefix_map, data_emb = compute_tsne(json_path=dataset["json_output_path"], 
    #                         id_query_map=pg_query_map)
    # plot_tsne_per_query(prefix_map, data_embedding=data_emb)
    from pipeline import prefix_index_map
    prefix_index_map = prefix_index_map(pg_query_map, faiss_index=index)
    from pipeline import prev_prefix_match_ordered
    # prev_prefix_index_map = prev_prefix_match_ordered(prefix_index_map)
    # from utils import plot_prefix_match_ordered
    # plot_prefix_match_ordered(prev_prefix_index_map, dataset["document_name"])
    from pipeline import prev_prefix_match_intersect
    prev_prefix_index_map_intersection = prev_prefix_match_intersect(prefix_index_map)

    from utils import plot_prefix_match_intersect
    # plot_prefix_match_intersect(prev_prefix_index_map_intersection, document_name=dataset["document_name"])
    from pipeline import bandwidth_consecutive
    query_id_bandwidth_use_final = bandwidth_consecutive(prev_prefix_index_map_intersection)

    from utils import plot_total_bandwidth
    plot_total_bandwidth(query_id_bandwidth_use_final)
    from utils import plot_total_bandwidth_heatmap
    plot_total_bandwidth_heatmap(query_id_bandwidth_use_final)

    from utils import bandwidth_mean
    mean_bandwidth, mean_delay = bandwidth_mean(query_id_bandwidth_use_final, document_name=dataset["document_name"])

    query_id_total_bwidth = bandwidth_consecutive(prev_prefix_index_map_intersection)
    
    from pipeline import bandwidth_consecutive_k_words

    k_values = [5, 10, 15, 20, 25]
    from pipeline import plot_bandwidth_k

    word_map = {}
    for k in k_values: 
        qid_bwidth = bandwidth_consecutive_k_words(prefix_index_map=prev_prefix_index_map_intersection, k=k)
        mean_bandwidth, mean_delay = bandwidth_mean(qid_bwidth, dataset["document_name"])
        word_map[k] = [mean_bandwidth, mean_delay]

    consec_mean_bwidth, consec_mean_delay = bandwidth_mean(query_id_total_bwidth, dataset["document_name"])
    word_map["consecutive"] = [consec_mean_bwidth, consec_mean_delay]
    query_id_total_bwidth = bandwidth_consecutive(prev_prefix_index_map_intersection)



    from computeIndexRetriever import AdaptiveStabilityRetriever

    for window_size in np.arange(1, 10, 2):  # Example range [1, 5] with step 0.2
        for penalty_factor in np.arange(0.1, 1, 0.2):
            for decay_rate in np.arange(0.1, 1, 0.2):
                for stability_threshold in np.arange(0.1, 1, 0.2):
                    asr = AdaptiveStabilityRetriever(window_size=window_size,
                                                     penalty_factor=penalty_factor,
                                                     decay_rate=decay_rate,
                                                     stability_threshold=stability_threshold
                                                     )
                    index_map, index = asr.should_retrieve(prev_prefix_index_map_intersection)

                    from utils import data_transfer_mean
                    mean_data_transfer, mean_delay = data_transfer_mean(index_map, dataset["document_name"])
                    word_map[f"WS:{window_size}-PF:{penalty_factor}-DR:{decay_rate}-ST:{stability_threshold}"] = [mean_data_transfer, mean_delay]


    from utils import plot_mean_bandwidth_delay
    plot_mean_bandwidth_delay(word_map, dataset["document_name"])
    




    

# This makes the script executable
if __name__ == "__main__":
    runs = [
        # {"dataset": "paul_graham", "model_path": "meta-llama/Llama-3.1-8B-Instruct", "query_type": "complex_20", "chunk_size": 200},
        {"dataset": "sec-10", "model_path": "meta-llama/Llama-3.1-8B-Instruct", "query_type" : "complex_20", "chunk_size" : 200}
        # {"dataset": "wikimqa", "model_path": "meta-llama/Llama-3.1-8B-Instruct", "query_type" : "complex_20", "chunk_size" : 200}

    ]

    # Loop through different configurations and execute
    for run in runs:
        run_pipeline(dataset_name=run["dataset"],
                     model_path=run["model_path"],
                     chunk_size=run["chunk_size"],
                     )

