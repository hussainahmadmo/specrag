
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
from pipeline import compute_prefix_index_map

# ---------- Logging Setup ----------
logging.basicConfig(filename="pipeline.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

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
    },
        "wikiqma": {
        "document_name": "wikiqma", 
        "context_path" : "./data/context_files/wikiqma.csv",
        "text_output_path": "./data/text_files/wikiqma/combined.txt",
        "json_output_path": "./data/json_files/wikiqma/wikiqma-csize200.json",
        "index_storage_path": "./index_store/wikiqma.faiss",
        "query_path": "./data/queries/wikiqma/wikiqma.csv"
    }, 

    "hotpotqa": {
        "document_name": "hotpotqa",
        "context_path" : "./data/context_files/hotpotqa.csv",
        "text_output_path" : "./data/text_files/hotpotqa/combined.txt",
        "json_output_path" : "./data/json_files/hotpotqa/hotpotqa-csize200.json",
        "index_storage_path": "./index_store/hotpotqa.faiss",
        "query_path": "./data/queries/hotpotqa/hotpotqa.csv"
    },

    "multifieldqa_en":
    {
        "document_name": "multifieldqa_en",
        "context_path": "./data/context_files/multifieldqa_en.csv",
        "text_output_path": "./data/text_files/multifieldqa_en/combined.txt",
        "json_output_path": "./data/json_files/multifieldqa_en/multifieldqa_en-csize200.json",
        "index_storage_path": "./index_store/multifieldqa_en.faiss",
        "query_path": "./data/queries/multifieldqa_en/multifieldqa_en.csv"
    },

    "musique":
    {
        "document_name": "musique",
        "context_path": "./data/context_files/musique.csv",
        "text_output_path": "./data/text_files/musique/combined.txt",
        "json_output_path": "./data/json_files/musique/musique-csize200.json",
        "index_storage_path": "./index_store/musique.faiss",
        "query_path": "./data/queries/musique/musique.csv"
    },
    
    "passage_retrieval_en": {
        "document_name": "passage_retrieval_en",
        "context_path": "./data/context_files/passage_retrieval_en.csv",
        "text_output_path": "./data/text_files/passage_retrieval_en/combined.txt",
        "json_output_path": "./data/json_files/passage_retrieval_en/passage_retrieval_en-csize200.json",
        "index_storage_path": "./index_store/passage_retrieval_en.faiss",
        "query_path": "./data/queries/passage_retrieval_en/passage_retrieval_en.csv"
    },
    
    "qasper": {
        "document_name": "qasper",
        "context_path": "./data/context_files/qasper.csv",
        "text_output_path": "./data/text_files/qasper/combined.txt",
        "json_output_path": "./data/json_files/qasper/qasper-csize200.json",
        "index_storage_path": "./index_store/qasper.faiss",
        "query_path": "./data/queries/qasper/qasper.csv"
    },
    
    "repobench-p": {
        "document_name": "repobench-p",
        "context_path": "./data/context_files/repobench-p.csv",
        "text_output_path": "./data/text_files/repobench-p/combined.txt",
        "json_output_path": "./data/json_files/repobench-p/repobench-p-csize200.json",
        "index_storage_path": "./index_store/repobench-p.faiss",
        "query_path": "./data/queries/repobench-p/repobench-p.csv"
    },

    "triviaqa": {
        "document_name": "triviaqa",
        "context_path": "./data/context_files/triviaqa.csv",
        "text_output_path": "./data/text_files/triviaqa/combined.txt",
        "json_output_path": "./data/json_files/triviaqa/triviaqa-csize200.json",
        "index_storage_path": "./index_store/triviaqa.faiss",
        "query_path": "./data/queries/triviaqa/triviaqa.csv"
    },

    "trec": {
        "document_name": "trec",
        "context_path": "./data/context_files/trec.csv",
        "text_output_path": "./data/text_files/trec/combined.txt",
        "json_output_path": "./data/json_files/trec/trec-csize200.json",
        "index_storage_path": "./index_store/trec.faiss",
        "query_path": "./data/queries/trec/trec.csv"
    },


    "samsum": {
        "document_name": "samsum",
        "context_path": "./data/context_files/samsum.csv",
        "text_output_path": "./data/text_files/samsum/combined.txt",
        "json_output_path": "./data/json_files/samsum/samsum-csize200.json",
        "index_storage_path": "./index_store/samsum.faiss",
        "query_path": "./data/queries/samsum/samsum.csv"
    },

    "qmsum": {
        "document_name": "qmsum",
        "context_path": "./data/context_files/qmsum.csv",
        "text_output_path": "./data/text_files/qmsum/combined.txt",
        "json_output_path": "./data/json_files/qmsum/qmsum-csize200.json",
        "index_storage_path": "./index_store/qmsum.faiss",
        "query_path": "./data/queries/qmsum/qmsum.csv"
    },

    "narrativeqa": {
        "document_name": "narrativeqa",
        "context_path": "./data/context_files/narrativeqa.csv",
        "text_output_path": "./data/text_files/narrativeqa/combined.txt",
        "json_output_path": "./data/json_files/narrativeqa/narrativeqa-csize200.json",
        "index_storage_path": "./index_store/narrativeqa.faiss",
        "query_path": "./data/queries/narrativeqa/narrativeqa.csv"
    },

}


from utils import generate_mean_statistics
from pipeline import establish_baseline
from utils import filter_questions
# from pipeline import average_chunks
def running_pipeline(dataset_name, 
                model_path, 
                chunk_size,
                use_pdf,
                ):
    """
    Runs the complete pipeline for a given dataset and query type.

    Args:
        dataset_name (str): The dataset to process.
        model_path (str): Path to the model to be used.
        chunk_size (int): Size of text chunks.
    """
    print(f"Running pipeline with dataset: {dataset_name}, model: {model_path}, chunk size: {chunk_size}, use_pdf: {use_pdf}")

    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(model_path)
    # Load dataset configuration
    dataset = datasets[dataset_name]
    # Extract text from the dataset

    from utils import load_context
    if use_pdf:
        text = load_pdf(pdf_paths=dataset["pdf_paths"], 
                        output_txt_path=dataset["text_output_path"])
    else:
        text = load_context(context_path = dataset["context_path"],
                            output_txt_path=dataset["text_output_path"]
                            )
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
                                        rag_type=None,
                                        flatten_distribution=False
                                               )

    from pipeline import compute_tsne
    # from pipeline import plot_tsne
    from pipeline import plot_tsne_per_query
    pg_query_map = {f"Q{i+1}": query for i, query in enumerate(queries_list)}
    # prefix_map, data_emb = compute_tsne(json_path=dataset["json_output_path"], 
    #                         id_query_map=pg_query_map)
    # plot_tsne_per_query(prefix_map, data_embedding=data_emb)
    prefix_index_map = compute_prefix_index_map(pg_query_map, faiss_index=index, document_name=dataset["document_name"])

        #Plot the distribution of query lengths.
    from utils import plot_query_length_distribution
    plot_query_length_distribution(prefix_index_map=prefix_index_map, document_name=dataset["document_name"])
    
    from pipeline import prev_prefix_match_ordered
    # prev_prefix_index_map = prev_prefix_match_ordered(prefix_index_map)
    # from utils import plot_prefix_match_ordered
    # plot_prefix_match_ordered(prev_prefix_index_map, dataset["document_name"])
    from pipeline import prev_prefix_match_intersect
    prev_prefix_index_map_intersection = prev_prefix_match_intersect(prefix_index_map)

    from utils import plot_prefix_match_intersect
    # plot_prefix_match_intersect(prev_prefix_index_map_intersection, document_name=dataset["document_name"])

    from utils import plot_total_bandwidth
    # plot_total_bandwidth(query_id_datatransferred_map)
    from utils import plot_total_bandwidth_heatmap
    # plot_total_bandwidth_heatmap(query_id_bandwidth_use_final)
    
    """ Plot data transfer for tio"""
    from pipeline import data_transfer_words
    word_lengths = [5, 10, 15, 20, 25]
    from pipeline import plot_bandwidth_k
    word_map = {}
    for k in word_lengths: 
        qid_bwidth = data_transfer_words(word_length=k, prefix_index_map = prefix_index_map)
        from utils import data_transfer_mean_k_words
        mean_bandwidth, mean_delay = data_transfer_mean_k_words(qid_bwidth, dataset["document_name"])
        word_map[k] = [mean_bandwidth, mean_delay]

    top_k_list = [5, 10, 20, 30, 40, 50]
    from pipeline import data_transfer_top_k_word
    for k in word_lengths: 
        for top_k in top_k_list:
            qid_bwidth = data_transfer_top_k_word(word_length=k, 
                                                 prefix_index_map = prefix_index_map,
                                                 top_k_chunks = top_k,
                                                 faiss_index=index)
        
            from utils import data_transfer_mean_k_words
            mean_bandwidth, mean_delay = data_transfer_mean_k_words(qid_bwidth, dataset["document_name"])
            word_map[f"WS:{k}-Top{top_k}chunks"] = [mean_bandwidth, mean_delay]
    
    """Plot datatransfer for different data/delay with CC Algo """

    top_k_chunks = [5, 10, 20, 30, 40, 50]
    from pipeline import datatransferred_consecutive_top_k
    for top_k in top_k_chunks:
        qid_bwidth = datatransferred_consecutive_top_k(prefix_index_map = prefix_index_map,
                                                top_k_chunks = top_k,
                                                faiss_index=index)
    
        from utils import data_transfer_mean_k_words
        mean_bandwidth, mean_delay = data_transfer_mean_k_words(qid_bwidth, dataset["document_name"])
        word_map[f"CC-Top{top_k}chunks"] = [mean_bandwidth, mean_delay]

    
    from utils import plot_mean_bandwidth_delay
    plot_mean_bandwidth_delay(word_map, dataset["document_name"])

    
    # from computeIndexRetriever import AdaptiveStabilityRetriever

    # for window_size in np.arange(2, 10, 2):  # Example range [1, 5] with step 0.2
    #             for stability_threshold in np.round(np.arange(0.5, 1, 0.2), 1):
    #                 asr = AdaptiveStabilityRetriever(window_size=int(window_size),
    #                                                  stability_threshold=stability_threshold
    #                                                  )
    #                 # index_map, index = asr.should_retrieve(prev_prefix_index_map_intersection)

    #                 # from utils import data_transfer_mean
    #                 # mean_data_transfer, mean_delay = data_transfer_mean(index_map, dataset["document_name"])
    #                 word_map[f"WS:{window_size}-ST:{stability_threshold}"] = [mean_data_transfer, mean_delay]


if __name__ == "__main__":
    print("Main Function")
    parser = argparse.ArgumentParser(description="Run dataset processing pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--chunk_size", type=int, required=True, help="Chunk size")
    parser.add_argument("--use_pdf", type=lambda x: (str(x).lower() == 'true'), required=True, help="Use PDF (true/false)")

    args = parser.parse_args()

    print("Parsed arguments:", args)  # Debugging line
    print(f"Dataset: {args.dataset}")
    print(f"Model Path: {args.model_path}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Use PDF: {args.use_pdf}")

    running_pipeline(args.dataset, args.model_path, args.chunk_size, args.use_pdf)