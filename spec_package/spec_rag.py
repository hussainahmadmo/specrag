import logging
import sys
import json
import yaml
import time
import numpy as np
import pandas as pd
from huggingface_hub import login
from llama_index.core import (
    get_response_synthesizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from datasets import load_dataset
from config_manager import ConfigManager
import os
config = ConfigManager.load_config()
dataset_dict = {}

def setup_config():
    # logging.info("Inside the setup config file")
    # """Setup environments settings."""
    # Settings.embed_model = HuggingFaceEmbedding(model_name=config['embed_model'])
    # Settings.chunk_size = config['chunk_size']
    # Settings.chunk_overlap = config['chunk_overlap']
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    # login(config["token_id"])
    pass

def embed(text, model):
    return model.encode(text)

from sentence_transformers import SentenceTransformer

def load_or_create_indices():
    """
    Initilize dataset and create embeddings, create FAISS index for each and save them to paths.
    Args:
        dataset_name : A huggingface dataset.
    Returns:
     ds_qa -  Dataset Object
     ds_text - Dataset Object
     """
    config = ConfigManager.get_config()
    qa_index_path = config["qa_index_path"]
    text_index_path = config["text_index_path"]
    embed_model = config['embed_model']

    if os.path.exists(qa_index_path) and os.path.exists(text_index_path):
        print("FAISS indices already exist. Loading them.")
        # Load the FAISS indices
        ds_qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer", split="test")
        ds_qa.load_faiss_index('embeddings', qa_index_path)

        ds_text = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")
        ds_text.load_faiss_index('embeddings', text_index_path)
    else:
        print("FAISS indices not found. Creating new ones.")
        model = SentenceTransformer(embed_model)

        # Create datasets with embeddings
        ds_qa = load_dataset("rag-datasets/rag-mini-wikipedia", 
                             "question-answer", 
                             split="test").map(lambda example: {'embeddings': embed(example["question"], model)})

        ds_text = load_dataset("rag-datasets/rag-mini-wikipedia", 
                               "text-corpus", 
                               split='passages').map(lambda example: {'embeddings': embed(example["passage"], model)})

        # Add and save FAISS indices
        ds_qa.add_faiss_index(column='embeddings')
        ds_qa.save_faiss_index('embeddings', qa_index_path)

        ds_text.add_faiss_index(column='embeddings')
        ds_text.save_faiss_index('embeddings', text_index_path)

    dataset_dict.update({"qa_dataset" : ds_qa, "text_dataset" : ds_text})
    return dataset_dict

import time
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

import faiss
import os
import numpy as np
import time
import logging
import pandas as pd

def check_scores(datasets, 
                 index_path: str,
                k_example: int, 
                query_text: str, 
                embed_model, 
                log_df: pd.DataFrame, 
                use_hnsw: bool = False):
    """
    Perform a similarity search using either HNSW FAISS or standard FAISS and log execution times.

    Args:
        datasets: The dataset containing the FAISS index.
        index_path (str): Path to the FAISS index file.
        k_example (int): Number of nearest neighbors to retrieve.
        query_text (str): The query for which to perform similarity search.
        embed_model: The embedding model used to encode queries.
        log_df (pd.DataFrame): DataFrame to store performance logs.
        use_hnsw (bool): Whether to use HNSW FAISS. Default is False (uses standard FAISS).
    
    Returns:
        log_df (pd.DataFrame): Updated log dataframe with execution times.
        retrieved_examples (list): List of retrieved dataset examples.
    """
    try:
        start_time = time.time()  # Start timing the whole function
        # Load or create FAISS index
        t1 = time.time()
        d = embed_model.get_sentence_embedding_dimension()  # Get embedding size

        if os.path.exists(index_path) and not use_hnsw:
            logging.info(f"Loading FAISS Flat L2 index from {index_path}")
            index = faiss.read_index(index_path)

        elif os.path.exists(index_path) and use_hnsw:
            logging.info(f"Creating a new HNSW index instead of using the saved FAISS index")
            index = faiss.IndexHNSWFlat(d, 32)
            index.hnsw.efSearch = max(50, k_example * 2)  # Dynamically adjust efSearch
            datasets.load_faiss_index("embeddings", index_path)  # Rebuild the index

        else:
            logging.info(f"Creating a new FAISS index. HNSW mode: {use_hnsw}")
            if use_hnsw:
                index = faiss.IndexHNSWFlat(d, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = max(50, k_example * 2)
            else:
                index = faiss.IndexFlatL2(d)

            # Ensure FAISS index is properly trained and populated
            if not index.is_trained:
                logging.info("Training FAISS index...")
                index.train(datasets["embeddings"])

            index.add(datasets["embeddings"])  # Explicitly add embeddings to FAISS

        if isinstance(index, faiss.IndexHNSWFlat):
            logging.info("Using HNSW FAISS index")
        elif isinstance(index, faiss.IndexFlatL2):
            logging.info("Using Flat L2 FAISS index ⚠️")

        # Generate the embedding for the query
        t2 = time.time()
        query_embedding = np.array([embed_model.encode(query_text)], dtype="float32")
        query_embed_time = time.time() - t2
        logging.info(f"Query embedding took {query_embed_time:.4f} seconds")

        # Perform the search
        t3 = time.time()
        distances, indices = index.search(query_embedding, k_example)
        search_time = time.time() - t3
        logging.info(f"Similarity search took {search_time:.4f} seconds")

        total_time = time.time() - start_time
        logging.info(f"Total execution time: {total_time:.4f} seconds")

        # Retrieve examples based on FAISS indices
        retrieved_examples = [datasets[int(idx)] for idx in indices[0] if idx != -1]

        # Append the results to the DataFrame
        dataset_name = "qa_dataset" if "question-answer" in index_path else "text_dataset"
        log_df.loc[len(log_df)] = [dataset_name, k_example, faiss_load_time, query_embed_time, search_time, total_time, use_hnsw, query_text]

        return log_df, retrieved_examples  # Return updated DataFrame and retrieved examples

    except Exception as e:
        logging.error(f"Error during score checking: {e}")
        raise



# # --- QUERY ENGINE SETUP ---
# def setup_query_engine(index, llm, config, prompt_template):
#     """Set up the query engine."""
#     prompt_helper = PromptHelper(context_window=32000)  # For Mistral
#     response_synthesizer = get_response_synthesizer(
#         response_mode="compact", prompt_helper=prompt_helper, streaming=True,
#     )
#     query_engine = index.as_query_engine(
#         llm=llm,
#         response_synthesizer=response_synthesizer,
#         similarity_top_k=config['similarity_top_k'],
#         streaming=True,
#     )
#     query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})
#     return query_engine

# # --- QUERY EXECUTION ---
# def execute_queries(query_engine, query_path, answers_path, output_file, num_requests):
#     """Execute queries and save results."""
#     query_list = get_queries(query_path, num_requests)
#     answers_list = get_answers(answers_path, num_requests)
#     assert len(query_list) == len(answers_list)

#     lambda_poisson = 20

#     for idx, query in enumerate(query_list):
#         start = time.time()
#         time_interval = np.random.exponential(1 / lambda_poisson)
#         print(query)
#         response = query_engine.query(query)
#         parsed_response = parser_answer(str(response))

#         answer = json.loads(answers_list[idx])['answer']

#         with open(output_file, 'a') as f:
#             f.write(f"{parsed_response};{answer};{time.time() - start}\n")

#         time.sleep(time_interval)


# # --- EVALUATION ---
# def evaluate_results(output_file):
#     """Evaluate the results using F1 score."""
#     df = pd.read_csv(output_file, sep=';', header=None)
#     df.columns = ['response', 'answer', 'time']
#     gt = df['answer'].values
#     pred = df['response'].values
#     assert len(gt) == len(pred)
#     scorer = Scorer(metric='f1')
#     scores = [scorer.compute_f1(pred[i], gt[i]) for i in range(len(gt))]

#     logging.info(f"F1 Score: {np.mean(scores)}")