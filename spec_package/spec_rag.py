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

def check_scores(datasets, index_path: str, k_example: int, query_text: str, embed_model, log_df: pd.DataFrame):
    """
    Perform a similarity search and log execution times.
    """
    try:
        start_time = time.time()  # Start timing the whole function
        # Load the FAISS index by its name
        t1 = time.time()
        datasets.load_faiss_index("embeddings", index_path)
        faiss_load_time = time.time() - t1
        logging.info(f"FAISS index loading took {faiss_load_time:.4f} seconds")
        
        # Generate the embedding for the query
        t2 = time.time()
        query_embedding = np.array([embed_model.encode(query_text)], dtype="float32")
        query_embed_time = time.time() - t2
        logging.info(f"Query embedding took {query_embed_time:.4f} seconds")
        
        # Perform the search
        t3 = time.time()
        scores, retrieved_examples = datasets.get_nearest_examples(
            index_name="embeddings",
            query=query_embedding,
            k=k_example
        )
        search_time = time.time() - t3
        logging.info(f"Similarity search took {search_time:.4f} seconds")

        total_time = time.time() - start_time
        logging.info(f"Total execution time: {total_time:.4f} seconds")

        # Append the results to the DataFrame
        dataset_name = "qa_dataset" if "question-answer" in index_path else "text_dataset"
        log_df.loc[len(log_df)] = [dataset_name, k_example, faiss_load_time, query_embed_time, search_time, total_time]

        return log_df  # Return updated DataFrame

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