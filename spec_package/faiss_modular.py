import os
import time
import logging
import numpy as np
import pandas as pd

# ---------- FAISS -----------
import faiss

# ---------- HuggingFace / Datasets -----------
from datasets import load_dataset

# ---------- Sentence Embeddings -----------
from sentence_transformers import SentenceTransformer

# ---------- Example "ConfigManager" -----------
# If you have your own config_manager.py, you can remove/replace this.
class ConfigManager:
    @staticmethod
    def get_config():
        return {
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            "qa_index_path": "qa_index.faiss",
            "text_index_path": "text_index.faiss"
        }

# ---------- Logging Setup -----------
logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ========== 1. HELPER FUNCTIONS FOR FAISS INDEX CREATION ==========

def create_flat_index(dimension: int) -> faiss.Index:
    """
    Create a Flat L2 FAISS index for the given dimension.
    """
    index = faiss.IndexFlatL2(dimension)
    return index

def create_hnsw_index(dimension: int, m: int = 32,
                      ef_construction: int = 200,
                      ef_search: int = 50) -> faiss.Index:
    """
    Create an HNSW FAISS index for the given dimension.
    
    :param dimension: Embedding size
    :param m: Number of neighbors each node connects to
    :param ef_construction: Determines graph quality (higher=slower but more accurate)
    :param ef_search: Larger=better recall, slower search
    """
    index = faiss.IndexHNSWFlat(dimension, m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    return index

def build_faiss_index(dataset, column_name: str, index_path: str, create_index_fn) -> faiss.Index:
    """
    Build a FAISS index by:
      1. Creating a new index via `create_index_fn`.
      2. Training the index (if needed).
      3. Adding dataset embeddings to the index.
      4. Saving the index to disk.
    """
    # Example embedding to determine dimension
    example_embedding = dataset[0][column_name]
    dimension = len(example_embedding)  # Usually the length of the embedding vector

    # 1) Create empty index
    index = create_index_fn(dimension)

    # 2) Train index (if it supports training, e.g., IVF). For Flat/HNSW, no real "training" needed.
    if not index.is_trained:
        logging.info("Training FAISS index (if supported).")
        embeddings = dataset[column_name]
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)
        index.train(embeddings)

    # 3) Add embeddings
    if isinstance(dataset[column_name], list):
        embeddings = np.array(dataset[column_name], dtype=np.float32)
        index.add(embeddings)
    else:
        # If dataset[column_name] is already a numpy array
        index.add(dataset[column_name])

    # 4) Save to disk
    faiss.write_index(index, index_path)
    logging.info(f"FAISS index saved to {index_path}")

    return index

def load_faiss_index(index_path: str) -> faiss.Index:
    """Load a FAISS index from the given path."""
    logging.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    return index

# ========== 2. LOADING OR CREATING DATASETS & INDEXES ==========
from model_server import get_embedding, start_model_server
start_model_server()

def embed(text):
    return {"embeddings": get_embedding(text)}

def load_or_create_indices(use_hnsw: bool = False):
    """
    Initialize the dataset, create embeddings, and build FAISS indexes.
    Returns both datasets & loaded FAISS indexes.
    
    :param use_hnsw: If True, build HNSW indexes; otherwise build Flat L2.
    :return: Dictionary with datasets & indexes.
    """
    config = ConfigManager.get_config()
    text_index_path = config["text_index_path"]
    embed_model_name = config["embed_model"]
    # Decide which function to use for index creation
    create_index_fn = create_hnsw_index if use_hnsw else create_flat_index

    # ---------- LOAD DATASETS ----------
    logging.info("Loading HuggingFace Datasets for QA and Text Corpus.")
    ds_text = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus", split="passages")

    # ---------- EMBEDDING STEP ----------
    if "embeddings" not in ds_text.column_names:
        ds_text = ds_text.map(lambda ex: {"embeddings": embed(ex["passage"])["embeddings"][0]})
        print(type(ds_text[0]["embeddings"]))  # Should print <class 'list'>
        print(len(ds_text[0]["embeddings"]))   # Should print 384
        print(ds_text[0]["embeddings"][:5])    # Print first 5 numbers to verify
    # ---------- CREATE / LOAD TEXT INDEX ----------
    if not os.path.exists(text_index_path):
        logging.info(f"No text index found at {text_index_path}. Building a new one.")
        text_index = build_faiss_index(ds_text, "embeddings", text_index_path, create_index_fn)
    else:
        logging.info(f"Text index already exists at {text_index_path}. Loading it.")
        text_index = load_faiss_index(text_index_path)

    # Return everything in a dict
    return {
        "text_dataset": ds_text,
        "text_index": text_index,
    }

# ========== 3. SIMILARITY SEARCH FUNCTION ==========

def check_scores(
    dataset,
    faiss_index: faiss.Index,
    k_example: int,
    query_text: str,
    log_df: pd.DataFrame,
    use_hnsw: bool = False,
    dataset_name: str = "qa_dataset"
):
    """
    Perform a similarity search using the provided FAISS index and log execution times.
    
    :param dataset: The HF dataset with data
    :param faiss_index: A pre-built or loaded FAISS index
    :param k_example: Number of neighbors to retrieve
    :param query_text: The query to encode & search
    :param log_df: DataFrame for logging results
    :param use_hnsw: Flag for logging only (HNSW or Flat)
    :param dataset_name: For logging, e.g. "qa_dataset" or "text_dataset"
    :return: (updated log_df, retrieved_examples)
    """
    try:
        start_time = time.time()

        # 1) Encode query using model server
        t_query = time.time()
        query_embedding = np.array(get_embedding(query_text), dtype=np.float32)  # Use model server function
        embed_time = time.time() - t_query

        # 2) Perform the FAISS search
        t_search = time.time()
        distances, indices = faiss_index.search(query_embedding, k_example)
        search_time = time.time() - t_search

        # 3) Gather retrieved examples
        retrieved_examples = [dataset[int(idx)] for idx in indices[0] if idx != -1]

        total_time = time.time() - start_time

        # 4) Logging
        logging.info(f"Using {'HNSW' if use_hnsw else 'Flat L2'} index")
        logging.info(f"Query: {query_text}")
        logging.info(f"Embed time: {embed_time:.4f} sec | Search time: {search_time:.4f} sec | Total: {total_time:.4f}")
        logging.info(f"Top {k_example} indices: {indices[0]}")
        logging.info(f"Top {k_example} distances: {distances[0]}")

        # 5) Append row to log_df
        log_df.loc[len(log_df)] = [
            dataset_name,  # "qa_dataset" or "text_dataset"
            k_example,
            embed_time,
            search_time,
            total_time,
            use_hnsw,
            query_text
        ]

        return log_df, retrieved_examples

    except Exception as e:
        logging.error(f"Error in check_scores: {str(e)}")
        return log_df, []

# ========== 4. MAIN DEMO ==========

if __name__ == "__main__":
    # 1) Decide whether to use HNSW or Flat L2
    USE_HNSW = True  # Toggle True/False to try different indexes

    # 2) Load or create indexes
    data_dict = load_or_create_indices(use_hnsw=USE_HNSW)
    ds_text = data_dict["text_dataset"]
    text_index = data_dict["text_index"]

    # 3) Prepare a DataFrame for logging
    columns = ["dataset", "k_example", "embed_time", "search_time", "total_time", "use_hnsw", "query_text"]
    log_df = pd.DataFrame(columns=columns)

    # 4) Example similarity searches
    queries = ["What is AI?", "What is Python?", "Orange", "Machine Learning"]
    k_values = [1, 10, 100]  # Different k-values to test

    for k_example in k_values:
        print(f"\n=== Running similarity search with k={k_example} ===\n")
        for query_text in queries:
            # Perform similarity search
            log_df, retrieved_text = check_scores(
                dataset=ds_text,
                faiss_index=text_index,
                k_example=k_example,  # Change k dynamically
                query_text=query_text,
                log_df=log_df,
                use_hnsw=USE_HNSW,
                dataset_name="text_dataset"
            )

            # Print retrieved passages
            print(f"\n[TEXT DATASET] Query: {query_text} (k={k_example})")
            for i, item in enumerate(retrieved_text, start=1):
                print(f"  {i}. {item['passage']}")  # Print retrieved text

    # 5) Save log to CSV
    log_df.to_csv("similarity_search_timings.csv", index=False)
    logging.info("Saved similarity search timings to similarity_search_timings.csv")

    print("\n✅ Finished searching. Check 'similarity_search_timings.csv' for timing logs.")
