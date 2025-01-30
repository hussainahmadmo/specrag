import logging
from config_manager import ConfigManager
from spec_rag import setup_config, load_or_create_indices, check_scores
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

# Set GPU ID to 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging
logging.basicConfig(
    filename="app.log",  # Log file name
    filemode="a",  # Append mode
    level=logging.INFO,  # Log level (INFO and above)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

logging.info("Starting script execution")

# Load configuration
try:
    config = ConfigManager.get_config()
    logging.info("Configuration loaded successfully")
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    raise

qa_index_path = config.get("qa_index_path", "")
text_index_path = config.get("text_index_path", "")
embed_model = config.get("embed_model", "")

def initialsetup():
    logging.info("Running initial setup")
    try:
        setup_config()
        logging.info("Initial setup completed successfully")
    except Exception as e:
        logging.error(f"Error in initial setup: {e}")
        raise

# --- MAIN FUNCTION ---
def main():
    logging.info("Starting main function")
    
    try:
        dataset_dictionary: dict = load_or_create_indices()
        logging.info(f"Loaded dataset dictionary with keys: {list(dataset_dictionary.keys())}")
    except Exception as e:
        logging.error(f"Error loading datasets: {e}")
        raise

    logging.info("Loading SentenceTransformer model once before the loop")
    embed_model = SentenceTransformer(config['embed_model'], device="cuda")

    # Initialize a single DataFrame to log all results
    log_df = pd.DataFrame(columns=[
        "dataset", 
        "k", 
        "faiss_load_time", 
        "query_embed_time", 
        "search_time", 
        "total_time",
        "use_hnsw",
        "query_text"  # Added query_text column
    ])

    k_nearest_n = [1, 10, 100, 1000]
    # Run check_scores for both HNSW and standard FAISS
    for i in k_nearest_n:
        for dataset_name, dataset in {"qa_dataset": dataset_dictionary.get("qa_dataset"), 
                                      "text_dataset": dataset_dictionary.get("text_dataset")}.items():
            
            query_text = "What is" if dataset_name == "qa_dataset" else "Orange"

            for use_hnsw in [False, True]:  # Run twice, once for standard FAISS and once for HNSW
                logging.info(f"Running check_scores on {dataset_name} with k={i}, HNSW={use_hnsw}, query='{query_text}'")
                
                try:
                    log_df, retrieved_examples = check_scores(
                        datasets=dataset,
                        index_path=qa_index_path if dataset_name == "qa_dataset" else text_index_path,
                        k_example=i,
                        query_text=query_text,
                        embed_model=embed_model,
                        log_df=log_df,
                        use_hnsw=use_hnsw
                    )

                    # Print retrieved examples
                    print(f"\n🔹 Dataset: {dataset_name}, k={i}, HNSW={use_hnsw}")
                    print(f"🔸 Query: {query_text}")
                    print(f"🔸 Retrieved Examples (Top {i}):")
                    for idx, example in enumerate(retrieved_examples, start=1):
                        print(f"  {idx}. {example}\n")
                    logging.info(f"Retrieved {len(retrieved_examples)} examples for query: '{query_text}'")

                except Exception as e:
                    logging.error(f"Error running check_scores for {dataset_name} (HNSW={use_hnsw}): {e}")
                    continue  # Skip to the next iteration if an error occurs
            

    # Save the DataFrame to a CSV file after all iterations
    log_df.to_csv("similarity_search_timings.csv", index=False)
    logging.info("Logged similarity search times to similarity_search_timings.csv")

if __name__ == "__main__":
    logging.info("Script execution started")
    initialsetup()
    main()
    logging.info("Script execution finished successfully")
