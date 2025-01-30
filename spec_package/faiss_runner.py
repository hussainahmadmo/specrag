from faiss_modular import create_flat_index, load_or_create_indices

if __name__ == "__main__":
    # 1) Decide whether to use HNSW or Flat L2
    USE_HNSW = True  # Toggle True/False to try different indexes

    # 2) Load or create indexes
    data_dict = load_or_create_indices(use_hnsw=USE_HNSW)
    ds_qa = data_dict["qa_dataset"]
    ds_text = data_dict["text_dataset"]
    qa_index = data_dict["qa_index"]
    text_index = data_dict["text_index"]
    embed_model = data_dict["model"]

    # 3) Prepare a DataFrame for logging
    columns = ["dataset", "k_example", "embed_time", "search_time", "total_time", "use_hnsw", "query_text"]
    log_df = pd.DataFrame(columns=columns)

    # 4) Example similarity searches
    queries = ["What is AI?", "What is Python?", "Orange", "Machine Learning"]
    for query_text in queries:
        # Search QA dataset
        log_df, retrieved_qa = check_scores(
            dataset=ds_qa,
            faiss_index=qa_index,
            k_example=5,
            query_text=query_text,
            embed_model=embed_model,
            log_df=log_df,
            use_hnsw=USE_HNSW,
            dataset_name="qa_dataset"
        )

        # Print some results
        print(f"\n[QA DATASET] Query: {query_text}")
        for i, item in enumerate(retrieved_qa, start=1):
            print(f"  {i}. {item}")

        # Search Text dataset
        log_df, retrieved_text = check_scores(
            dataset=ds_text,
            faiss_index=text_index,
            k_example=5,
            query_text=query_text,
            embed_model=embed_model,
            log_df=log_df,
            use_hnsw=USE_HNSW,
            dataset_name="text_dataset"
        )

        print(f"\n[TEXT DATASET] Query: {query_text}")
        for i, item in enumerate(retrieved_text, start=1):
            print(f"  {i}. {item}")

    # 5) Save log to CSV
    log_df.to_csv("similarity_search_timings.csv", index=False)
    logging.info("Saved similarity search timings to similarity_search_timings.csv")

    print("\n✅ Finished searching. Check 'similarity_search_timings.csv' for timing logs.")