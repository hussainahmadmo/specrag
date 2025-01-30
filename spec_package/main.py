from config_manager import ConfigManager
from spec_rag import setup_config
from spec_rag import load_or_create_indices
from spec_rag import check_scores

config = ConfigManager.get_config()
qa_index_path = config["qa_index_path"]
text_index_path = config["text_index_path"]
embed_model = config['embed_model']

def initialsetup():
    setup_config()

# # --- MAIN FUNCTION ---
def main():
    
    dataset_dictionary: dict = load_or_create_indices()
    print(dataset_dictionary)
    #generic query

    for i in range(100):
        check_scores(dataset_dictionary.get("qa_dataset"), index_path = qa_index_path, k_example=i, query_text = "What is")
        check_scores(dataset_dictionary.get("text_dataset"), qa_index_path = text_index_path, k_example=i, query_text = "Orange")





    # check_scores(datasets)
    


    
if __name__ == "__main__":
    initialsetup()
    main()