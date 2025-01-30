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
