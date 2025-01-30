import multiprocessing
import numpy as np
import uuid
from sentence_transformers import SentenceTransformer

# Global variables (initially set to None)
manager = None
request_queue = None
response_queue = None
process = None  

def model_server(request_queue, response_queue):
    """Runs the sentence transformer model and processes requests sequentially."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Model server is running...")
    while True:
        request = request_queue.get()
        if request == "exit":
            print("🛑 Model server shutting down.")
            break  # Graceful shutdown
        
        request_id, text = request  # Unpack (ID, text)
        embedding = model.encode([text]).tolist()
        response_queue.put((request_id, embedding))  # Return (ID, embedding)

def start_model_server():
    """Starts the model server as a background process if not already running."""
    global manager, request_queue, response_queue, process

    if process is None or not process.is_alive():
        manager = multiprocessing.Manager()
        request_queue = manager.Queue()
        response_queue = manager.Queue()

        process = multiprocessing.Process(target=model_server, args=(request_queue, response_queue))
        process.daemon = True  # Ensures process runs in the background
        process.start()
        print("✅ Model server started successfully.")

def get_embedding(text):
    """Sends a text request and ensures the response is retrieved correctly."""
    global request_queue, response_queue
    if request_queue is None or response_queue is None:
        raise RuntimeError("❌ Model server is not running. Call `start_model_server()` first.")
    
    request_id = str(uuid.uuid4())  # Generate a unique ID
    request_queue.put((request_id, text))  # Send (ID, text)

    while True:
        response_id, embedding = response_queue.get()
        if response_id == request_id:
            return np.array(embedding, dtype=np.float32)  # Ensure NumPy format

def stop_model_server():
    """Stops the model server gracefully."""
    global process, request_queue
    if process and process.is_alive():
        request_queue.put("exit")  # Send shutdown signal
        process.join()
        print("🛑 Model server stopped.")

# ✅ Ensure multiprocessing is only initialized when running the script directly
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Prevents bootstrapping issues
    start_model_server()
    print("🟢 Model server is now running and waiting for requests.")
    process.join()
