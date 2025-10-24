from flask import Flask, g, render_template, request, jsonify
import os
import logging
import time

# Import llama-index and related libraries
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.litellm import LiteLLM
from my_config import MY_CONFIG
import query_utils


os.environ['HF_ENDPOINT'] = MY_CONFIG.HF_ENDPOINT



app = Flask(__name__)

# Global variables for LLM and index
vector_index = None

initialization_complete = False
def initialize():
    """
    Initialize LLM and Milvus vector database using llama-index.
    This function sets up the necessary components for the chat application.
    """
    global vector_index, initialization_complete
    
    if initialization_complete:
        return
    
    logging.info("Initializing LLM and vector database...")
    
    # raise Exception ("init exception test") # debug
    
    try:
        ## embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name = MY_CONFIG.EMBEDDING_MODEL
        )
        print("✅ Using embedding model: ", MY_CONFIG.EMBEDDING_MODEL)
        
        # Setup LLM using LiteLLM
        llm = LiteLLM(
            model=MY_CONFIG.LLM_MODEL,
            temperature=0.1
        )
        print("✅ LLM run environment: ", MY_CONFIG.LLM_RUN_ENV)
        print("✅ Using LLM model : ", MY_CONFIG.LLM_MODEL)
        Settings.llm = llm
        
        # Initialize Milvus vector store for Vector RAG only
        vector_store = MilvusVectorStore(
            uri = MY_CONFIG.MILVUS_URI_VECTOR ,  # Use dedicated Vector-only database
            dim = MY_CONFIG.EMBEDDING_LENGTH , 
            collection_name = MY_CONFIG.COLLECTION_NAME,
            overwrite=False  # so we load the index from db
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print ("✅ Connected to Vector-only Milvus instance: ", MY_CONFIG.MILVUS_URI_VECTOR )
        
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context)
        print ("✅ Loaded Vector-only index from:", MY_CONFIG.MILVUS_URI_VECTOR )

        logging.info("Successfully initialized LLM and vector database")
    
        initialization_complete = True
    except Exception as e:
        initialization_complete = False
        logging.error(f"Error initializing LLM and vector database: {str(e)}")
        raise (e)
        # return False
## -------------

## ----
@app.route('/')
def index():
    init_error = app.config.get('INIT_ERROR', '')
    # init_error = g.get('init_error', None)
    return render_template('index.html', init_error=init_error)
## end --- def index():


## -----
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    
    # Get response from LLM
    response = get_llm_response(user_message)
    # print (response)
    
    return jsonify({'response': response})
## end : def chat():


def get_llm_response(message):
    """
    Process the user message and get a response from the LLM using Vector RAG
    with structured prompting
    """
    global vector_index, initialization_complete
    
    # Check if LLM and index are initialized
    if vector_index is None or  initialization_complete is None:
        return "System did not initialize. Please try again later."
    
    start_time = time.time()
    response_text = ''
    
    try:
        # raise Exception ("chat exception test") ## debug
        # Create a query engine from the index
        query_engine = vector_index.as_query_engine()
        
        # Apply query optimization
        message = query_utils.tweak_query(message, MY_CONFIG.LLM_MODEL)
        
        # Get initial vector response
        vector_response = query_engine.query(message)
        vector_text = str(vector_response).strip()
        
        # Structured prompt
        structured_prompt = f"""Please provide a comprehensive, well-structured answer using the provided document information.

Question: {message}

Document Information:
{vector_text}

Instructions:
1. Provide accurate, factual information based on the documents
2. Structure your response clearly with proper formatting
3. Be comprehensive yet concise
4. Highlight key relationships and important details when relevant
5. Use bullet points or sections when appropriate for clarity

Please provide your answer:"""
        
        # Use structured prompt for final synthesis
        final_response = query_engine.query(structured_prompt)
        
        if final_response:
            response_text = str(final_response).strip()
        
    except Exception as e:
        logging.error(f"Error getting LLM response: {str(e)}")
        response_text =  f"Sorry, I encountered an error while processing your request:\n{str(e)}"
        
    end_time = time.time()
    
    # add timing stat
    response_text += f"\n⏱️ *Total time: {(end_time - start_time):.1f} seconds*"
    return response_text
    
## --- end: def get_llm_response():


    

## -------
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.info("App starting up...")
    
    # Initialize LLM and vector database
    try:
        initialize()
    except Exception as e:
        logging.warning("Starting without LLM and vector database. Responses will be limited.")
        app.config['INIT_ERROR'] = str(e)
        # g.init_error = str(e)
        
    
    # Vector RAG Flask App - Port 8080
    PORT = int(os.environ.get("PORT", 8080))  # Vector RAG only
    print(f"🚀 Vector RAG Flask app starting on port {PORT}")
    app.run(host="0.0.0.0", debug=False, port=PORT)
## -- end main ----