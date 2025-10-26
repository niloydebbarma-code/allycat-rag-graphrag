# Customizing Allycat

Allycat is highly customizable. See [configuration](configuration.md) for all available configs.

**Available Interfaces:**
- Flask GraphRAG: `app_flask_graph.py` (port 8080)
- Chainlit GraphRAG: `app_chainlit_graph.py` (port 8000 native, 8083 Docker)
- Flask Vector RAG: `app_flask.py` (port 8081)
- Chainlit Vector RAG: `app_chainlit.py` (port 8000 native, 8082 Docker)

**Topics:**
- [1 - To try a different LLM with Ollama](#1---to-try-a-different-llm-with-ollama)
- [2 - Trying a different model with Replicate](#2---trying-a-different-model-with-replicate)
- [3 - Trying various embedding models](#3---trying-various-embedding-models)
- [4 - Running GraphRAG vs Vector RAG](#4---running-graphrag-vs-vector-rag)


## 1 - To try a different LLM with Ollama

**Edit file [my_config.py](../my_config.py)**

```python
MY_CONFIG.LLM_RUN_ENV = 'local_ollama'
MY_CONFIG.LLM_MODEL = "gemma3:1b" 
``` 

Change the model to something else:

```python
MY_CONFIG.LLM_MODEL = "qwen3:1.7b"
```

**Download the ollama model**

```bash
ollama  pull qwen3:1.7b
```

Verify the model is ready:

```bash
ollama   list
```

Make sure the model is listed.  Now the model is ready to be used.

Now you can run the query:

```bash
python   4_query.py
```

## 2 - Trying a different model with Replicate

**Edit file [my_config.py](../my_config.py)**

```python
MY_CONFIG.LLM_RUN_ENV = 'replicate'
MY_CONFIG.LLM_MODEL = "meta/meta-llama-3-8b-instruct"
```

Change the model to the desired model:

```python
MY_CONFIG.LLM_MODEL = "ibm-granite/granite-3.2-8b-instruct
```

Now you can run the query:

```bash
python   4_query.py
```

## 3 - Trying various embedding models

**Edit file [my_config.py](../my_config.py)** and change these lines:

```python
MY_CONFIG.EMBEDDING_MODEL = 'ibm-granite/granite-embedding-30m-english'
MY_CONFIG.EMBEDDING_LENGTH = 384
```

You can find [embedding models here](https://huggingface.co/spaces/mteb/leaderboard)

Once embedding model is changed:

1) Rerun chunking and embedding again

```bash
python  3_save_to_vector_db.py
```

2) Run query

```bash
python   4_query.py
```

## 4 - Running GraphRAG vs Vector RAG

AllyCat supports both traditional Vector RAG and advanced GraphRAG.

**To run GraphRAG applications:**

```bash
# Flask GraphRAG interface
python app_flask_graph.py

# Chainlit GraphRAG interface
chainlit run app_chainlit_graph.py
```

**To run Vector RAG applications:**

```bash
# Flask Vector RAG interface
python app_flask.py

# Chainlit Vector RAG interface  
chainlit run app_chainlit.py
```

**In Docker**, set the `APP_TYPE` environment variable in your `.env` file:

```bash
APP_TYPE=flask_graph      # Flask GraphRAG (default)
APP_TYPE=chainlit_graph   # Chainlit GraphRAG
APP_TYPE=flask            # Flask Vector RAG
APP_TYPE=chainlit         # Chainlit Vector RAG
```

See [Docker Deployment Guide](docker-deployment-guide.md) for more details.


