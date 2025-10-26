# GraphRAG Demo Setup

This guide walks you through setting up and running the AllyCAT GraphRAG demo natively with Python.

## Prerequisites

### 1. Install Python Requirements
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file with the following settings:

#### Neo4j Configuration
```bash
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

Get a free Neo4j Aura instance at: https://neo4j.com/cloud/aura/

#### Vector Database Configuration
```bash
# For Cloud (Zilliz) - Recommended
VECTOR_DB_TYPE=cloud_zilliz
ZILLIZ_CLUSTER_ENDPOINT=https://your-cluster.zilliz.cloud
ZILLIZ_TOKEN=your_zilliz_token

# Or for Local (Milvus Lite)
VECTOR_DB_TYPE=local
```

Get a free Zilliz Cloud cluster at: https://cloud.zilliz.com/

#### LLM Configuration
```bash
# Cloud LLM (recommended)
LLM_RUN_ENV=cloud
LLM_MODEL=cerebras/llama3.1-8b
CEREBRAS_API_KEY=your_cerebras_api_key

# Or Gemini
GEMINI_API_KEY=your_gemini_api_key

# Or local Ollama
LLM_RUN_ENV=local_ollama
LLM_MODEL=ollama/gemma3:1b
OLLAMA_MODEL=gemma3:1b
```

Get free API keys:
- Cerebras: https://cerebras.ai/
- Gemini: https://aistudio.google.com/

## Pipeline Workflow (Run in Order)

### Step 1: Crawl Website
```bash
python 1_crawl_site.py
```
**Output:** Downloads HTML/PDF files to `workspace/crawl_data/`

### Step 2: Process Files to Markdown
```bash
python 2_process_files.py
```
**Output:** Converts files to markdown in `workspace/processed_data/`

### Step 3: Save to Vector Database
```bash
# For Zilliz Cloud
python 3_save_to_vector_db_zilliz.py

# Or for local Milvus
python 3_save_to_vector_db.py
```
**Output:** Creates embeddings and stores in vector database

### Step 4: Process Graph Data (3 Phases)
```bash
# Phase 1: Extract entities and relationships
python 2b_process_graph_phase1.py

# Phase 2: Build communities
python 2b_process_graph_phase2.py

# Phase 3: Generate community summaries
python 2b_process_graph_phase3.py
```
**Output:** Creates graph data files in `workspace/graph_data/`

### Step 5: Save to Graph Database
```bash
python 3b_save_to_graph_db.py
```
**Output:** Loads graph data into Neo4j database

## Running the Applications

### Command Line Query
```bash
# Vector RAG Query
python 4_query.py

# GraphRAG Query 
python 4b_query_graph.py
```

### Flask Web Applications
```bash
# GraphRAG Flask (default port 8080)
python app_flask_graph.py

# Simple Vector RAG Flask (default port 8081)
python app_flask.py
```

### Chainlit Chat Applications
```bash
# GraphRAG Chainlit with Interactive UI
chainlit run app_chainlit_graph.py

# Vector RAG Chainlit
chainlit run app_chainlit.py
```

**Note:** Chainlit's default port is **8000**. To use custom ports configured in your `.env` file, add the `--port` flag:

```bash
# With custom ports from .env configuration
chainlit run app_chainlit_graph.py --port 8083
chainlit run app_chainlit.py --port 8082
```

The custom ports (8082, 8083) are primarily used for Docker deployments to avoid port conflicts when running multiple applications simultaneously.

## Application Ports

AllyCAT supports multiple application types with configurable ports via environment variables in `.env`:

| Application | Type | Environment Variable | Default Port | Features |
|-------------|------|---------------------|--------------|----------|
| app_flask_graph.py | Flask | FLASK_GRAPH_PORT | 8080 | Full GraphRAG with vector + graph |
| app_flask.py | Flask | FLASK_VECTOR_PORT | 8081 | Simple vector RAG only |
| app_chainlit_graph.py | Chainlit | CHAINLIT_GRAPH_PORT | 8083 | Interactive chat with GraphRAG |
| app_chainlit.py | Chainlit | CHAINLIT_VECTOR_PORT | 8082 | Interactive chat with vector RAG |

### Port Configuration

Add to your `.env` file to customize ports:
```bash
FLASK_GRAPH_PORT=8080         # Flask GraphRAG app port
FLASK_VECTOR_PORT=8081        # Flask Vector RAG app port
CHAINLIT_GRAPH_PORT=8083      # Chainlit GraphRAG app port
CHAINLIT_VECTOR_PORT=8082     # Chainlit Vector RAG app port
```

### Running Applications

```bash
# Flask apps use the port from MY_CONFIG automatically
python app_flask_graph.py              # Uses FLASK_GRAPH_PORT (8080)
python app_flask.py                    # Uses FLASK_VECTOR_PORT (8081)

# Chainlit apps run on default port 8000, or specify custom port
chainlit run app_chainlit_graph.py                    # Default: http://localhost:8000
chainlit run app_chainlit_graph.py --port 8083        # Custom port from .env

chainlit run app_chainlit.py                          # Default: http://localhost:8000
chainlit run app_chainlit.py --port 8082              # Custom port from .env
```

**Important:** The custom ports defined in `.env` (CHAINLIT_GRAPH_PORT, CHAINLIT_VECTOR_PORT) are primarily for Docker deployments to avoid port conflicts. For native Python execution, Chainlit defaults to port 8000 unless you explicitly specify `--port`.

### Access URLs

| Application | Default URL | With Custom Port |
|-------------|-------------|------------------|
| Flask GraphRAG | `http://localhost:8080` | Uses FLASK_GRAPH_PORT from .env |
| Flask Vector | `http://localhost:8081` | Uses FLASK_VECTOR_PORT from .env |
| Chainlit GraphRAG | `http://localhost:8000` | `http://localhost:8083` (if using `--port 8083`) |
| Chainlit Vector | `http://localhost:8000` | `http://localhost:8082` (if using `--port 8082`) |

**Note:** Chainlit's native default port is 8000. Custom ports are configured in `.env` for Docker deployments.

## Next Steps

For Docker deployment and production setup, see:
- [Docker Deployment Guide](../docker-deployment-guide.md)
- [Running in Docker](../running-in-docker.md)
