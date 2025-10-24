# GraphRAG Demo Setup

## Quick Start (Automated Pipeline)

### Option 1: Docker with Automatic Pipeline (Recommended for Cloud Deployment)

This is the **easiest way** to deploy to Heroku, AWS, Google Cloud Run, or any cloud platform. The pipeline runs automatically on startup.

**Step 1: Configure environment**
```bash
cp .env.cloud.sample .env
# Edit .env and set:
# - AUTO_RUN_PIPELINE=true
# - WEBSITE_URL=https://your-website.com
# - All required API keys (Cerebras, Zilliz, Neo4j, etc.)
```

**Step 2: Deploy with Docker**
```bash
# Build and run
docker-compose -f docker-compose.cloud.yml up -d

# The container will automatically:
# 1. Crawl the website
# 2. Process files to markdown
# 3. Save to vector database
# 4. Process graph data (3 phases)
# 5. Save to graph database
# 6. Start the Flask GraphRAG app on port 8080
```

**Access:** Open `http://localhost:8080` (or your cloud URL)

**For cloud platforms:**
- **Heroku**: Deploy with Heroku container registry
- **Google Cloud Run**: Deploy with `gcloud run deploy`
- **AWS**: Deploy to ECS/Fargate or Elastic Beanstalk

---

### Option 2: Manual Pipeline (For Development/Testing)

Run each step manually for more control and debugging.

## Prerequisites

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create `.env` file with the following settings:

#### Pipeline Automation (Docker Only)
```bash
# Set to 'true' to automatically run pipeline on container startup
AUTO_RUN_PIPELINE=true
WEBSITE_URL=https://your-website.com
```

#### Neo4j Configuration
```bash
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

#### Vector Database Configuration
```bash
# For Cloud (Zilliz)
VECTOR_DB_TYPE=cloud_zilliz
ZILLIZ_CLUSTER_ENDPOINT=https://your-cluster.zilliz.cloud
ZILLIZ_TOKEN=your_zilliz_token

# Or for Local (Milvus Lite)
VECTOR_DB_TYPE=local
```

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

## Manual Pipeline Workflow (Run in Order)

**Note:** If using Docker with `AUTO_RUN_PIPELINE=true`, these steps run automatically. Manual execution is only needed for local development or debugging.

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

## Query Applications

### Command Line Query
```bash
# Vector RAG Query
python 4_query.py

# GraphRAG Query 
python 4b_query_graph.py
```

### Flask Web Applications
```bash
# Simple Vector RAG Flask (Port 8080)
python app_flask.py

# GraphRAG Flask (Port 8080)
python app_flask_graph.py
```

### Chainlit Chat Applications
```bash
# GraphRAG Chainlit with Interactive UI (Port 8000)
chainlit run app_chainlit_graph.py
```
**Access:** Open `http://localhost:8000` in your browser

### Chainlit Graphrag Chat Applications
```bash
# GraphRAG GraphRag Chainlit with Interactive UI (Port 8000)
chainlit run app_chainlit_graph.py --port 8001
```
**Access:** Open `http://localhost:8001` in your browser


## Application Ports

| Application | Type | Port | Features |
|-------------|------|------|----------|
| app_flask.py | Flask | 8080 | Simple vector RAG |
| app_flask_graph.py | Flask | 8080 | Full GraphRAG|
| app_chainlit_graph.py | Chainlit | 8000 | Interactive chat with GraphRAG |
| app_chainlit_graph.py | Chainlit | 8001 | Interactive chat with GraphRAG
