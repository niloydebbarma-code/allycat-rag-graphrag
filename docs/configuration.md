# Allycat Configuration

Here are all the config parameters for Allycat.

The default config values are in [my_config.py](../my_config.py)

Some config values can be overridden in `.env` file

## Port Configuration

AllyCAT supports multiple application types, each with configurable ports:

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| FLASK_GRAPH_PORT | 8080 | Port for GraphRAG Flask app (`app_flask_graph.py`) - Auto-configured via MY_CONFIG <br> Override in `.env` file |
| FLASK_VECTOR_PORT | 8081 | Port for vector-only RAG Flask app (`app_flask.py`) - Auto-configured via MY_CONFIG <br> Override in `.env` file |
| CHAINLIT_GRAPH_PORT | 8083 | Port for GraphRAG Chainlit app (`app_chainlit_graph.py`) - Docker only (native: 8000) <br> Override in `.env` file |
| CHAINLIT_VECTOR_PORT | 8082 | Port for vector-only RAG Chainlit app (`app_chainlit.py`) - Docker only (native: 8000) <br> Override in `.env` file |
| DOCKER_PORT | 8080 | External Docker exposed port (host side) <br> Override in `.env` file |
| DOCKER_APP_PORT | 8080 | Internal container port (container side) <br> Should match your selected APP_TYPE port <br> Override in `.env` file |
| OLLAMA_PORT | 11434 | Ollama server port (for local LLM mode) <br> Override in `.env` file |

**Important Notes:**
- **Flask apps:** Automatically use ports from `MY_CONFIG` (no command-line flags needed)
- **Chainlit apps:** Native default port is **8000**. Custom ports (8082, 8083) are for Docker deployments only
  - Native Python: `chainlit run app_chainlit_graph.py` → `http://localhost:8000`
  - With custom port: `chainlit run app_chainlit_graph.py --port 8083` → `http://localhost:8083`

**Docker Port Mapping:**
- `DOCKER_PORT` → External port users access (e.g., `http://localhost:8080`)
- `DOCKER_APP_PORT` → Internal port the app runs on inside container
- Set `DOCKER_APP_PORT` to match your chosen `APP_TYPE`:
  - `APP_TYPE=flask_graph` → `DOCKER_APP_PORT=8080` (or your FLASK_GRAPH_PORT)
  - `APP_TYPE=flask` → `DOCKER_APP_PORT=8081` (or your FLASK_VECTOR_PORT)
  - `APP_TYPE=chainlit_graph` → `DOCKER_APP_PORT=8083` (or your CHAINLIT_GRAPH_PORT)
  - `APP_TYPE=chainlit` → `DOCKER_APP_PORT=8082` (or your CHAINLIT_VECTOR_PORT)

## Crawl Configurations

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| CRAWL_MAX_DOWNLOADS | 100           | How many files to download.  Override this by specifying `--max-downloads` to script `1_crawl_site.py` |
| CRAWL_MAX_DEPTH     | 3             | How many levels to crawl.  Override this by specifying `--depth` to script `1_crawl_site.py`           |
| WAITTIME_BETWEEN_REQUESTS     | 0.1             | How long to wait before making download request (in seconds). <br> Override in `.env` file       |

## Workspace

This is where the artifacts are saved

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| WORKSPACE_DIR | `workspace` (native) <br> `/allycat/workspace` (Docker) | Where files/models/databases are stored. <br> **Native execution**: Use relative path `workspace` <br> **Docker**: Use absolute path `/allycat/workspace` <br> Override in `.env` file |

## Embeddings

We support open-source embeddings.  You can find them at [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| EMBEDDING_MODEL | `ibm-granite/granite-embedding-30m-english`           | Embedding model to use. <br> Override in `.env` file |
| EMBEDDING_LENGTH | 384           | Embedding vector size. <br> Override in `.env` file. <br>  Must match `EMBEDDING_MODEL` setting above |
| CHUNK_SIZE | 512           | Chunk size  <br> Override in `.env` file. |
| CHUNK_OVERLAP | 20           | Chunk overlap  <br> Override in `.env` file. |

## LLM Configuration

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| LLM_RUN_ENV | cloud | LLM runtime environment. Options: `cloud` or `local_ollama` <br> Override in `.env` file |
| LLM_MODEL | `cerebras/llama3.1-8b` | LLM model to use. <br> Cloud options: `cerebras/llama3.1-8b`, `gemini/gemini-1.5-flash`, `nebius/meta-llama/Meta-Llama-3.1-8B-Instruct` <br> Local options: `ollama/gemma3:1b` <br> Override in `.env` file |
| CEREBRAS_API_KEY | None | Cerebras API key for cloud LLM and graph extraction <br> Get free key at: https://cerebras.ai/ <br> Override in `.env` file |
| GEMINI_API_KEY | None | Google Gemini API key for cloud LLM and graph extraction <br> Get free key at: https://aistudio.google.com/ <br> Override in `.env` file |
| NEBIUS_API_KEY | None | Nebius API key for cloud LLM <br> Get key at: https://studio.nebius.ai/ <br> Override in `.env` file |
| REPLICATE_API_TOKEN | None | Replicate API token (optional) <br> Override in `.env` file |

## Vector Database Configuration

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| VECTOR_DB_TYPE | `cloud_zilliz` | Vector database type. Options: `cloud_zilliz` (recommended) or `local` <br> Override in `.env` file |
| ZILLIZ_CLUSTER_ENDPOINT | None | Zilliz Cloud cluster endpoint URL <br> Get free cluster at: https://cloud.zilliz.com/ <br> Override in `.env` file |
| ZILLIZ_TOKEN | None | Zilliz Cloud authentication token <br> Override in `.env` file |
| COLLECTION_NAME | `pages` | Milvus collection name <br> Cannot be overridden in `.env` file |

**Local Milvus (when VECTOR_DB_TYPE=local):**
- Vector-only RAG: `workspace/vector_only_milvus.db`
- Hybrid GraphRAG: `workspace/hybrid_graph_milvus.db`

## Graph Database Configuration (Neo4j)

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| NEO4J_URI | None | Neo4j connection URI <br> Cloud: `neo4j+s://your-instance.databases.neo4j.io` <br> Local: `bolt://localhost:7687` <br> Get free cloud instance at: https://neo4j.com/cloud/aura/ <br> Override in `.env` file |
| NEO4J_USERNAME | None | Neo4j username (usually `neo4j`) <br> Override in `.env` file |
| NEO4J_PASSWORD | None | Neo4j password <br> Override in `.env` file |
| NEO4J_DATABASE | None | Neo4j database name (usually `neo4j`) <br> Override in `.env` file |

## Graph Extraction Configuration

These parameters control entity and relationship extraction for GraphRAG:

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| GRAPH_MIN_ENTITIES | 5 | Minimum number of entities to extract per chunk <br> Override in `.env` file |
| GRAPH_MAX_ENTITIES | 15 | Maximum number of entities to extract per chunk <br> Override in `.env` file |
| GRAPH_MIN_RELATIONSHIPS | 3 | Minimum number of relationships to extract per chunk <br> Override in `.env` file |
| GRAPH_MAX_RELATIONSHIPS | 8 | Maximum number of relationships to extract per chunk <br> Override in `.env` file |
| GRAPH_MIN_CONFIDENCE | 0.8 | Minimum confidence score for extracted entities/relationships (0.0-1.0) <br> Override in `.env` file |
| GRAPH_MAX_CONTENT_CHARS | 12000 | Maximum characters to process per chunk for graph extraction <br> Override in `.env` file |
| GRAPH_SENTENCE_BOUNDARY_RATIO | 0.7 | Ratio for determining sentence boundaries during chunking <br> Override in `.env` file |

## Application Settings

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| APP_TYPE | `flask_graph` | Application type for Docker deployment <br> Options: `flask_graph`, `flask`, `chainlit_graph`, `chainlit` <br> Override in `.env` file |
| AUTO_RUN_PIPELINE | false | Automatically run complete pipeline on container startup (Docker only) <br> Set to `true` for cloud deployments <br> Override in `.env` file |
| UI_STARTER_PROMPTS | (see below) | Pipe-separated list of starter prompts for the UI <br> Default: `What is this website? \| What are upcoming events? \| Who are some of the partners?` <br> Override in `.env` file |
| HF_ENDPOINT | `https://thealliance.ai/` | Hugging Face endpoint (for custom mirrors or Chinese users) <br> Override in `.env` file |

## Directory Structure

| Configuration       | Default Value | Description                                                                                            |
|---------------------|---------------|--------------------------------------------------------------------------------------------------------|
| WORKSPACE_DIR | `workspace` (native) <br> `/allycat/workspace` (Docker) | Root directory for all workspace files. <br> **Native execution**: Use relative path `workspace` <br> **Docker**: Use absolute path `/allycat/workspace` <br> Override in `.env` file |
| CRAWL_DIR | `workspace/crawled` | Directory for crawled HTML files <br> Cannot be overridden |
| PROCESSED_DATA_DIR | `workspace/processed` | Directory for processed documents <br> Cannot be overridden |
| GRAPH_DATA_DIR | `workspace/graph_data` | Directory for graph data files <br> Cannot be overridden |
| LLAMA_INDEX_CACHE_DIR | `workspace/llama_index_cache` | Cache directory for LlamaIndex models <br> Cannot be overridden |

## Environment Variable Priority

Configuration values are loaded in the following order (highest priority first):

1. **Environment variables** set in your shell or cloud platform
2. **`.env` file** in the project root
3. **Default values** in `my_config.py`

Example:
```bash
# In .env file
FLASK_GRAPH_PORT=8080
LLM_MODEL=gemini/gemini-1.5-flash
VECTOR_DB_TYPE=cloud_zilliz
```

## Quick Reference

### Minimal Cloud Configuration

For a minimal cloud deployment, set these in your `.env` file:

```bash
# LLM
LLM_RUN_ENV=cloud
LLM_MODEL=cerebras/llama3.1-8b
CEREBRAS_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Vector Database
VECTOR_DB_TYPE=cloud_zilliz
ZILLIZ_CLUSTER_ENDPOINT=https://your-cluster.zilliz.cloud
ZILLIZ_TOKEN=your_token_here

# Graph Database
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Website to crawl
WEBSITE_URL=https://your-website.com
AUTO_RUN_PIPELINE=true
```

### Minimal Local Configuration

For local development with all local services:

```bash
# LLM
LLM_RUN_ENV=local_ollama
LLM_MODEL=ollama/gemma3:1b

# Vector Database
VECTOR_DB_TYPE=local

# Graph Database (use cloud Neo4j even for local dev)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Gemini for graph extraction
GEMINI_API_KEY=your_key_here

# Website
WEBSITE_URL=https://your-website.com
```

## See Also

- [Environment Sample Files](../env.sample.txt) - Complete example configurations
- [Running Natively Guide](running-natively.md) - Setup for native Python
- [Running in Docker Guide](running-in-docker.md) - Setup for Docker
- [Docker Deployment Guide](docker-deployment-guide.md) - Production deployment
- [LLM Setup Guide](llms-remote.md) - Cloud LLM configuration
- [Local LLM Guide](llm-local.md) - Ollama setup
