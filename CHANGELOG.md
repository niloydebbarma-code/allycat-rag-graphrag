# AllyCat Changelog

All notable technical changes to AllyCat GraphRAG will be documented in this file.

## [Unreleased]

### Added

#### GraphRAG Implementation
- **GraphRAG Core System**: Implemented Microsoft GraphRAG-inspired architecture
  - Entity extraction and relationship mapping from documents
  - Community detection algorithms for knowledge graph clustering
  - Multi-phase graph processing pipeline (phase 1: entities/relationships, phase 2: communities, phase 3: summaries)
  - Graph-based query system with hierarchical summarization
  - Neo4j integration for graph database storage (`3b_save_to_graph_db.py`)
  - Graph query functions in `query_graph_functions/` directory
  - Dual RAG modes: Traditional Vector RAG + Advanced GraphRAG
  - **Note**: More improvements planned based on [Microsoft GraphRAG Project](https://github.com/microsoft/graphrag)

#### LLM Provider Support
- **Cerebras API Integration**: Added support for Cerebras ultra-fast inference
- **Google Gemini API Integration**: Added support for Google's Gemini models
- **LiteLLM Framework**: Implemented `litellm_patch.py` for unified LLM API interface
  - Supports multiple providers: OpenAI, Replicate, Nebius, Cerebras, Gemini, Anthropic, and more
  - Simplified provider switching via environment variables

#### Database Solutions
- **Zilliz Cloud Integration**: Added cloud-based vector database support
  - Implemented `3_save_to_vector_db_zilliz.py` for Zilliz Cloud
  - Cloud vector database eliminates need for local Milvus server
  - Configurable via `VECTOR_DB_TYPE` environment variable
- **Neo4j Graph Database**: Integrated for GraphRAG knowledge graph storage
  - Stores entities, relationships, and community structures
  - Enables complex graph traversal queries

#### Docker Deployment System
- **Three Deployment Modes**: Flexible deployment configurations
  - **Cloud Mode** (`docker-compose.cloud.yml`): Cloud LLM + Cloud Zilliz vector DB
  - **Hybrid Mode** (`docker-compose.hybrid.yml`): Cloud LLM + Local Milvus
  - **Local Mode** (`docker-compose.local.yml`): Local Ollama + Local Milvus
- **Automated Deployment Script**: `docker-startup.sh` orchestrates full deployment
  - Conditional service startup based on deployment mode
  - Automatic Ollama model download for local mode
  - Smart service detection and initialization

#### Automatic Pipeline Execution
- **End-to-End Automation**: Single command deployment from crawling to running application
  - Automatic website crawling when `WEBSITE_URL` is set
  - Sequential pipeline execution: crawl → process → vector DB → graph processing → graph DB
  - Automatic application startup after pipeline completion
  - Controlled via `AUTO_RUN_PIPELINE` environment variable
  - **User Action Required**: Only set environment variables in `.env` file

#### Document Processing Improvements
- **HTML/HTM Processing**: Switched to `html2text` library for better HTML parsing
  - Improved markdown conversion quality
  - Better handling of HTML structure and formatting
  - **Resolves**: [Issue #50](https://github.com/The-AI-Alliance/allycat/issues/50)
- **PDF Processing**: Integrated `docling` library for advanced PDF parsing
  - High-quality PDF to markdown conversion
  - Preserves document structure and formatting
  - Handles complex PDF layouts

#### Port Management System
- **Multiple Dynamic Ports**: Flexible port configuration for different services
  - `FLASK_GRAPH_PORT=8080` - Flask GraphRAG application
  - `FLASK_VECTOR_PORT=8081` - Flask Vector RAG application
  - `CHAINLIT_GRAPH_PORT=8083` - Chainlit GraphRAG application
  - `CHAINLIT_VECTOR_PORT=8082` - Chainlit Vector RAG application
  - `DOCKER_PORT` - Host machine port mapping
  - `DOCKER_APP_PORT=8080` - Internal container port
  - `OLLAMA_PORT=11434` - Ollama server port (local mode)
- **Smart Port Routing**: Automatic port selection based on `APP_TYPE` environment variable
  - Supports: `flask_graph`, `flask`, `chainlit_graph`, `chainlit`

#### Memory Optimization System
- **CLEANUP_PIPELINE_DEPS Feature**: Post-pipeline dependency cleanup
  - Created `requirements-runtime.txt` (~300 MB) - minimal packages for running Flask GraphRAG app
  - Created `requirements-build.txt` (~500 MB) - pipeline-only packages that can be removed
  - Created `cleanup_pipeline_deps.sh` - automated cleanup script
  - Integrated cleanup into `docker-startup.sh` with conditional execution
  - Added `CLEANUP_PIPELINE_DEPS` configuration to all `.env` sample files
  - Created comprehensive technical documentation in `docs/docker-memory-optimization.md`
  - Updated `docs/docker-deployment-guide.md` with memory optimization section
  - **Benefits**: Reduces container RAM from ~800 MB to ~300 MB, enabling 1GB deployments
  - **Cost Savings**: DigitalOcean 1GB ($12/mo) vs 2GB ($25/mo) = $156/year savings (52% reduction)

### Changed
- **Chainlit Port Configuration**: Reverted Chainlit apps to use default port behavior
  - Removed custom port configuration code from Python files
  - Chainlit now uses default port 8000 for native Python execution
  - Docker deployments use custom ports via `--port` flag in `docker-startup.sh`
  - Updated documentation in `docs/graphrag-demo/Setup.md`, `docs/configuration.md`, and `my_config.py`
  - Native Python: `chainlit run app_chainlit_graph.py` (port 8000) or with custom `--port 8083`
  - Docker: Custom ports 8082 (vector) and 8083 (graph) configured via environment variables


### Fixed
- **HTML/HTM File Processing**: Fixed HTML parsing issues ([Issue #50](https://github.com/The-AI-Alliance/allycat/issues/50))
  - Switched from previous parser to `html2text` library
  - Improved markdown conversion quality and reliability
- **Pipeline Error Handling**: Confirmed robust error handling with `|| echo "Warning..."` pattern
  - Cleanup script runs even if pipeline steps fail
  - Application starts successfully regardless of pipeline completion status

## [Previous Versions]

### 2025-07-14: Major Update

#### Added
- **Robust Web Crawler** ([#31](https://github.com/The-AI-Alliance/allycat/issues/31))
  - Complete rewrite of web crawler
  - More robust handling of edge cases
  - Support for multiple file types (not just text/html)
  - Correct handling of anchor tags (`a.html#news`) in HTML files
  - Customizable pause between requests to avoid hammering webservers
  - Fixed issue with repeatedly downloading same content

- **LiteLLM Integration** ([#34](https://github.com/The-AI-Alliance/allycat/issues/34))
  - Unified LLM backend support replacing Replicate and Ollama setup
  - Seamless access to local LLMs (using Ollama) and cloud inference providers
  - Support for providers: Nebius, Replicate, and more
  - Significantly simplified LLM configuration
  - Added `python-dotenv` for environment variable management

- **Expanded File Type Support** ([#37](https://github.com/The-AI-Alliance/allycat/issues/37))
  - Support for PDF, DOCX, and other popular file types (previously only HTML)
  - Integration with [Docling](https://github.com/docling-project/docling) for file processing
  - Fixed issue with PDF downloads ([#35](https://github.com/The-AI-Alliance/allycat/issues/35))
  - Processing all downloaded file types
  - Updated process_file script

- **UV Package Manager Support** ([#26](https://github.com/The-AI-Alliance/allycat/issues/26))
  - Added [uv](https://docs.astral.sh/uv/) project structure
  - Updated documentation for uv
  - Continued support for `requirements.txt` and other package managers

- **Better Config Management** ([#19](https://github.com/The-AI-Alliance/allycat/issues/19))
  - User configuration via `.env` file
  - Simplified config management
  - Easier experimentation without code changes
  - Documented configuration options
  - Updated env.sample file with settings

- **Metrics Collection**: Added metrics collection scripts and issue templates

#### Changed
- **Chainlit App Updates** ([#38](https://github.com/The-AI-Alliance/allycat/issues/38))
  - Updated Chainlit application
  - Customizable starter prompts
- **Logo Updates** ([#39](https://github.com/The-AI-Alliance/allycat/issues/39))
  - Updated logo to AllyCAT
- **App Naming**: Changed Flask and Chainlit app names for clarity
- Code cleanup improvements
- Documentation updates across the project

### 2025-05: Chainlit Integration

#### Added
- **Chainlit Chat Interface** ([#17](https://github.com/The-AI-Alliance/allycat/issues/17))
  - Introduced Chainlit-based chat interface as alternative to Flask UI
  - Improved chat UI experience

#### Changed
- Updated README with license and issues links

### 2025-04: Dockerization and Local LLM Support

#### Added
- **Docker Support**: Complete dockerization of the application
  - Docker deployment configurations
  - Updated Google Cloud deployment guide
  - Comprehensive Docker documentation (`running-in-docker.md`)
- **Ollama Integration**: Local LLM support with Ollama
  - Local LLM configuration and setup
  - Local Jupyter Lab support
  - Small tweaks to local LLM config
- **Python Scripts**: Added Python script versions of notebooks

#### Changed
- Updated deployment documentation
- Native running documentation (`running-natively.md`)

### 2025-03: Database and LLM Updates

#### Added
- **Weaviate Database**: Added Weaviate vector database support
- **Local LLM Support**: Initial local LLM integration

#### Changed
- **LLM Switch**: Changed from initial LLM to Llama
- Added logo and GitHub link to UI
- README and deploy guide updates

### 2025-02: Initial Release - AllyCAT (formerly AllyChat)

#### Added
- **Initial Vector RAG System**: First version of AllyCAT
  - Basic RAG implementation
  - Vector database for document storage and retrieval
  - Query system for document Q&A
- **Flask Web Interface**: Web-based chat interface
- **Basic Crawling**: Initial website crawling functionality
- **Document Processing**: Basic document processing pipeline



---
