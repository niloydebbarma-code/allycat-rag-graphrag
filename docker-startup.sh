#!/bin/bash

echo "=== AllyCAT GraphRAG Docker Startup ==="

# Check deployment mode from environment
LLM_MODE=${LLM_RUN_ENV:-cloud}
VECTOR_MODE=${VECTOR_DB_TYPE:-cloud_zilliz}

echo "LLM Mode: $LLM_MODE"
echo "Vector DB Mode: $VECTOR_MODE"

# Conditional: Start Ollama only if in local mode
if [ "$LLM_MODE" = "local_ollama" ]; then
    echo "Starting Ollama in local mode..."
    
    # Define OLLAMA_MODELS dir
    if [ -z "$OLLAMA_MODELS" ]; then
        export OLLAMA_MODELS=/allycat/workspace/ollama
    fi
    
    echo "Env variables for OLLAMA:"
    env | grep OLLAMA
    
    # Start ollama
    ollama_model=${OLLAMA_MODEL:-gemma3:1b}
    echo "Starting Ollama server..."
    ollama serve > /allycat/ollama-serve.out 2>&1 &
    
    # Wait for ollama to start
    while ! nc -z localhost 11434; do
        sleep 1
    done
    echo "✅ Ollama started on port 11434"
    
    # Only download the model if we are in DEPLOY mode
    if [ "$1" == "deploy" ]; then
        echo "Downloading Ollama model: $ollama_model"
        ollama pull $ollama_model
        echo "✅ Ollama model downloaded: $ollama_model"
    fi
else
    echo "✅ Using cloud LLM mode - Ollama not started"
fi

# Conditional: Setup local vector DB only if needed
if [ "$VECTOR_MODE" = "local" ]; then
    echo "Setting up local Milvus vector database..."
    mkdir -p /allycat/workspace
    echo "✅ Local vector database directory created"
else
    echo "✅ Using Zilliz Cloud for vector database"
fi

# Run GraphRAG pipeline if AUTO_RUN_PIPELINE is enabled and in deploy mode
if [ "$1" == "deploy" ] && [ "${AUTO_RUN_PIPELINE:-false}" = "true" ]; then
    echo ""
    echo "=== Running GraphRAG Pipeline Automatically ==="
    echo ""
    
    # Step 1: Crawl website
    if [ -n "$WEBSITE_URL" ]; then
        echo "Step 1/5: Crawling website: $WEBSITE_URL"
        python3 1_crawl_site.py || echo "⚠️  Warning: Crawl failed, continuing..."
        echo "✅ Step 1 complete"
        echo ""
    else
        echo "⚠️  Skipping crawl - WEBSITE_URL not set"
    fi
    
    # Step 2: Process files to markdown
    echo "Step 2/5: Processing files to markdown..."
    python3 2_process_files.py || echo "⚠️  Warning: Processing failed, continuing..."
    echo "✅ Step 2 complete"
    echo ""
    
    # Step 3: Save to vector database
    echo "Step 3/5: Saving to vector database..."
    if [ "$VECTOR_MODE" = "cloud_zilliz" ]; then
        python3 3_save_to_vector_db_zilliz.py || echo "⚠️  Warning: Vector DB save failed, continuing..."
    else
        python3 3_save_to_vector_db.py || echo "⚠️  Warning: Vector DB save failed, continuing..."
    fi
    echo "✅ Step 3 complete"
    echo ""
    
    # Step 4: Process graph data (3 phases)
    echo "Step 4/5: Processing graph data (3 phases)..."
    echo "  Phase 1: Extracting entities and relationships..."
    python3 2b_process_graph_phase1.py || echo "⚠️  Warning: Phase 1 failed, continuing..."
    echo "  Phase 2: Building communities..."
    python3 2b_process_graph_phase2.py || echo "⚠️  Warning: Phase 2 failed, continuing..."
    echo "  Phase 3: Generating community summaries..."
    python3 2b_process_graph_phase3.py || echo "⚠️  Warning: Phase 3 failed, continuing..."
    echo "✅ Step 4 complete"
    echo ""
    
    # Step 5: Save to graph database
    echo "Step 5/5: Saving to graph database..."
    python3 3b_save_to_graph_db.py || echo "⚠️  Warning: Graph DB save failed, continuing..."
    echo "✅ Step 5 complete"
    echo ""
    
    echo "=== ✅ Pipeline Complete - Starting Application ==="
    echo ""
fi

# Start the appropriate web application
APP_TYPE=${APP_TYPE:-flask_graph}

if [ "$1" == "deploy" ]; then
    echo "In deploy mode..."
    
    case $APP_TYPE in
        "flask_graph")
            echo "Starting Flask GraphRAG app on port 8080..."
            python3 app_flask_graph.py
            ;;
        "chainlit_graph")
            echo "Starting Chainlit GraphRAG app on port 8080..."
            chainlit run app_chainlit_graph.py --host 0.0.0.0 --port 8080
            ;;
        "flask")
            echo "Starting Flask Vector RAG app on port 8080..."
            python3 app_flask.py
            ;;
        *)
            echo "Starting default Flask GraphRAG app on port 8080..."
            python3 app_flask_graph.py
            ;;
    esac
elif [ "$1" == "pipeline" ]; then
    echo "Running GraphRAG pipeline..."
    chmod +x ./run_pipeline.sh
    ./run_pipeline.sh
else
    echo "Not in deploy mode, entering interactive shell."
    echo ""
    echo "Available commands:"
    echo "  ./run_pipeline.sh                - Run complete GraphRAG pipeline"
    echo "  python3 app_flask_graph.py       - Start Flask GraphRAG app"
    echo "  chainlit run app_chainlit_graph.py - Start Chainlit GraphRAG app"
    echo "  python3 app_flask.py             - Start Flask Vector RAG app"
    if [ "$LLM_MODE" = "local_ollama" ]; then
        echo "  ollama pull $ollama_model        - Download Ollama model"
    fi
    echo ""
    echo "To run the pipeline: ./run_pipeline.sh"
    echo "To start the app: python3 app_flask_graph.py"
    /bin/bash
fi
