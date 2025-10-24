FROM python:3.11-slim

# Build arguments for conditional installation
ARG INSTALL_OLLAMA=false
ARG INSTALL_LOCAL_VECTOR_DB=false

# Set working directory
WORKDIR /allycat

# Set environment variables - Cloud-first defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_RUN_ENV=cloud \
    VECTOR_DB_TYPE=cloud_zilliz


# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    netcat-traditional \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file - Use cloud-optimized by default
COPY requirements-docker-cloud.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker-cloud.txt

# Conditional: Install Ollama only if requested
RUN if [ "$INSTALL_OLLAMA" = "true" ]; then \
        echo "Installing Ollama for local LLM support..."; \
        curl -fsSL https://ollama.com/install.sh | sh; \
    else \
        echo "Skipping Ollama installation - using cloud LLM mode"; \
    fi

# Conditional: Install local vector DB dependencies
RUN if [ "$INSTALL_LOCAL_VECTOR_DB" = "true" ]; then \
        echo "Installing milvus-lite for local vector database..."; \
        pip install --no-cache-dir milvus-lite==2.4.11; \
    else \
        echo "Skipping local vector DB - using Zilliz Cloud"; \
    fi

# Copy project files
COPY . .
RUN chmod +x ./docker-startup.sh

# Cleanup unnecessary files
RUN rm -rf .env workspace/* __pycache__ *.pyc

# Expose the port for webapp
EXPOSE 8080  
# Conditional: Expose Ollama port only if installed
EXPOSE 11434

ENTRYPOINT ["./docker-startup.sh"]
