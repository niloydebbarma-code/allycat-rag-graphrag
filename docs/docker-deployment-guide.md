# AllyCAT GraphRAG - Docker Deployment Guide

## Overview

AllyCAT GraphRAG offers flexible deployment options to suit different use cases and infrastructure requirements. This guide covers three deployment modes optimized for various scenarios.

## Deployment Modes

### Mode Comparison

| Mode | Image Size | Best For | Configuration |
|------|-----------|----------|---------------|
| **Cloud** | ~800 MB | Production, Free Tier, Scalability | All services in cloud |
| **Hybrid** | ~1.5 GB | Data Privacy with Cloud LLM | Local vector DB, cloud LLM |
| **Local** | ~4+ GB | Offline, Development, Testing | All services local |

---

## Prerequisites

### System Requirements
- Docker 20.10+ or Docker Desktop
- 4GB+ RAM (8GB+ recommended for local mode)
- 5GB+ free disk space (15GB+ for local mode)
- Internet connection (for cloud mode)

### Required Accounts (Cloud Mode)
- At least one LLM provider:
  - [Cerebras](https://cerebras.ai/) (recommended, free tier available)
  - [Google AI Studio](https://makersuite.google.com/app/apikey) (for Gemini)
  - [Nebius AI Studio](https://studio.nebius.ai/)
- [Zilliz Cloud](https://cloud.zilliz.com/) (vector database)
- [Neo4j Aura](https://neo4j.com/cloud/aura/) (graph database)

---

## Quick Start

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/allycat.git
cd allycat
```

### Step 2: Choose Deployment Mode

**Cloud Mode with Automatic Pipeline (Recommended for Production):**

This mode automatically runs the complete GraphRAG pipeline on startup - perfect for Heroku, AWS, Google Cloud Run deployments.

```bash
# Copy cloud configuration template
cp .env.cloud.sample .env

# Edit .env and configure:
nano .env
# Set:
#   AUTO_RUN_PIPELINE=true
#   WEBSITE_URL=https://your-website.com
#   All API keys (Cerebras, Gemini, Zilliz, Neo4j)

# Build optimized cloud image
docker build \
  --build-arg INSTALL_OLLAMA=false \
  --build-arg INSTALL_LOCAL_VECTOR_DB=false \
  -t allycat-graphrag:cloud .

# Deploy using docker-compose
docker-compose -f docker-compose.cloud.yml up -d

# The container will automatically:
# 1. Crawl the website (from WEBSITE_URL)
# 2. Process files to markdown
# 3. Save to Zilliz Cloud vector database
# 4. Process graph data (extract entities, build communities, generate summaries)
# 5. Save to Neo4j Aura graph database
# 6. Start Flask GraphRAG app on port 8080

# Access the application
open http://localhost:8080
```

**Cloud Mode without Pipeline (Manual Data Preparation):**

If you want to prepare data manually before running the app:

```bash
# Copy cloud configuration template
cp .env.cloud.sample .env

# Edit .env and set AUTO_RUN_PIPELINE=false
nano .env

# Build and run
docker-compose -f docker-compose.cloud.yml up -d

# Then run pipeline manually (see Manual Pipeline section below)
```

**Local Mode (For offline or development use):**

```bash
# Copy local configuration template
cp .env.local.sample .env

# Edit .env if needed
# For automatic pipeline: set AUTO_RUN_PIPELINE=true
# For manual steps: set AUTO_RUN_PIPELINE=false (default)
nano .env

# Build full local image (includes Ollama)
docker build \
  --build-arg INSTALL_OLLAMA=true \
  --build-arg INSTALL_LOCAL_VECTOR_DB=true \
  -t allycat-graphrag:local .

# Deploy using docker-compose
docker-compose -f docker-compose.local.yml up -d

# Access the application
open http://localhost:8080
```

**Hybrid Mode (Cloud LLM + Local Vector DB):**

```bash
# Copy cloud template and modify
cp .env.cloud.sample .env

# Set VECTOR_DB_TYPE=local in .env
# For automatic pipeline: set AUTO_RUN_PIPELINE=true
# For manual steps: set AUTO_RUN_PIPELINE=false (default)
nano .env

# Build hybrid image
docker build \
  --build-arg INSTALL_OLLAMA=false \
  --build-arg INSTALL_LOCAL_VECTOR_DB=true \
  -t allycat-graphrag:hybrid .

# Deploy using docker-compose
docker-compose -f docker-compose.hybrid.yml up -d
```

---

## Configuration Guide

### Cloud Mode Configuration

Create a `.env` file with the following structure:

```env
# LLM Configuration
LLM_RUN_ENV=cloud
LLM_MODEL=cerebras/llama3.1-8b

# API Keys (provide at least one)
CEREBRAS_API_KEY=your_cerebras_api_key
GEMINI_API_KEY=your_gemini_api_key
NEBIUS_API_KEY=your_nebius_api_key

# Vector Database (Zilliz Cloud)
VECTOR_DB_TYPE=cloud_zilliz
ZILLIZ_CLUSTER_ENDPOINT=https://your-cluster.zilliz.cloud
ZILLIZ_TOKEN=your_zilliz_token

# Graph Database (Neo4j Aura)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

# Application Settings
APP_TYPE=flask_graph
WEBSITE_URL=https://your-website.com
```

See `.env.cloud.sample` for complete configuration options.

### Local Mode Configuration

Create a `.env` file for local deployment:

```env
# Pipeline Automation (optional)
AUTO_RUN_PIPELINE=false  # Set to 'true' for automatic pipeline on startup
WEBSITE_URL=https://your-website.com  # Required if AUTO_RUN_PIPELINE=true

# LLM Configuration
LLM_RUN_ENV=local_ollama
LLM_MODEL=ollama/gemma3:1b
OLLAMA_MODEL=gemma3:1b

# Vector Database (Local Milvus)
VECTOR_DB_TYPE=local

# Graph Database
# Option 1: Local Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_local_password

# Option 2: Neo4j Aura Cloud (recommended for graph extraction)
# NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io

# Graph extraction API (recommended even in local mode)
GEMINI_API_KEY=your_gemini_api_key

# Application Settings
APP_TYPE=flask_graph
```

See `.env.local.sample` for complete configuration options.

---

## Docker Build Options

### Build Arguments

The Dockerfile supports conditional installation of components:

```bash
# Minimal cloud build (recommended)
docker build \
  --build-arg INSTALL_OLLAMA=false \
  --build-arg INSTALL_LOCAL_VECTOR_DB=false \
  -t allycat-graphrag:cloud .

# Full local build
docker build \
  --build-arg INSTALL_OLLAMA=true \
  --build-arg INSTALL_LOCAL_VECTOR_DB=true \
  -t allycat-graphrag:local .

# Hybrid build (cloud LLM, local data)
docker build \
  --build-arg INSTALL_OLLAMA=false \
  --build-arg INSTALL_LOCAL_VECTOR_DB=true \
  -t allycat-graphrag:hybrid .
```

### Application Types

Set `APP_TYPE` in your `.env` file to choose the interface:

```env
# Flask-based GraphRAG interface (default)
APP_TYPE=flask_graph

# Chainlit interactive chat interface
APP_TYPE=chainlit_graph

# Simple Flask vector RAG (legacy)
APP_TYPE=flask
```

---

## Docker Compose Deployment

### Using Docker Compose Files

Each deployment mode has a dedicated docker-compose file:

```bash
# Cloud mode
docker-compose -f docker-compose.cloud.yml up -d

# Local mode
docker-compose -f docker-compose.local.yml up -d

# Hybrid mode
docker-compose -f docker-compose.hybrid.yml up -d
```

### Managing Containers

```bash
# View running containers
docker-compose -f docker-compose.cloud.yml ps

# View logs
docker-compose -f docker-compose.cloud.yml logs -f

# Stop containers
docker-compose -f docker-compose.cloud.yml down

# Restart containers
docker-compose -f docker-compose.cloud.yml restart

# Remove containers and volumes
docker-compose -f docker-compose.cloud.yml down -v
```

---

## Advanced Usage

### Direct Docker Run Commands

If you prefer not to use docker-compose:

**Cloud Mode:**

```bash
docker run -d \
  --name allycat-cloud \
  -p 8080:8080 \
  --env-file .env \
  -e LLM_RUN_ENV=cloud \
  -e VECTOR_DB_TYPE=cloud_zilliz \
  allycat-graphrag:cloud deploy
```

**Local Mode:**

```bash
docker run -d \
  --name allycat-local \
  -p 8080:8080 \
  -p 11434:11434 \
  --env-file .env \
  -e LLM_RUN_ENV=local_ollama \
  -e VECTOR_DB_TYPE=local \
  -v $(pwd)/workspace:/allycat/workspace \
  allycat-graphrag:local deploy
```

### Interactive Development Mode

For debugging or development:

```bash
# Start container without auto-deploy
docker run -it --env-file .env allycat-graphrag:cloud

# Inside container, you can manually start services:
# python3 app_flask_graph.py
# python3 4b_query_graph.py
```

### Volume Mounting

For local mode, mount the workspace directory to persist data:

```bash
docker run -d \
  -v $(pwd)/workspace:/allycat/workspace \
  -v $(pwd)/logs:/allycat/logs \
  --env-file .env \
  allycat-graphrag:local deploy
```

---

## Monitoring and Maintenance

### Health Checks

All docker-compose configurations include health checks:

```bash
# View container health status
docker ps

# Inspect health check details
docker inspect allycat-cloud | grep -A 10 Health
```

### Viewing Logs

```bash
# View application logs
docker logs -f allycat-cloud

# View last 100 lines
docker logs --tail 100 allycat-cloud

# View logs with timestamps
docker logs -t allycat-cloud
```

### Accessing Container Shell

```bash
# Access running container
docker exec -it allycat-cloud /bin/bash

# Check environment variables
docker exec allycat-cloud env

# View running processes
docker exec allycat-cloud ps aux
```

---

## Troubleshooting

### Common Issues

**Issue: Container fails to start**

```bash
# Check container logs
docker logs allycat-cloud

# Verify environment variables
docker exec allycat-cloud env | grep -E "LLM|VECTOR|NEO4J"

# Check if ports are already in use
lsof -i :8080
```

**Issue: API connection errors in cloud mode**

```bash
# Verify API keys are set
docker exec allycat-cloud env | grep -E "API_KEY|TOKEN"

# Test network connectivity
docker exec allycat-cloud curl -I https://api.cerebras.ai
```

**Issue: Ollama not starting in local mode**

```bash
# Check Ollama logs
docker exec allycat-local cat /allycat/ollama-serve.out

# Verify Ollama is running
docker exec allycat-local nc -z localhost 11434 && echo "Ollama is running"

# Manually start Ollama
docker exec -it allycat-local ollama serve
```

**Issue: Out of memory**

```bash
# Check Docker resource limits
docker stats allycat-local

# Increase Docker memory limit:
# Docker Desktop > Settings > Resources > Memory
# Recommended: 8GB+ for local mode, 4GB+ for cloud mode
```

**Issue: Vector database connection failed**

```bash
# For cloud mode: verify Zilliz credentials
docker exec allycat-cloud env | grep ZILLIZ

# For local mode: check workspace directory permissions
ls -la workspace/
```

---

## Performance Optimization

### Resource Allocation

**Cloud Mode:**
- Minimum: 2 CPU cores, 4GB RAM
- Recommended: 4 CPU cores, 8GB RAM

**Local Mode:**
- Minimum: 4 CPU cores, 8GB RAM
- Recommended: 8 CPU cores, 16GB RAM

### Image Size Optimization

Current image sizes:

| Mode | Actual Size | Optimization |
|------|-------------|--------------|
| Cloud | ~800 MB | Production-ready |
| Hybrid | ~1.5 GB | Balanced |
| Local | ~2.9 GB | Full features |

To further optimize:

```bash
# Remove unused images
docker image prune -a

# Use multi-stage builds (already implemented)
# Minimize layers (already optimized)
```

---

## Migration from Previous Versions

### Upgrading from Old Configuration

If you're upgrading from a previous version:

1. **Backup existing data:**

```bash
# Backup workspace and configuration
cp -r workspace workspace.backup
cp .env .env.backup
```

2. **Update configuration:**

```bash
# Copy new sample configuration
cp .env.cloud.sample .env

# Migrate your API keys from .env.backup to new .env
```

3. **Rebuild with new Dockerfile:**

```bash
# Remove old image
docker rmi allycat-graphrag:old

# Build new image
docker build --build-arg INSTALL_OLLAMA=false -t allycat-graphrag:cloud .
```

4. **Deploy with new configuration:**

```bash
# Stop old container
docker stop allycat-old
docker rm allycat-old

# Start new container
docker-compose -f docker-compose.cloud.yml up -d
```

### Configuration Changes

Key changes in the new version:

- Default mode changed from `local_ollama` to `cloud`
- Vector database defaults to `cloud_zilliz` instead of `local`
- Ollama installation is now conditional
- New requirements file: `requirements-docker-cloud.txt`
- **Automatic pipeline**: Set `AUTO_RUN_PIPELINE=true` to run the complete pipeline on startup

---

## Cloud Platform Deployment

### Deploying to Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create new app
heroku create your-app-name

# Set environment variables
heroku config:set AUTO_RUN_PIPELINE=true
heroku config:set WEBSITE_URL=https://your-website.com
heroku config:set LLM_RUN_ENV=cloud
heroku config:set VECTOR_DB_TYPE=cloud_zilliz
heroku config:set CEREBRAS_API_KEY=your_key
heroku config:set ZILLIZ_CLUSTER_ENDPOINT=https://your-cluster.zilliz.cloud
heroku config:set ZILLIZ_TOKEN=your_token
heroku config:set NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
heroku config:set NEO4J_USERNAME=neo4j
heroku config:set NEO4J_PASSWORD=your_password
heroku config:set GEMINI_API_KEY=your_gemini_key

# Deploy using container registry
heroku container:login
heroku container:push web --arg INSTALL_OLLAMA=false,INSTALL_LOCAL_VECTOR_DB=false
heroku container:release web

# Open your app
heroku open
```

### Deploying to Google Cloud Run

```bash
# Set project
gcloud config set project your-project-id

# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project-id/allycat-graphrag \
  --build-arg INSTALL_OLLAMA=false \
  --build-arg INSTALL_LOCAL_VECTOR_DB=false

# Deploy to Cloud Run
gcloud run deploy allycat-graphrag \
  --image gcr.io/your-project-id/allycat-graphrag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars AUTO_RUN_PIPELINE=true,WEBSITE_URL=https://your-website.com,LLM_RUN_ENV=cloud,VECTOR_DB_TYPE=cloud_zilliz \
  --set-secrets CEREBRAS_API_KEY=cerebras_key:latest,ZILLIZ_TOKEN=zilliz_token:latest,NEO4J_PASSWORD=neo4j_password:latest

# Get the URL
gcloud run services describe allycat-graphrag --region us-central1 --format 'value(status.url)'
```

### Deploying to AWS (Elastic Container Service)

```bash
# Create ECR repository
aws ecr create-repository --repository-name allycat-graphrag

# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com
docker build --build-arg INSTALL_OLLAMA=false -t allycat-graphrag:cloud .
docker tag allycat-graphrag:cloud your-account-id.dkr.ecr.us-east-1.amazonaws.com/allycat-graphrag:latest
docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/allycat-graphrag:latest

# Create task definition with environment variables:
# - AUTO_RUN_PIPELINE=true
# - WEBSITE_URL=https://your-website.com
# - All API keys as secrets

# Deploy to ECS Fargate
# (Use AWS Console or CLI to create service)
```

**Important Notes for Cloud Deployment:**
- Set `AUTO_RUN_PIPELINE=true` to run the pipeline automatically on container startup
- Ensure all API keys are set as environment variables or secrets
- The pipeline will run once on startup, then the app will be available
- Pipeline execution may take 5-15 minutes depending on website size
- Use cloud services (Zilliz, Neo4j Aura) for best results

---

## Security Best Practices

### API Key Management

1. **Never commit `.env` files to version control**

```bash
# Ensure .env is in .gitignore
echo ".env" >> .gitignore
```

2. **Use environment-specific configurations**

```bash
# Development
cp .env.cloud.sample .env.dev

# Production
cp .env.cloud.sample .env.prod
```

3. **Rotate API keys regularly**

### Network Security

For production deployments:

```bash
# Use reverse proxy (nginx, traefik)
# Enable HTTPS
# Restrict network access
# Use Docker secrets for sensitive data
```

---

## Production Deployment

### Recommended Setup for Production

1. **Use cloud mode for optimal performance**
2. **Enable health checks and monitoring**
3. **Set up logging aggregation**
4. **Configure auto-restart policies**
5. **Use managed services for databases**

### Example Production docker-compose.yml

```yaml
version: '3.8'

services:
  allycat:
    image: allycat-graphrag:cloud
    restart: always
    ports:
      - "8080:8080"
    env_file:
      - .env.prod
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Additional Resources

### Documentation
- [Configuration Guide](configuration.md) - Detailed configuration options
- [Customization Guide](customizing-allycat.md) - Extending functionality
- [LLM Setup Guide](llms-remote.md) - LLM provider configuration
- [Local LLM Guide](llm-local.md) - Running Ollama locally

### Support
- GitHub Issues: Report bugs or request features
- Documentation: Comprehensive guides in `/docs`
- Examples: Sample configurations in repository

### Contributing
- Fork the repository
- Create feature branch
- Submit pull request with clear description

---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## Changelog

### v2.0 (Current)
- Cloud-first architecture
- Conditional Ollama installation
- 80% image size reduction
- Multi-mode support (cloud/local/hybrid)
- Complete GraphRAG dependencies
- Docker Compose support
- Health checks and monitoring

### v1.0 (Previous)
- Local-first architecture
- Always included Ollama
- Basic Docker support

