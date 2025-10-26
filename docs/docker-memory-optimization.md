# Docker Memory Optimization Guide

## Problem Statement

Docker containers with full AllyCAT dependencies consume approximately 700-800 MB RAM, causing:

- OOM (Out of Memory) errors on 1 GB containers
- Requirement for 2 GB RAM instances ($25/month vs $12/month on DigitalOcean)
- Higher deployment costs

**Solution:** Remove pipeline dependencies after pipeline completes, reducing RAM to approximately 300-450 MB.

## Cost-Benefit Analysis

| Configuration | RAM Required | DigitalOcean Cost | Savings |
|--------------|--------------|-------------------|---------|
| Full Dependencies | 2 GB | $25/month | - |
| After Cleanup | 1 GB | $12/month | $156/year |
| + Cloud Embeddings | 1 GB | $12/month + API | 52% reduction |

## Implementation Strategy

### Phase 1: Dependency Cleanup

**Files Created:**
1. `requirements-runtime.txt` - Minimal packages for Flask GraphRAG app (~300 MB)
2. `requirements-build.txt` - Pipeline-only packages to remove (~500 MB)
3. `cleanup_pipeline_deps.sh` - Automated cleanup script
4. Updated `docker-startup.sh` - Auto-cleanup after pipeline

**How It Works:**
```
1. Docker starts → Install full requirements.txt (~800 MB)
2. AUTO_RUN_PIPELINE=true → Run pipeline (20-30 min)
3. Pipeline completes → Run cleanup_pipeline_deps.sh
4. Remove heavy packages → Save ~350-500 MB
5. Flask app runs → Only needs runtime deps (~300 MB)
6. Redeploy → Automatically reinstalls & repeats cycle
```

## Package Categories

### Runtime Dependencies (Required - ~300 MB)

Essential for Flask GraphRAG App:
- `flask` - Web framework
- `nest_asyncio` - Async support
- `neo4j` - Graph database client
- `pymilvus` - Vector database client
- `litellm` - LLM integration
- `llama-index` + plugins - Query engine
- `networkx` - Graph queries
- `openai`, `google-generativeai` - LLM APIs
- `orjson`, `json-repair` - JSON handling
- `python-dotenv`, `pandas`, `humanfriendly` - Utilities

Heavy but Required:
- `torch` (~250 MB) - For local embeddings (Granite model) - Future: Remove with cloud embeddings

### Pipeline-Only Dependencies (Can be Removed - ~500 MB)

Document Processing (removed after Step 2):
- `docling` (~100 MB) - PDF/HTML to markdown
- `html2text` (~10 MB) - HTML processing

Graph Community Detection (removed after Step 4):
- `igraph` (~50 MB) - Graph analysis
- `leidenalg` (~30 MB) - Community detection
- `python-louvain` (~20 MB) - Louvain algorithm
- `graspologic` (~40 MB) - Graph statistics

Optional/Development (removed if not needed):
- `milvus-lite` (~100 MB) - Not needed with cloud Zilliz
- `chainlit` - Not needed if using Flask only
- `tqdm` - Progress bars (nice to have)
- `ipykernel` - Jupyter (development only)
- `fastmcp` - MCP support (development only)

## Usage Instructions

### Option 1: Automatic Cleanup (Recommended)

Add to your `.env` file:
```bash
# Enable automatic cleanup after pipeline
CLEANUP_PIPELINE_DEPS=true

# Run pipeline on first deployment
AUTO_RUN_PIPELINE=true
```

Process:
1. First deployment: Pipeline runs → Cleanup → App starts (~300 MB RAM)
2. Subsequent restarts: App starts immediately (~300 MB RAM)
3. Redeploy: Pipeline runs again → Cleanup → App starts

### Option 2: Manual Cleanup

Run the cleanup script manually after pipeline:
```bash
chmod +x ./cleanup_pipeline_deps.sh
./cleanup_pipeline_deps.sh
```

### Option 3: Use requirements-runtime.txt

For production-only deployments (no pipeline):
```dockerfile
# In Dockerfile, replace:
RUN pip install -r requirements.txt

# With:
RUN pip install -r requirements-runtime.txt
```

## Cleanup Script Details

What Gets Removed:
```bash
# Document processing
pip uninstall -y docling html2text

# Graph community detection
pip uninstall -y python-louvain igraph leidenalg graspologic

# Development tools
pip uninstall -y tqdm ipykernel fastmcp

# Conditional removals
# - milvus-lite (if VECTOR_DB_TYPE=cloud_zilliz)
# - chainlit (if APP_TYPE=flask_graph or flask)
```

Estimated Savings: 350-500 MB

## Phase 2: Cloud Embeddings (Future Enhancement)

### Why Cloud Embeddings

Current Setup:
- Local Granite embedding model: ~500 MB
- Requires `torch` + `sentence-transformers`
- Model download + loading time on startup

Cloud Setup:
- API calls to embedding service: ~0 MB local
- No model download or loading
- Often better quality embeddings

### Implementation Plan

1. Add Configuration:
```python
# In my_config.py
MY_CONFIG.EMBEDDING_RUN_ENV = os.getenv("EMBEDDING_RUN_ENV", "local")
MY_CONFIG.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local/granite-embedding")
MY_CONFIG.EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
```

2. Update .env:
```bash
# Embedding Configuration
EMBEDDING_RUN_ENV=cloud              # local or cloud
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=your_openai_key_here
```

3. Additional Savings:
- Remove `torch` (~250 MB)
- Remove `sentence-transformers` (~100 MB)
- Total Phase 2 Savings: ~350 MB
- Combined Savings: ~700 MB total

### Cost Considerations

Cloud Embedding Costs (OpenAI):
- `text-embedding-3-small`: $0.00002 per 1K tokens
- Average document: ~500 tokens
- 100 documents: $0.001 (essentially free)
- 10,000 documents: $0.10

Tradeoff:
- API costs are negligible for most use cases
- Saves $13/month on server costs
- Net savings: Still significant

## Deployment Workflow

### First-Time Deployment

```bash
# 1. Set environment variables in .env
AUTO_RUN_PIPELINE=true
CLEANUP_PIPELINE_DEPS=true
WEBSITE_URL=https://example.com

# 2. Deploy to cloud
docker-compose -f docker-compose.cloud.yml up -d

# 3. Process (takes 20-30 minutes):
#    - Crawl website
#    - Process files
#    - Save to databases
#    - Build graph communities
#    - AUTO CLEANUP (~500 MB freed)
#    - Start Flask app

# 4. After successful deployment:
#    Change .env: AUTO_RUN_PIPELINE=false
#    Prevents re-running pipeline on restarts
```

### Subsequent Restarts

```bash
# App restarts use cached data
# No pipeline, no cleanup needed
# Fast startup (<30 seconds)
# Low RAM usage (~300 MB)
```

### Redeployment (New Content)

```bash
# Change .env: AUTO_RUN_PIPELINE=true
# Redeploy → Full pipeline runs → Auto cleanup → App starts
```

## Verification

### Check RAM Usage

Before cleanup:
```bash
docker stats
# CONTAINER    MEM USAGE / LIMIT    MEM %
# allycat      750MB / 1GB          75%
```

After cleanup:
```bash
docker stats
# CONTAINER    MEM USAGE / LIMIT    MEM %
# allycat      320MB / 1GB          32%
```

### Check Installed Packages

```bash
# Check if cleanup ran
docker exec -it <container> pip list | grep -E "docling|igraph|leidenalg"
# Should return nothing if cleanup succeeded
```

## Best Practices

### Recommended:
- Enable `CLEANUP_PIPELINE_DEPS=true` for production
- Set `AUTO_RUN_PIPELINE=false` after first successful run
- Use cloud Zilliz instead of Milvus Lite
- Monitor RAM usage with `docker stats`
- Test cleanup script locally before production

### Not Recommended:
- Run cleanup if you plan to re-run pipeline without redeployment
- Remove packages manually (use the script)
- Enable AUTO_RUN_PIPELINE permanently (costs API tokens)
- Skip monitoring on first deployment

## Troubleshooting

### Issue: OOM Error During Pipeline

Symptom: Container crashes during Phase 2 or 3

Solution:
```bash
# Temporarily use 2 GB RAM for first deployment
# After pipeline + cleanup, downgrade to 1 GB
```

### Issue: App Fails After Cleanup

Symptom: Import errors for missing packages

Solution:
```bash
# Check what failed, add to requirements-runtime.txt
# Redeploy with updated runtime requirements
```

### Issue: Pipeline Needs to Re-run

Symptom: Need to update content, but cleanup removed packages

Solution:
```bash
# Set AUTO_RUN_PIPELINE=true
# Redeploy → Full requirements install → Pipeline runs
```

## Future Optimizations

1. **Multi-stage Docker builds**
   - Build stage: Full dependencies
   - Runtime stage: Only runtime deps
   - Even smaller final image

2. **Cloud embeddings** (Phase 2)
   - Save additional ~350 MB
   - Better embedding quality
   - Faster startup

3. **Lazy loading**
   - Load graph modules only when needed
   - Further reduce initial RAM usage

4. **Caching strategies**
   - Cache LLM responses
   - Reduce API calls during queries


If you encounter issues:
1. Check `docker-startup.sh` logs for cleanup errors
2. Verify `.env` has correct settings
3. Run `docker stats` to monitor RAM
4. Review `cleanup_pipeline_deps.sh` output


This optimization reduces Docker RAM from 800 MB to 300 MB, enabling 1 GB deployments at 52% lower cost while maintaining full functionality.
